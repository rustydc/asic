import re
import tempfile
import unittest
from pathlib import Path

from hw.board import Board
from hw import kicad_gen as kg


def balanced(text: str) -> bool:
    depth = 0
    in_string = False
    for ch in text:
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return False
    return depth == 0 and not in_string


class KicadGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.board = Board.load()
        cls.design = kg.build_design(cls.board)

    def test_every_component_is_placed_on_the_card(self) -> None:
        refs = {part.ref for part in self.design.parts}
        for ref in self.board.components:
            self.assertIn(ref, refs)
        for part in self.design.parts:
            if part.body_w == 0:
                continue
            bw, bh = (part.body_h, part.body_w) if part.rotation in (90, 270) else (part.body_w, part.body_h)
            self.assertGreaterEqual(part.x - bw / 2, 0.0, part.ref)
            self.assertLessEqual(part.x + bw / 2, kg.CARD_LENGTH, part.ref)
            self.assertGreaterEqual(part.y - bh / 2, 0.0, part.ref)
            self.assertLessEqual(part.y + bh / 2, kg.CARD_HEIGHT, part.ref)

    def test_no_two_bodies_overlap(self) -> None:
        boxes = []
        for part in self.design.parts:
            if part.body_w == 0:
                continue
            bw, bh = (part.body_h, part.body_w) if part.rotation in (90, 270) else (part.body_w, part.body_h)
            boxes.append((part.ref, part.x - bw / 2, part.y - bh / 2, part.x + bw / 2, part.y + bh / 2))
        for i, (ra, ax0, ay0, ax1, ay1) in enumerate(boxes):
            for rb, bx0, by0, bx1, by1 in boxes[i + 1:]:
                overlap = ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1
                self.assertFalse(overlap, f"{ra} overlaps {rb}")

    def test_ring_is_fully_routed_within_the_link_limit(self) -> None:
        hops = kg.ring_hops(self.board)
        self.assertEqual(len(self.design.hop_lengths), len(hops))
        signals = kg.link_signals(self.board)
        for hop, (lo, hi) in self.design.hop_lengths.items():
            self.assertLessEqual(hi, kg.MAX_LINK_MM, hop)
            self.assertGreater(lo, 0.0)
        for h in range(len(hops)):
            routed = {t.net for t in self.design.tracks if t.net.startswith(f"LINK{h}_") and t.layer != "F.Cu"}
            self.assertEqual(routed, {kg.ring_net(h, s) for s in signals}, f"hop {h}")

    def test_asics_share_one_ball_map(self) -> None:
        asics = [p for p in self.design.parts if p.part_class in ("layer_asic", "head_asic")]
        reference = {pad.name: (pad.x, pad.y, pad.escape, (pad.net or "").split("_")[0][:4]) for pad in asics[0].pads}
        for part in asics[1:]:
            for pad in part.pads:
                x, y, escape, kind = reference[pad.name]
                self.assertEqual((pad.x, pad.y, pad.escape), (x, y, escape), f"{part.ref}.{pad.name}")
        # Link balls carry ring nets, and the same ball is the same lane index on every ASIC.
        lane = {}
        for part in asics:
            for pad in part.pads:
                if pad.escape:
                    index = int(re.sub(r"\D", "", pad.net.split("_", 1)[1]) or 0)
                    lane.setdefault(pad.name, index)
                    self.assertEqual(lane[pad.name], index, f"{part.ref}.{pad.name}")

    def test_fpga_link_pins_are_assigned_by_the_router(self) -> None:
        fpga = self.design.part("U_FPGA")
        link_pads = [pad for pad in fpga.pads if pad.escape]
        self.assertEqual(len(link_pads), 72)
        self.assertTrue(all(pad.net and pad.net.startswith("LINK") for pad in link_pads))

    def test_memory_channels_connect_asic_and_device(self) -> None:
        nets = {}
        for part in self.design.parts:
            for pad in part.pads:
                if pad.net and "_CH" in pad.net:
                    nets.setdefault(pad.net, set()).add(part.ref)
        self.assertGreater(len(nets), 16 * 60)
        for net, refs in nets.items():
            self.assertEqual(len(refs), 2, net)

    def test_core_current_report_flags_multiple_planes(self) -> None:
        rows = kg.core_current_density(self.design)
        self.assertEqual(len(rows), 8)
        for _, amps, _, density in rows:
            self.assertGreater(amps, 100)
            self.assertGreater(density, 30)

    def test_files_are_well_formed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            kg.generate(self.board, out)
            for name in (f"{kg.PROJECT}.kicad_pcb", f"{kg.PROJECT}.kicad_sch", "asics.kicad_sch", "memory.kicad_sch",
                         "fpga.kicad_sch", "power.kicad_sch"):
                text = (out / name).read_text(encoding="utf-8")
                self.assertTrue(balanced(text), name)
            pcb = (out / f"{kg.PROJECT}.kicad_pcb").read_text(encoding="utf-8")
            self.assertEqual(pcb.count("(footprint "), len(self.design.parts))
            self.assertIn('(layer "Edge.Cuts")', pcb)
            self.assertIn("(zone ", pcb)
            report = (out / "report.md").read_text(encoding="utf-8")
            self.assertIn("Activation ring", report)
            self.assertIn("| U_H1 -> U_FPGA |", report)
            self.assertTrue((out / "floorplan.svg").read_text(encoding="utf-8").startswith("<svg"))


@unittest.skipUnless(__import__("importlib").util.find_spec("pcbnew"), "KiCad's pcbnew module not installed")
class KicadDrcTest(unittest.TestCase):
    def test_board_passes_drc_except_unrouted_nets(self) -> None:
        import pcbnew
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            kg.generate(Board.load(), out)
            board = pcbnew.LoadBoard(str(out / f"{kg.PROJECT}.kicad_pcb"))
            pcbnew.ZONE_FILLER(board).Fill(board.Zones())
            pcbnew.WriteDRCReport(board, str(out / "drc.rpt"), pcbnew.EDA_UNITS_MILLIMETRES, False)
            kinds = set(re.findall(r"^\[(\w+)\]", (out / "drc.rpt").read_text(encoding="utf-8"), re.M))
        self.assertTrue(kinds <= {"unconnected_items"}, kinds)


if __name__ == "__main__":
    unittest.main()
