import math
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

    def test_every_component_is_placed_on_the_board(self) -> None:
        refs = {part.ref for part in self.design.parts}
        for ref in self.board.components:
            self.assertIn(ref, refs)
        ff = self.design.form_factor
        for part in self.design.parts:
            if part.body_w == 0:
                continue
            for x, y in part.corners():
                self.assertGreaterEqual(x, 0.0, part.ref)
                self.assertLessEqual(x, ff.width, part.ref)
                self.assertGreaterEqual(y, 0.0, part.ref)
                self.assertLessEqual(y, ff.depth, part.ref)

    def test_ring_is_a_regular_polygon_with_memories_out_and_regulators_in(self) -> None:
        lay = self.design.layout
        refs = [source.component for source, _ in self.board.ring()]
        self.assertEqual(len(lay.nodes), len(refs))
        self.assertEqual(refs[0], "U_FPGA")
        cx, cy = lay.centre
        for k, ref in enumerate(refs):
            part = self.design.part(ref)
            self.assertAlmostEqual(math.dist((part.x, part.y), (cx, cy)), lay.radius, places=6, msg=ref)
            # Tangential: the chip's local north points away from the centre.
            nx, ny = part.direction(90.0)
            self.assertAlmostEqual(nx * (cx - part.x) + ny * (cy - part.y), -lay.radius, places=6, msg=ref)
            nxt = self.design.part(refs[(k + 1) % len(refs)])
            self.assertAlmostEqual(math.dist((part.x, part.y), (nxt.x, nxt.y)), lay.side, places=6, msg=ref)
            # Clockwise: the next chip is to the right of this one's link-out direction.
            ox, oy = part.direction(0.0)
            self.assertLess(ox * (nxt.y - part.y) - oy * (nxt.x - part.x), 0.0, ref)
        for ref in self.board.instances("layer_asic"):
            asic = self.design.part(ref)
            for device in [b for a, b in self.board.nets["memory"]["channels"] if a.startswith(ref + ".")]:
                mem = self.design.part(device.split(".")[0])
                self.assertGreater(math.dist((mem.x, mem.y), (cx, cy)), lay.radius + 10.0, device)
                self.assertAlmostEqual((mem.rotation - asic.rotation) % 360.0, 90.0, places=6)
        for ref in self.board.instances("layer_asic") + self.board.instances("head_asic"):
            asic = self.design.part(ref)
            vrm = self.design.part(f"VRM_{ref[2:]}")
            self.assertLess(math.dist((vrm.x, vrm.y), (cx, cy)), lay.radius - 10.0, ref)
            self.assertAlmostEqual(vrm.rotation, asic.rotation, places=6)

    def test_chassis_keepouts_hold_only_their_connectors(self) -> None:
        ff = self.design.form_factor
        for part in self.design.parts:
            if part.body_w == 0 or part.part_class in kg.KEEPOUT_RESIDENTS:
                continue
            for name, (kx0, ky0, kx1, ky1) in ff.keepouts:
                self.assertFalse(kg.polygons_overlap(part.corners(), kg.rect(kx0, ky0, kx1, ky1)), f"{part.ref} in {name}")
        # The connectors sit on the edges they serve.
        self.assertLess(self.design.part("J_HOST").y, 12.0)
        self.assertTrue(all(self.design.part(f"J_FAN{k}").y > 340.0 for k in range(6)))
        self.assertTrue(all(self.design.part(f"J_PSU{k}").x > 273.0 for k in range(2)))
        with self.assertRaises(ValueError):
            moved = kg.build_design(self.board)
            moved.part("U_BMC").x = 350.0
            kg.check_fit(moved)

    def test_no_two_bodies_overlap(self) -> None:
        boxed = [p for p in self.design.parts if p.body_w > 0]
        for i, a in enumerate(boxed):
            for b in boxed[i + 1:]:
                self.assertFalse(kg.polygons_overlap(a.corners(), b.corners()), f"{a.ref} overlaps {b.ref}")
        # The oriented test is real: a rotated chip and its own memory would fail an axis-aligned box test.
        asic = self.design.part("U_A1")
        mem = self.design.part("U_M1a")
        ax, ay = asic.extent()
        mx, my = mem.extent()
        self.assertTrue(abs(asic.x - mem.x) < (ax + mx) / 2 and abs(asic.y - mem.y) < (ay + my) / 2)

    def test_27b_variant_is_a_hexagon_of_hbm_packages_without_board_memory(self) -> None:
        board = Board.load(Path(__file__).resolve().parents[1] / "board_27b.yaml")
        self.assertEqual(board.check(), [])
        design = kg.build_design(board)
        lay = design.layout
        self.assertEqual(len(lay.nodes), 6)
        self.assertEqual(design.pinout.package.name, "FCBGA3025_55x55_P1.0")
        self.assertEqual([p.ref for p in design.parts if p.part_class == "lpddr5x"], [])
        self.assertEqual(len([p for p in design.parts if p.part_class == "head_asic"]), 1)
        self.assertTrue(all(hi <= kg.MAX_LINK_MM for _, hi in design.hop_lengths.values()), design.hop_lengths)
        self.assertLessEqual(max(design.hop_bends.values()), kg.MAX_BEND_DEG)
        # Every shared rail of the power tree has a regulator and no ASIC rail is left without a source.
        regulators = {pad.net for p in design.parts if p.part_class == "regulator" for pad in p.pads}
        for rail, spec in board.data["power_tree"]["rails"].items():
            if spec.get("shared") or spec.get("per") == ["fpga"]:
                self.assertIn(rail, regulators, rail)
        boxed = [p for p in design.parts if p.body_w > 0]
        for i, a in enumerate(boxed):
            for b in boxed[i + 1:]:
                self.assertFalse(kg.polygons_overlap(a.corners(), b.corners()), f"{a.ref} overlaps {b.ref}")

    def test_hops_bend_gently_and_the_ribbon_stays_on_one_layer(self) -> None:
        for hop, bend in self.design.hop_bends.items():
            self.assertLessEqual(bend, 25.0, hop)
        layers = {t.layer for t in self.design.tracks if t.net.startswith("LINK")} - {"F.Cu"}
        self.assertEqual(layers, {kg.LINK_LAYER})
        hops = list(self.design.hop_lengths.values())
        self.assertLess(max(hi for _, hi in hops), 45.0)          # every hop is short
        self.assertLess(max(hi for _, hi in hops) - min(hi for _, hi in hops), 3.0)   # and alike

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
