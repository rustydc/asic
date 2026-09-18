import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from hw import pinout
from hw.board import Board


class PinoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.board = Board.load()
        cls.pinout = pinout.derive(cls.board)

    def test_requirements_follow_the_power_model(self) -> None:
        need = self.pinout.requirements
        rules = self.board.data["package_selection"]
        self.assertAlmostEqual(need.core_amps, self.board.core_current_a("layer_asic", rules["rated_tokens_per_second"]))
        self.assertEqual(need.core_balls, -(-int(round(need.core_amps * 1000)) // int(rules["amps_per_ball"] * 1000)))
        self.assertEqual(need.signal_balls, 2 * 36 + 2 * 65 + 14)
        self.assertEqual(need.total, need.signal_balls + need.signal_grounds + 2 * need.core_balls + sum(need.rail_balls.values()))

    def test_smallest_sufficient_package_is_selected(self) -> None:
        # At the 50K tokens/s rating the core needs 354 balls each of power and ground;
        # only the 1225-ball 29 mm package carries that.  The 784-ball 23 mm package is
        # enough at the 14.5K design point, and 400 balls never are.
        self.assertEqual(self.pinout.package.name, "FCBGA1225_35x35_P0.8")
        self.assertTrue(any(name == "FCBGA784_28x28_P0.8" for name, _ in self.pinout.rejected))
        smaller = pinout.derive(self.board, rated_tokens_per_second=14_500)
        self.assertEqual(smaller.package.name, "FCBGA784_28x28_P0.8")
        self.assertTrue(any(name == "FCBGA400_20x20_P1.0" for name, _ in smaller.rejected))
        self.assertGreater(self.pinout.requirements.core_balls, 3 * smaller.requirements.core_balls)

    def test_ball_map_places_every_signal_once_in_the_outer_rows(self) -> None:
        p = self.pinout.package
        balls = self.pinout.balls
        self.assertEqual(len(balls), p.balls)
        self.assertEqual(len({b.name for b in balls}), p.balls)
        signals = [(b.interface, b.signal) for b in balls if b.kind == "signal"]
        self.assertEqual(len(signals), len(set(signals)))
        self.assertEqual(len(signals), self.pinout.requirements.signal_balls)
        link_in = [b for b in balls if b.interface == "link_in"]
        self.assertEqual(len(link_in), 36)
        self.assertEqual({b.col for b in link_in}, {0, 1})
        rows = sorted({b.row for b in link_in})
        self.assertEqual(rows, list(range(rows[0], rows[0] + 18)))
        self.assertLessEqual(abs(rows[0] + 8.5 - (p.rows - 1) / 2), 0.5)   # centred on the edge
        self.assertTrue(all(b.escape == ("W", b.col) for b in link_in))
        self.assertTrue(all(b.escape == ("E", p.cols - 1 - b.col) for b in balls if b.interface == "link_out"))
        memory = [b for b in balls if b.interface.startswith("lpddr")]
        self.assertEqual(len(memory), 130)
        self.assertTrue(all(b.row < rows[0] + 2 for b in memory))            # north rows
        misc = [b for b in balls if b.interface in ("mgmt", "jtag", "refclk", "strap")]
        self.assertTrue(all(b.row == p.rows - 1 for b in misc))              # south row
        rules = self.board.data["package_selection"]
        outer = [b for b in balls if b.kind == "signal" and b.interface.startswith("link")]
        self.assertTrue(all(min(b.col, p.cols - 1 - b.col) < rules["signal_rows"] for b in outer))

    def test_byte_lanes_are_followed_by_ground(self) -> None:
        by_pos = {(b.row, b.col): b for b in self.pinout.balls}
        ordered = [b for b in self.pinout.balls if b.interface == "lpddr_ch0"]
        lane0 = [b for b in ordered if b.signal in {f"DQ{k}" for k in range(8)} | {"DQS0_P", "DQS0_N", "DMI0"}]
        self.assertEqual(len(lane0), 11)
        last = max(lane0, key=lambda b: (b.row, b.col))
        after = by_pos.get((last.row, last.col + 1)) or by_pos.get((last.row + 1, 0))
        self.assertEqual(after.kind, "ground")

    def test_core_and_ground_counts_meet_the_requirement(self) -> None:
        need = self.pinout.requirements
        self.assertGreaterEqual(self.pinout.count("rail", "VDD_CORE"), need.core_balls)
        self.assertGreaterEqual(self.pinout.count("ground"), need.ground_balls + need.signal_grounds)
        for rail, count in need.rail_balls.items():
            self.assertEqual(self.pinout.count("rail", rail), count)

    def test_outputs_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            pinout.write_outputs(self.pinout, self.board, out)
            with (out / "asic_ballmap.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            data = json.loads((out / "asic_ballmap.json").read_text(encoding="utf-8"))
            report = (out / "report.md").read_text(encoding="utf-8")
        self.assertEqual(len(rows), self.pinout.package.balls)
        self.assertEqual(rows[0]["ball"], "A1")
        self.assertEqual(data["package"]["name"], self.pinout.package.name)
        self.assertEqual(sum(1 for b in data["balls"] if b["escape"]), 72)
        self.assertIn("**selected**", report)

    def test_in_package_memory_leaves_the_north_rows_to_power(self) -> None:
        board = Board.load(Path(__file__).resolve().parents[1] / "board_27b.yaml")
        p = pinout.derive(board)
        # Eight layer dies rather than four halve the per-die current, and the
        # package falls two sizes with it.
        self.assertEqual(p.package.name, "FCBGA1225_35x35_P0.8")
        self.assertEqual(p.requirements.signal_balls, 2 * 36 + 14)
        self.assertEqual(sum(1 for b in p.balls if b.interface.startswith("lpddr")), 0)
        self.assertGreaterEqual(p.count("rail", "VDD_CORE"), p.requirements.core_balls)
        self.assertEqual(p.count("rail", "VDD_HBM_1V1"), 24)
        self.assertIn("HBM", pinout.report_markdown(p, board))

    def test_psram_die_puts_both_link_ports_on_the_south_edge_and_memory_on_three(self) -> None:
        board = Board.load(Path(__file__).resolve().parents[1] / "board_psram.yaml")
        p = pinout.derive(board)
        self.assertEqual(pinout.link_rows(board), 6)
        self.assertEqual(pinout.memory_edges(board), ["N", "E", "W"])
        # 16 ports of 19 signals plus a ground each: 320 balls in the outer five rows of three edges.
        self.assertEqual(pinout.memory_balls_needed(board), 320)
        self.assertEqual(p.package.name, "FCBGA784_28x28_P0.8")
        self.assertEqual(p.requirements.signal_balls, 24 + 304 + 14 + 4)     # the four shared PSRAM clocks
        rows, cols = p.package.rows, p.package.cols
        link = [b for b in p.balls if b.interface in ("link_in", "link_out")]
        self.assertEqual(len(link), 24)
        self.assertTrue(all(b.row >= rows - 2 and b.escape[0] == "S" for b in link))
        # link_in sits west of link_out along the edge, the blocks a finger group apart.
        in_cols = [b.col for b in link if b.interface == "link_in"]
        out_cols = [b.col for b in link if b.interface == "link_out"]
        self.assertLess(max(in_cols), min(out_cols))
        self.assertGreaterEqual((min(out_cols) - min(in_cols)) * p.package.pitch, 12.0)
        memory = [b for b in p.balls if b.kind == "signal" and b.interface.startswith("psram_") and b.interface != "psram_clk"]
        self.assertEqual(len(memory), 304)
        clocks = [b for b in p.balls if b.interface == "psram_clk"]
        self.assertEqual(len(clocks), 4)
        self.assertTrue(all(b.row >= rows - 2 for b in clocks))              # with the small interfaces
        depth = [min(b.row, b.col, rows - 1 - b.row, cols - 1 - b.col) for b in memory]
        self.assertLessEqual(max(depth), 4)
        self.assertTrue(all(b.row >= rows - 2 for b in p.balls if b.interface in ("mgmt", "jtag", "refclk", "strap")))
        report = pinout.report_markdown(p, board)
        self.assertIn("S edge, beside it", report)
        # The single-board rule set is untouched: the 9B map still has 36-lane ports on W and E.
        self.assertEqual(pinout.link_rows(self.board), 18)
        self.assertEqual(pinout.memory_edges(self.board), ["N"])

    def test_memory_capacity_rule_rejects_a_package_whose_edges_are_too_short(self) -> None:
        board = Board.load(Path(__file__).resolve().parents[1] / "board_psram.yaml")
        # Memory on two edges instead of three: the 28x28 holds 230 memory balls in its outer
        # five rows there, the 35x35 300, both short of the 320 needed, so the 45x45 is taken.
        data = copy.deepcopy(board.data)
        data["package_selection"]["edges"]["memory"] = ["N", "E"]
        p = pinout.derive(Board(data))
        self.assertEqual(p.package.name, "FCBGA2025_45x45_P1.0")
        self.assertEqual(sum(1 for _, reason in p.rejected if "memory balls needed" in reason), 3)

    def test_no_candidate_is_an_error(self) -> None:
        data = copy.deepcopy(self.board.data)
        data["package_selection"]["candidates"] = data["package_selection"]["candidates"][:1]
        with self.assertRaises(ValueError):
            pinout.derive(Board(data))


if __name__ == "__main__":
    unittest.main()
