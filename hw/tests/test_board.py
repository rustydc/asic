import copy
import unittest

from hw.board import Board


class BoardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.board = Board.load()

    def test_board_description_is_consistent(self) -> None:
        self.assertEqual(self.board.check(), [])

    def test_population(self) -> None:
        self.assertEqual(len(self.board.instances("layer_asic")), 8)
        self.assertEqual(len(self.board.instances("head_asic")), 2)
        self.assertEqual(len(self.board.instances("lpddr5x")), 16)
        self.assertEqual(len(self.board.instances("fpga")), 1)

    def test_ring_visits_layers_then_heads_and_closes(self) -> None:
        hops = self.board.ring()
        path = [hops[0][0].component] + [hop[1].component for hop in hops]
        self.assertEqual(path[0], "U_FPGA")
        self.assertEqual(path[-1], "U_FPGA")
        self.assertEqual(path[1:9], [f"U_A{i}" for i in range(8)])
        self.assertEqual(path[9:11], ["U_H0", "U_H1"])

    def test_power_fits_input(self) -> None:
        _, input_w = self.board.power_budget_w()
        self.assertLess(input_w, self.board.power_available_w())

    def test_asic_power_follows_the_energy_model(self) -> None:
        model = self.board.power_model
        spec = self.board.classes["layer_asic"]
        tps = 10_000.0
        expected = spec["static_w"] + tps * (float(spec["macs_per_token"]) * model["mac_energy_pj"]
                                             + model["memory_bytes_per_token_per_asic"] * model["memory_energy_pj_per_byte"]) * 1e-12
        self.assertAlmostEqual(self.board.part_power_w("layer_asic", tps), expected)
        self.assertAlmostEqual(self.board.part_power_w("head_asic", 0.0), self.board.classes["head_asic"]["static_w"])
        self.assertEqual(self.board.part_power_w("fpga", tps), self.board.classes["fpga"]["tdp_w"])
        low, _ = self.board.power_budget_w(5_000)
        high, _ = self.board.power_budget_w(50_000)
        self.assertGreater(high, 2 * low)
        self.assertGreater(self.board.core_current_a("layer_asic", 50_000), 100)

    def test_checks_catch_power_over_input(self) -> None:
        data = copy.deepcopy(self.board.data)
        data["power_model"]["design_tokens_per_second"] = 100_000
        problems = Board(data).check()
        self.assertTrue(any("power budget" in p for p in problems))

    def test_checks_catch_missing_memory(self) -> None:
        data = copy.deepcopy(self.board.data)
        data["nets"]["memory"]["channels"].pop()
        problems = Board(data).check()
        self.assertTrue(any("needs two LPDDR channels" in p for p in problems))
        self.assertTrue(any("unattached memory" in p for p in problems))

    def test_checks_catch_broken_ring(self) -> None:
        data = copy.deepcopy(self.board.data)
        data["nets"]["ring"]["hops"][3] = ["U_A2.link_out", "U_A5.link_in"]
        problems = Board(data).check()
        self.assertTrue(any("ring breaks" in p for p in problems))

    def test_checks_catch_memory_on_head(self) -> None:
        data = copy.deepcopy(self.board.data)
        data["nets"]["memory"]["channels"][0] = ["U_H0.link_in", "U_M0a.channel"]
        problems = Board(data).check()
        self.assertTrue(any("head ASIC" in p or "is link" in p for p in problems))

    def test_svg_and_summary_render(self) -> None:
        svg = self.board.svg()
        self.assertTrue(svg.startswith("<svg"))
        for ref in ("U_FPGA", "U_A7", "U_H1", "M0a"):
            self.assertIn(ref, svg)
        summary = self.board.summary_markdown()
        self.assertIn("Activation ring", summary)


if __name__ == "__main__":
    unittest.main()
