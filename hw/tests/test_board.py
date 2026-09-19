import copy
import unittest
from pathlib import Path

from hw.board import Board

HW = Path(__file__).resolve().parents[1]


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
        self.assertTrue(any("needs memory channels" in p for p in problems))
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

    def test_memory_devices_follow_the_memory_net_kind(self) -> None:
        self.assertEqual(self.board.memory_kind, "lpddr5x_x32")
        self.assertEqual(self.board.memory_interfaces("layer_asic"), ["lpddr_ch0", "lpddr_ch1"])
        self.assertEqual(self.board.memory_device_classes(), ["lpddr5x"])
        self.assertFalse(self.board.is_modular)
        hbm = Board.load(HW / "board_27b.yaml")
        self.assertEqual(hbm.memory_interfaces("layer_asic"), [])
        self.assertEqual(hbm.memory_device_classes(), [])

    def test_modular_board_puts_every_ring_chip_and_its_memory_on_a_card(self) -> None:
        board = Board.load(HW / "board_psram.yaml")
        self.assertEqual(board.check(), [])
        self.assertTrue(board.is_modular)
        self.assertEqual(board.memory_kind, "psram_x16")
        self.assertEqual(len(board.memory_interfaces("layer_asic")), 16)
        self.assertEqual(board.memory_device_classes(), ["psram"])
        self.assertEqual(len(board.instances("psram")), 128)
        self.assertEqual(sorted(int(spec["slot"]) for spec in board.modules.values()), list(range(10)))
        self.assertEqual(len(board.module_members("M0")), 17)          # one chip and sixteen devices
        self.assertEqual(len(board.module_members("M8")), 1)           # a head module carries no memory
        # Only the ring, the small interfaces and power cross the connector: the memory stays on the card.
        self.assertEqual(board.connector_signal_count("M0"), 2 * 12 + 5 + 5 + 2 + 2)
        self.assertEqual(board.connector_signal_count("M8"), board.connector_signal_count("M0"))
        self.assertIn("Modules", board.summary_markdown())
        svg = board.svg()
        self.assertIn("slot 9", svg)
        self.assertIn("38 signals over the edge", svg)

    def test_fpga_part_fits_its_interfaces(self) -> None:
        board = Board.load(HW / "board_psram.yaml")
        fit = {resource: (needed, available) for resource, needed, available in board.fpga_fit()}
        self.assertEqual(fit["HP I/O (1.2 V: DDR4)"], (124, 156))
        self.assertEqual(fit["HD I/O (1.8 V: links, management, configuration)"], (52, 72))
        self.assertEqual(fit["transceiver lanes (PCIe x8, SFP+)"], (9, 12))
        self.assertIn("| HP I/O (1.2 V: DDR4) | 124 | 156 |", board.summary_markdown())
        # A part with one transceiver quad too few is caught.
        data = copy.deepcopy(board.data)
        data["part_classes"]["fpga"]["resources"]["gth"] = 8
        self.assertTrue(any("transceiver lanes" in p and "needs 9" in p for p in Board(data).check()))
        # The single board has not chosen a part and reports no fit.
        self.assertEqual(self.board.fpga_fit(), [])

    def test_modular_checks_catch_a_device_on_the_wrong_card_and_a_shared_slot(self) -> None:
        board = Board.load(HW / "board_psram.yaml")
        data = copy.deepcopy(board.data)
        data["components"]["U_M0_3"]["module"] = "M1"
        problems = Board(data).check()
        self.assertTrue(any("not on the module of U_A0" in p for p in problems))
        data = copy.deepcopy(board.data)
        data["modules"]["M1"]["slot"] = 0
        self.assertTrue(any("both sit in slot 0" in p for p in Board(data).check()))
        data = copy.deepcopy(board.data)
        del data["components"]["U_H1"]["module"]
        self.assertTrue(any("every ring chip of a modular board" in p for p in Board(data).check()))
        data = copy.deepcopy(self.board.data)
        data["components"]["U_A0"]["module"] = "M0"
        self.assertTrue(any("names a module but the board has none" in p for p in Board(data).check()))

    def test_svg_and_summary_render(self) -> None:
        svg = self.board.svg()
        self.assertTrue(svg.startswith("<svg"))
        for ref in ("U_FPGA", "U_A7", "U_H1", "M0a"):
            self.assertIn(ref, svg)
        summary = self.board.summary_markdown()
        self.assertIn("Activation ring", summary)


if __name__ == "__main__":
    unittest.main()
