"""Open-tooling synthesis smoke test.

Runs only when a yosys is on the PATH (native or ``yowasp-yosys``) and the
``FABRIC_LIBERTY`` environment variable points at a liberty file, for example
sky130_fd_sc_hd__tt_025C_1v80.lib from the OpenROAD flow scripts.
"""

import os
import shutil
import unittest
from pathlib import Path

import tempfile

from fabric.pnr import PLATFORMS, parse_results
from fabric.sta import parse_report
from fabric.synth import filter_liberty, merge_liberty, nand2_area, parse_stat, synthesize

LIBERTY = os.environ.get("FABRIC_LIBERTY")
HAVE_YOSYS = shutil.which("yosys") or shutil.which("yowasp-yosys")

LIB_A = """library (a) {
  delay_model : table_lookup;
  cell (NAND2_X1) {
    area : 0.798;
    pin (A) { direction : input; }
  }
  cell (INV_X1) {
    area : 0.532;
  }
}
"""
LIB_B = """library (b) {
  cell (DFF_X1) {
    area : 4.522;
  }
}
"""


class ParseTest(unittest.TestCase):
    def test_merge_liberty_keeps_one_header_and_all_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "a.lib").write_text(LIB_A, encoding="utf-8")
            (work / "b.lib").write_text(LIB_B, encoding="utf-8")
            count = merge_liberty([work / "a.lib", work / "b.lib"], work / "m.lib")
            merged = (work / "m.lib").read_text(encoding="utf-8")
        self.assertEqual(count, 3)
        self.assertEqual(merged.count("library ("), 1)
        for cell in ("NAND2_X1", "INV_X1", "DFF_X1"):
            self.assertIn(f"cell ({cell})", merged)
        self.assertTrue(merged.rstrip().endswith("}"))

    def test_nand2_area_reads_the_smallest_nand(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.lib"
            path.write_text(LIB_A, encoding="utf-8")
            self.assertEqual(nand2_area(path), 0.798)

    def test_filter_liberty_removes_matching_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "a.lib"
            src.write_text(LIB_A.replace("INV_X1", "sky130_fd_sc_hd__lpflow_lsbuf_1"), encoding="utf-8")
            removed = filter_liberty(src, Path(directory) / "f.lib")
            text = (Path(directory) / "f.lib").read_text(encoding="utf-8")
        self.assertEqual(removed, 1)
        self.assertIn("cell (NAND2_X1)", text)
        self.assertNotIn("lpflow", text)

    def test_parse_pnr_reports_scale_library_units(self) -> None:
        log = "\n".join([
            "Startpoint: early", "   1.000   data arrival time", "   0.500   slack (MET)",
            "wns max 0.5", "[INFO GPL-0006] NumInstances: 21000",
            "Startpoint: a", "   2.950   data arrival time", "  -0.250   slack (VIOLATED)",
            "Total            580308        176889           30.48%             0 /  0 /  3",
            "[INFO GRT-0018] Total wirelength: 131319 um",
            "   0.120 skew", "Design area 123456 u^2 47% utilization.",
            "wns max -0.25", "tns max -12.5", ""])
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "openroad.log").write_text(log, encoding="utf-8")
            result = parse_results(work, PLATFORMS["sky130hd"], 3000.0, detailed_route=False)
        self.assertEqual(result.instances, 21000)
        self.assertAlmostEqual(result.worst_slack_ps, -250.0)          # sky130 reports in ns
        self.assertAlmostEqual(result.tns_ps, -12500.0)
        self.assertAlmostEqual(result.critical_path_ps, 2950.0)
        self.assertAlmostEqual(result.design_area_um2, 123456.0)
        self.assertAlmostEqual(result.utilization_pct, 47.0)
        self.assertAlmostEqual(result.clock_skew_ps, 120.0)
        self.assertAlmostEqual(result.max_frequency_mhz, 1e6 / 3250.0)
        self.assertAlmostEqual(result.wirelength_um, 131319.0)
        self.assertEqual(result.overflow, 3)
        self.assertEqual(result.stage, "global_route")

    def test_parse_pnr_reports_older_report_formats(self) -> None:
        # Older OpenROAD builds print "wns X", a skew table, and no instance count;
        # the count then comes from the written DEF.
        log = "\n".join([
            "[INFO GPL-0006] NumInstances: 21449",
            "Startpoint: a", "  904.505   data arrival time", "  -2.689   slack (VIOLATED)",
            "Clock clk", "Latency      CRPR       Skew", "_40822_/CLK ^", " 288.74", "_40061_/CLK ^",
            " 215.26      0.00      73.48", "Design area 2623 u^2 45% utilization.",
            "wns -2.69", "tns -39.46", ""])
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "openroad.log").write_text(log, encoding="utf-8")
            (work / "design.def").write_text("DESIGN x ;\nCOMPONENTS 23621 ;\n", encoding="utf-8")
            result = parse_results(work, PLATFORMS["asap7"], 700.0, detailed_route=False)
        self.assertEqual(result.instances, 23621)
        self.assertAlmostEqual(result.worst_slack_ps, -2.69)             # ASAP7 reports in ps
        self.assertAlmostEqual(result.tns_ps, -39.46)
        self.assertAlmostEqual(result.critical_path_ps, 904.505)
        self.assertAlmostEqual(result.clock_skew_ps, 73.48)
        self.assertAlmostEqual(result.max_frequency_mhz, 1e6 / 702.69)
        self.assertTrue(result.wirelength_um != result.wirelength_um)   # NaN when absent
        self.assertEqual(result.overflow, -1)

    def test_parse_sta_report_reads_clock_group_slack(self) -> None:
        report = """
Startpoint: rst_n (input port clocked by clk)
Endpoint: _1_ (recovery check against rising-edge clock clk)
Path Group: asynchronous
                                     2.478   slack (MET)
Startpoint: _36191_ (rising edge-triggered flip-flop clocked by clk)
Endpoint: _35376_ (rising edge-triggered flip-flop clocked by clk)
Path Group: clk
                                  4383.052   data arrival time
                                  2441.106   data required time
                                  -1941.946   slack (VIOLATED)
"""
        result = parse_report(report, 2500.0)
        self.assertEqual(result.startpoint, "_36191_")
        self.assertEqual(result.endpoint, "_35376_")
        self.assertAlmostEqual(result.worst_slack_ps, -1941.946)
        self.assertAlmostEqual(result.critical_path_ps, 4383.052)
        self.assertAlmostEqual(result.max_frequency_mhz, 1e6 / (2500.0 + 1941.946))

    def test_parse_stat_reads_cells_area_and_flops(self) -> None:
        log = """
Printing statistics.

=== fabric_columns ===

   Number of cells:               1234
     sky130_fd_sc_hd__dfrtp_1       48
     sky130_fd_sc_hd__nand2_1      600

   Chip area for module '\\fabric_columns': 5678.900000
"""
        self.assertEqual(parse_stat(log), (1234, 5678.9, 48))


@unittest.skipUnless(HAVE_YOSYS and LIBERTY and Path(LIBERTY).exists(), "yosys or FABRIC_LIBERTY not available")
class SynthesisTest(unittest.TestCase):
    def test_column_datapath_synthesizes(self) -> None:
        result = synthesize(Path(LIBERTY), rows=64, cols=4, rows_per_cycle=1)
        self.assertGreater(result.cells, 100)
        self.assertGreater(result.area_um2, 0)
        # 4 columns x 24-bit accumulators plus the 4 x 8-bit output register and control.
        self.assertGreaterEqual(result.flops, 4 * 24 + 4 * 8)


if __name__ == "__main__":
    unittest.main()
