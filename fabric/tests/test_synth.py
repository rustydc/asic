"""Open-tooling synthesis smoke test.

Runs only when a yosys is on the PATH (native or ``yowasp-yosys``) and the
``FABRIC_LIBERTY`` environment variable points at a liberty file, for example
sky130_fd_sc_hd__tt_025C_1v80.lib from the OpenROAD flow scripts.
"""

import os
import shutil
import unittest
from pathlib import Path

from fabric.synth import parse_stat, synthesize

LIBERTY = os.environ.get("FABRIC_LIBERTY")
HAVE_YOSYS = shutil.which("yosys") or shutil.which("yowasp-yosys")


class ParseTest(unittest.TestCase):
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
