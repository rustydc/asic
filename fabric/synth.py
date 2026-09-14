"""Synthesize the fabric column datapath with yosys against an open liberty file.

Measures the standard-cell area of ``fabric_columns`` (everything in the tile
except the ROM macro) so the density model's MAC placeholders can be replaced
with numbers from a real, if old, process.  Works with a native ``yosys`` or
the ``yowasp-yosys`` PyPI package.

Example::

    python -m fabric.synth --liberty /path/sky130_fd_sc_hd__tt_025C_1v80.lib \
        --rows 256 --cols 8 --rows-per-cycle 2

The per-column area is independent of ``rows`` beyond the cycle counter, so a
shallow tile synthesizes quickly and reports the same column cost.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

RTL = Path(__file__).parent / "rtl" / "fabric_tile.sv"


@dataclass
class SynthResult:
    liberty: str
    rows: int
    cols: int
    rows_per_cycle: int
    cells: int
    area_um2: float
    area_per_column_um2: float
    flops: int
    abc_delay_ps: float | None      # ABC's mapped critical-path estimate, if reported
    log_excerpt: str

    def as_dict(self) -> dict:
        return asdict(self)


def yosys_command() -> list[str]:
    for candidate in ("yosys", "yowasp-yosys"):
        if shutil.which(candidate):
            return [candidate]
    raise FileNotFoundError("no yosys found; install yosys or `pip install yowasp-yosys`")


def synthesize(liberty: Path, *, rows: int, cols: int, rows_per_cycle: int, weight_bits: int = 4,
               act_bits: int = 8, acc_bits: int = 24, top: str = "fabric_columns",
               target_ps: int | None = None) -> SynthResult:
    liberty = liberty.resolve()
    params = {"ROWS": rows, "COLS": cols, "WB": weight_bits, "AB": act_bits, "P": rows_per_cycle, "ACC": acc_bits}
    chparam = " ".join(f"-set {name} {value}" for name, value in params.items())
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        # yowasp runs in a sandbox rooted at the working directory; copy inputs next to the script.
        shutil.copy(RTL, work / "fabric_tile.sv")
        shutil.copy(liberty, work / "cells.lib")
        script = "\n".join([
            "read_verilog -sv fabric_tile.sv",
            f"chparam {chparam} {top}",
            f"hierarchy -check -top {top}",
            f"synth -top {top} -flatten",
            "dfflibmap -liberty cells.lib",
            "abc -liberty cells.lib" + (f" -D {target_ps}" if target_ps else ""),
            "opt_clean",
            "stat -liberty cells.lib",
        ]) + "\n"
        (work / "synth.ys").write_text(script, encoding="utf-8")
        result = subprocess.run([*yosys_command(), "-q", "-l", "synth.log", "synth.ys"], cwd=work,
                                capture_output=True, text=True)
        log = (work / "synth.log").read_text(encoding="utf-8") if (work / "synth.log").exists() else result.stdout
        if result.returncode != 0:
            raise RuntimeError(f"yosys failed:\n{result.stderr[-4000:]}\n{log[-4000:]}")
    cells, area, flops = parse_stat(log)
    delays = re.findall(r"Delay\s*=\s*([0-9.]+)\s*ps", log)
    delay = float(delays[-1]) if delays else None
    tail = log[log.rfind("Printing statistics"):] if "Printing statistics" in log else log[-3000:]
    return SynthResult(str(liberty), rows, cols, rows_per_cycle, cells, area, area / cols, flops, delay, tail[-3000:])


def parse_stat(log: str) -> tuple[int, float, int]:
    section = log[log.rfind("Printing statistics"):] if "Printing statistics" in log else log
    cells_match = re.search(r"Number of cells:\s+(\d+)", section)
    area_match = re.search(r"Chip area for (?:top )?module.*?:\s+([0-9.]+)", section)
    if not cells_match or not area_match:
        raise ValueError(f"could not parse yosys stat output:\n{section[-2000:]}")
    flops = sum(int(n) for name, n in re.findall(r"\s+(\S*df\S*)\s+(\d+)", section) if "df" in name)
    return int(cells_match.group(1)), float(area_match.group(1)), flops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liberty", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows-per-cycle", type=int, default=2)
    parser.add_argument("--target-ps", type=int, default=None, help="ask ABC to map for this clock period")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args()
    result = synthesize(args.liberty, rows=args.rows, cols=args.cols, rows_per_cycle=args.rows_per_cycle,
                        target_ps=args.target_ps)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        delay = f", ABC delay {result.abc_delay_ps:.0f} ps" if result.abc_delay_ps else ""
        print(f"{Path(result.liberty).name}: rows={result.rows} cols={result.cols} P={result.rows_per_cycle}: "
              f"{result.cells} cells, {result.flops} flops, {result.area_um2:.0f} um2 total, "
              f"{result.area_per_column_um2:.0f} um2 per column{delay}")


if __name__ == "__main__":
    main()
