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
from typing import Sequence

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


def merge_liberty(paths: Sequence[Path], output: Path) -> int:
    """Concatenate the cell groups of several liberty files into one library block.

    ASAP7 ships its cells across several files (SIMPLE, INVBUF, AO, OA, SEQ);
    yosys-abc wants one library, so this does what the OpenROAD flow's
    mergeLib step does: keep the first file's header, append every ``cell``
    group, close the block.  Returns the number of cells written.
    """
    texts = [Path(p).read_text(encoding="utf-8", errors="ignore") for p in paths]

    def first_cell(text: str) -> int:
        match = re.search(r"\n\s*cell\s*\(", text)
        return match.start() + 1 if match else len(text)

    header = texts[0][: first_cell(texts[0])]
    bodies = []
    for text in texts:
        body = text[first_cell(text):]
        bodies.append(body[: body.rstrip().rfind("}")])   # drop the library's closing brace
    merged = header + "".join(bodies) + "\n}\n"
    Path(output).write_text(merged, encoding="utf-8")
    return len(re.findall(r"\n\s*cell\s*\(", merged))


def nand2_area(liberty: Path, names: Sequence[str] = ("NAND2_X1", "sky130_fd_sc_hd__nand2_1", "sg13g2_nand2_1",
                                                      "NAND2xp5_ASAP7_75t_R")) -> float | None:
    """Area of the library's smallest 2-input NAND, for cross-library scaling."""
    text = Path(liberty).read_text(encoding="utf-8", errors="ignore")
    for name in names:
        match = re.search(r"cell\s*\(\s*\"?%s\"?\s*\)\s*\{(.*?)\n\s*cell\s*\(" % re.escape(name), text, re.S)
        block = match.group(1) if match else ""
        area = re.search(r"\barea\s*:\s*([0-9.]+)", block)
        if area:
            return float(area.group(1))
    return None


def yosys_command() -> list[str]:
    for candidate in ("yosys", "yowasp-yosys"):
        if shutil.which(candidate):
            return [candidate]
    raise FileNotFoundError("no yosys found; install yosys or `pip install yowasp-yosys`")


def synthesize(liberty: Path | Sequence[Path], *, rows: int, cols: int, rows_per_cycle: int, weight_bits: int = 4,
               act_bits: int = 8, acc_bits: int = 24, top: str = "fabric_columns",
               target_ps: int | None = None) -> SynthResult:
    """Synthesize ``top`` against one liberty file, or several (e.g. ASAP7 splits cells across files)."""
    liberties = [Path(p).resolve() for p in ([liberty] if isinstance(liberty, (str, Path)) else liberty)]
    params = {"ROWS": rows, "COLS": cols, "WB": weight_bits, "AB": act_bits, "P": rows_per_cycle, "ACC": acc_bits}
    chparam = " ".join(f"-set {name} {value}" for name, value in params.items())
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        # yowasp runs in a sandbox rooted at the working directory; copy inputs next to the script.
        shutil.copy(RTL, work / "fabric_tile.sv")
        names = []
        for index, path in enumerate(liberties):
            name = f"cells{index}.lib"
            shutil.copy(path, work / name)
            names.append(name)
        lib_args = " ".join(f"-liberty {name}" for name in names)
        script = "\n".join([
            "read_verilog -sv fabric_tile.sv",
            f"chparam {chparam} {top}",
            f"hierarchy -check -top {top}",
            f"synth -top {top} -flatten",
            f"dfflibmap {lib_args}",
            f"abc {lib_args}" + (f" -D {target_ps}" if target_ps else ""),
            "opt_clean",
            f"stat {lib_args}",
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
    label = "+".join(path.name for path in liberties)
    return SynthResult(label, rows, cols, rows_per_cycle, cells, area, area / cols, flops, delay, tail[-3000:])


def parse_stat(log: str) -> tuple[int, float, int]:
    section = log[log.rfind("Printing statistics"):] if "Printing statistics" in log else log
    cells_match = re.search(r"Number of cells:\s+(\d+)", section)
    area_match = re.search(r"Chip area for (?:top )?module.*?:\s+([0-9.]+)", section)
    if not cells_match or not area_match:
        raise ValueError(f"could not parse yosys stat output:\n{section[-2000:]}")
    flops = sum(int(n) for name, n in re.findall(r"\s+(\S*df\S*)\s+(\d+)", section, flags=re.IGNORECASE))
    return int(cells_match.group(1)), float(area_match.group(1)), flops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liberty", type=Path, required=True, action="append",
                        help="liberty file; repeat for libraries split across files")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows-per-cycle", type=int, default=2)
    parser.add_argument("--target-ps", type=int, default=None, help="ask ABC to map for this clock period")
    parser.add_argument("--merge", type=Path, default=None,
                        help="write the given liberty files merged into this one file and exit")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args()
    if args.merge:
        print(f"wrote {args.merge} with {merge_liberty(args.liberty, args.merge)} cells")
        return
    result = synthesize(args.liberty, rows=args.rows, cols=args.cols, rows_per_cycle=args.rows_per_cycle,
                        target_ps=args.target_ps)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        delay = f", ABC delay {result.abc_delay_ps:.0f} ps" if result.abc_delay_ps else ""
        nand2 = nand2_area(args.liberty[0])
        equiv = f", {result.area_per_column_um2 / nand2:.0f} NAND2-eq per column" if nand2 else ""
        print(f"{result.liberty}: rows={result.rows} cols={result.cols} P={result.rows_per_cycle}: "
              f"{result.cells} cells, {result.flops} flops, {result.area_um2:.0f} um2 total, "
              f"{result.area_per_column_um2:.0f} um2 per column{equiv}{delay}")


if __name__ == "__main__":
    main()
