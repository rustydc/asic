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
    abc_delay_ps: float | None      # ABC stime critical path after buffering, no wire load (timing mode only)
    critical_path: str | None       # ABC's reported start-point -> end-point
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

    def template_blocks(header: str) -> dict[str, str]:
        """Top-level ``*_template (name) { ... }`` groups, keyed by kind and name."""
        blocks = {}
        for match in re.finditer(r"\n(\s*(\w+_template)\s*\(\s*\"?([\w.]+)\"?\s*\)\s*\{)", header):
            start = match.start(1)
            depth, i = 0, header.index("{", start)
            while True:
                if header[i] == "{":
                    depth += 1
                elif header[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            blocks[(match.group(2), match.group(3))] = header[start:i + 1]
        return blocks

    header = texts[0][: first_cell(texts[0])]
    templates = template_blocks(header)
    extra = []
    for text in texts[1:]:
        for key, block in template_blocks(text[: first_cell(text)]).items():
            if key not in templates:
                templates[key] = block
                extra.append(block)
    bodies = []
    for text in texts:
        body = text[first_cell(text):]
        bodies.append(body[: body.rstrip().rfind("}")])   # drop the library's closing brace
    merged = header + "".join("\n" + block + "\n" for block in extra) + "".join(bodies) + "\n}\n"
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
    """A native yosys first (its ABC mapped arithmetic better in our runs), else the yowasp build."""
    for candidate in ("yosys", "yowasp-yosys"):
        if shutil.which(candidate):
            return [candidate]
    raise FileNotFoundError("no yosys found; install yosys or `pip install yowasp-yosys`")


# Cells the OpenROAD flow scripts mark dont_use for each open platform: sky130's
# low-power/level-shift and probe cells; ASAP7's weakest drive strengths, scan
# flops and clock gates.  Filtering them out of the liberty works with any yosys.
DONT_USE_PATTERNS = ("*lpflow*", "*probe*", "*x1p*_ASAP7*", "*xp*_ASAP7*", "SDF*", "ICG*")


def filter_liberty(source: Path, output: Path, patterns: Sequence[str] = DONT_USE_PATTERNS) -> int:
    """Copy a liberty file without the cell groups whose names match any glob pattern.

    Works with any yosys version, unlike ``abc -dont_use``.  Returns the number
    of cells removed.
    """
    import fnmatch
    text = Path(source).read_text(encoding="utf-8", errors="ignore")
    out = []
    pos = 0
    removed = 0
    for match in re.finditer(r"\n(\s*cell\s*\(\s*\"?([\w.]+)\"?\s*\)\s*\{)", text):
        start = match.start(1)
        if start < pos:
            continue
        depth, i = 0, text.index("{", start)
        while True:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        name = match.group(2)
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            out.append(text[pos:start])
            pos = i + 1
            removed += 1
    out.append(text[pos:])
    Path(output).write_text("".join(out), encoding="utf-8")
    return removed


_DONT_USE_SUPPORT: dict[str, bool] = {}


def yosys_supports_dont_use(command: list[str]) -> bool:
    key = command[0]
    if key not in _DONT_USE_SUPPORT:
        result = subprocess.run([*command, "-p", "help abc"], capture_output=True, text=True)
        _DONT_USE_SUPPORT[key] = "-dont_use" in result.stdout
    return _DONT_USE_SUPPORT[key]


def synthesize(liberty: Path | Sequence[Path], *, rows: int, cols: int, rows_per_cycle: int, weight_bits: int = 4,
               act_bits: int = 8, acc_bits: int = 24, top: str = "fabric_columns",
               target_ps: int | None = None, keep_netlist: Path | None = None) -> SynthResult:
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
            # Drop the cells a real flow marks dont_use, which ABC otherwise picks as slow buffers.
            filter_liberty(path, work / name)
            names.append(name)
        lib_args = " ".join(f"-liberty {name}" for name in names)
        command = yosys_command()
        dont_use = ""
        if target_ps:
            # Timing-driven mapping plus ABC's own static timing (buffered, no wire load).
            # No upsize/dnsize: ABC aborts when the target is unreachable, and sizing is a
            # place-and-route job anyway.
            (work / "abc.script").write_text("\n".join([
                "strash", "dch -f", f"map -D {target_ps}", "buffer -p",
                "topo", "stime -p", "print_stats -m"]) + "\n", encoding="utf-8")
            abc_cmd = f"abc {lib_args} {dont_use} -script abc.script"
        else:
            abc_cmd = f"abc {lib_args} {dont_use}"
        script = "\n".join([
            "read_verilog -sv fabric_tile.sv",
            f"chparam {chparam} {top}",
            f"hierarchy -check -top {top}",
            f"synth -top {top} -flatten",
            f"dfflibmap {lib_args}",
            abc_cmd,
            "opt_clean",
            "write_verilog -noattr netlist.v",
            f"stat {lib_args}",
        ]) + "\n"
        (work / "synth.ys").write_text(script, encoding="utf-8")
        result = subprocess.run([*command, "-q", "-l", "synth.log", "synth.ys"], cwd=work,
                                capture_output=True, text=True)
        log = (work / "synth.log").read_text(encoding="utf-8") if (work / "synth.log").exists() else result.stdout
        if result.returncode != 0 and target_ps and "abc.script" in script:
            # ABC's buffer/stime script aborts on some libraries (ASAP7 in our runs);
            # fall back to plain timing-driven mapping so OpenSTA can still time it.
            script = script.replace(abc_cmd, f"abc {lib_args} -D {target_ps}")
            (work / "synth.ys").write_text(script, encoding="utf-8")
            result = subprocess.run([*command, "-q", "-l", "synth.log", "synth.ys"], cwd=work,
                                    capture_output=True, text=True)
            log = (work / "synth.log").read_text(encoding="utf-8") if (work / "synth.log").exists() else result.stdout
        if result.returncode != 0:
            raise RuntimeError(f"yosys failed:\n{result.stderr[-4000:]}\n{log[-4000:]}")
        if keep_netlist and (work / "netlist.v").exists():
            shutil.copy(work / "netlist.v", keep_netlist)
    cells, area, flops = parse_stat(log)
    delays = re.findall(r"Delay\s*=\s*([0-9.]+)\s*ps", log)   # ABC stime, after buffering
    delay = float(delays[-1]) if delays else None
    path = re.findall(r"ABC: Start-point = (.*?)\.\s+End-point = (.*?)\.", log)
    critical = f"{path[-1][0]} -> {path[-1][1]}" if path else None
    tail = log[log.rfind("Printing statistics"):] if "Printing statistics" in log else log[-3000:]
    label = "+".join(p.name for p in liberties)
    return SynthResult(label, rows, cols, rows_per_cycle, cells, area, area / cols, flops, delay, critical,
                       tail[-3000:])


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
    parser.add_argument("--target-ps", type=int, default=None,
                        help="timing mode: map for this clock period and report ABC's critical path")
    parser.add_argument("--netlist", type=Path, default=None, help="keep the mapped netlist here (for OpenSTA)")
    parser.add_argument("--merge", type=Path, default=None,
                        help="write the given liberty files merged into this one file and exit")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args()
    if args.merge:
        print(f"wrote {args.merge} with {merge_liberty(args.liberty, args.merge)} cells")
        return
    result = synthesize(args.liberty, rows=args.rows, cols=args.cols, rows_per_cycle=args.rows_per_cycle,
                        target_ps=args.target_ps, keep_netlist=args.netlist)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        delay = (f", ABC critical path {result.abc_delay_ps:.0f} ps ({result.critical_path})"
                 if result.abc_delay_ps else "")
        nand2 = nand2_area(args.liberty[0])
        equiv = f", {result.area_per_column_um2 / nand2:.0f} NAND2-eq per column" if nand2 else ""
        print(f"{result.liberty}: rows={result.rows} cols={result.cols} P={result.rows_per_cycle}: "
              f"{result.cells} cells, {result.flops} flops, {result.area_um2:.0f} um2 total, "
              f"{result.area_per_column_um2:.0f} um2 per column{equiv}{delay}")


if __name__ == "__main__":
    main()
