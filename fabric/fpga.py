"""What the controller's datapath costs on an FPGA.

The FPGA at the head of the ring is a part of the appliance, not a step on
the way to it, so the question is not whether the gateware fits a prototype
board but how much of the part it leaves for everything else: the PCIe
endpoint, the DDR4 controller for the embedding table, the ring's physical
layer, and the soft side's scheduler.  This runs yosys's own FPGA flows over
``fabric_controller_top`` at the design's geometry and reports the cells.

    python -m fabric.fpga --family ecp5 --d 4096 --k 32

No vendor tools are involved, so the numbers are yosys's mapping rather than
a vendor's, and a vendor's will differ; what they are good for is the shape
-- which of logic, registers and memory the datapath is made of, and whether
it is a corner of a part or most of one.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

RTL_DIR = Path(__file__).parent / "rtl"
SOURCES = ("fabric_vector.sv", "fabric_controller.sv", "fabric_ring.sv", "fabric_controller_top.sv")
FAMILIES = {
    "ecp5": ("synth_ecp5", {"LUT4": "luts", "TRELLIS_FF": "flops", "DP16KD": "block_rams", "MULT18X18D": "multipliers"}),
    "xilinx": ("synth_xilinx", {"LUT1": "luts", "LUT2": "luts", "LUT3": "luts", "LUT4": "luts", "LUT5": "luts",
                                "LUT6": "luts", "FDRE": "flops", "FDSE": "flops", "FDCE": "flops", "FDPE": "flops",
                                "RAMB18E1": "block_rams", "RAMB36E1": "block_rams", "DSP48E1": "multipliers"}),
}


def yosys_command() -> list[str]:
    if shutil.which("yosys"):
        return ["yosys"]
    return ["python3", "-m", "yowasp_yosys"]


def synthesize_controller(family: str = "ecp5", d: int = 4096, k: int = 32, top: str = "fabric_controller_top") -> dict:
    """Map the datapath for one family and count what it took."""
    script, wanted = FAMILIES[family]
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        for name in (*SOURCES, "fabric_fx.svh"):
            shutil.copy(RTL_DIR / name, work / name)
        from fabric import layer as L
        L.write_luts(work)
        lines = [f"read_verilog -sv -DFABRIC_SYNTH {name}" for name in SOURCES]
        lines += [f"chparam -set D {d} -set K {k} {top}", f"hierarchy -check -top {top}",
                  f"{script} -top {top}", "stat"]
        (work / "synth.ys").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run = subprocess.run([*yosys_command(), "-q", "-l", "synth.log", "synth.ys"], cwd=work,
                             capture_output=True, text=True)
        log = (work / "synth.log").read_text(encoding="utf-8") if (work / "synth.log").exists() else run.stdout
        if run.returncode != 0:
            return {"family": family, "d": d, "k": k, "error": log[-1500:]}
    # The last statistics block is the whole design's.
    section = log[log.rfind("Printing statistics"):]
    counts: dict[str, int] = {}
    for cell, n in re.findall(r"^\s+(\w+)\s+(\d+)\s*$", section, re.M):
        key = wanted.get(cell)
        if key:
            counts[key] = counts.get(key, 0) + int(n)
    total = re.search(r"Number of cells:\s+(\d+)", section)
    return {"family": family, "d": d, "k": k, "cells": int(total.group(1)) if total else 0, **counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="ecp5", choices=sorted(FAMILIES))
    parser.add_argument("--d", type=int, default=4096, help="hidden elements")
    parser.add_argument("--k", type=int, default=32, help="rows a head die's list may hold")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = synthesize_controller(args.family, args.d, args.k)
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        rows = [r for r in (json.loads(args.out.read_text()) if args.out.exists() else [])
                if not (r["family"] == result["family"] and r["d"] == result["d"] and r["k"] == result["k"])]
        args.out.write_text(json.dumps(rows + [result], indent=2))


if __name__ == "__main__":
    main()
