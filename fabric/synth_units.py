"""Synthesis and static timing of every unit of the layer engine, per library.

The column datapath has its own flow (``fabric.synth``, ``fabric.sta``);
this runs the same flow over the vector units, the state engine, the
attention core, the memory units and the sequencer, and writes the critical
path and area of each per library.  The result is a bracket on the clock,
not a clock: post-synthesis, no wires, one corner, on open libraries at
7 nm (predictive) and 45 nm.

Most units are listed at a small geometry and again (``*_real``) at the 9B
elaboration, because the small one is not representative of the path the
way it was assumed to be.  Area scales with the lanes, as expected, but so
does the fanout of any register a lane count multiplies, and the mapper
cannot buffer a register's own output: at 9B the rotation spends 4.07 of
its 4.60 ns on one flop with 1,001 loads, the attention core 3.57 of 5.84
on a one-bit flag with 780, and the norm 1.72 of 2.31 on a read address
with 203.  A unit measured narrow can hide most of its real path, so the
clock has to come from the ``*_real`` rows.

What ABC is asked for is a second number, and it belongs to the unit rather
than the library.  ABC area-recovers every path its own delay model scores
as slack-positive, and that model is more optimistic than the timer that
grades the result, so at the clock it trades away paths that turn out to be
critical; asking for less buys them back.  How much less is not a constant,
and the curve is not monotonic -- each unit has an interior best and a cliff
past it, and they are in different places:

    index scan   2000: 1,849   1600: 1,739   1400: 1,656   1200: 1,574
                 1000: 1,965   800: 1,965      (identical -- ABC saturates)
    state engine 1600: 1,777   1200: 1,511    900: 1,230    600: 1,218

So there is no global setting to prefer: 900 is the state engine's best and
25 percent worse than 1200 for the index scan.  A unit with nothing to
reclaim is not helped at all -- the attention core's paths sit inside 244 ps
of each other, and 1600, 900 and 400 map it to the same netlist, to the cell;
what works there is keeping a hierarchy (``fabric_cs_resolve``) so ABC maps
the block by itself.  Sweep a unit before trusting a number for it.

    python -m fabric.synth_units --lib asap7=asap7.lib:800 --lib nangate45=ng45.lib:1600:1200 \\
        --sta /path/to/sta --out fabric/results/synth_units.json
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from fabric import layer as L
from fabric.sta import run_sta
from fabric import sram
from fabric.synth import RTL_DIR, nand2_area, synthesize, heavy_inputs

VEC = ("fabric_sram.sv", "fabric_vector.sv")


@dataclass(frozen=True)
class Unit:
    name: str
    top: str
    sources: tuple[str, ...]
    params: dict
    note: str
    luts: bool = False
    noshare: bool = False
    keep_hier: tuple[str, ...] = ()              # modules mapped once rather than per instance
    files: tuple[tuple[str, str], ...] = ()      # (name, contents) written next to the sources


def _program_image(depth: int = 64, seed: int = 1) -> str:
    """A program memory of random words, so the sequencer's ROM stays a ROM (an empty one is optimized away)."""
    import random
    rng = random.Random(seed)
    return "\n".join(f"{rng.getrandbits(256):064x}" for _ in range(depth)) + "\n"


UNITS = [
    Unit("columns", "fabric_columns", VEC + ("fabric_tile.sv",), {"ROWS": 256, "COLS": 16, "WB": 4, "AB": 8, "P": 2, "ACC": 24},
         "the tile's column datapath, 16 columns"),
    Unit("rmsnorm", "fabric_rmsnorm", VEC + ("fabric_norm.sv",), {"D": 64, "XW": 16, "OW": 8, "L": 2, "SW": 44},
         "the norm, two lanes, with the inverse square root", luts=True),
    Unit("delta_state8", "fabric_delta_state8", VEC + ("fabric_recurrent.sv",), {"K": 16, "V": 4, "YSH": 9},
         "the int8 state engine, four lanes of a 16-row state", luts=True),
    Unit("conv_silu", "fabric_conv_silu", VEC + ("fabric_recurrent.sv",), {"K": 4, "L": 2}, "the causal conv and SiLU, two lanes", luts=True),
    Unit("head_gates", "fabric_head_gates", VEC + ("fabric_recurrent.sv",), {"ACC": 24}, "the per-head gates", luts=True),
    Unit("swiglu", "fabric_swiglu", VEC + ("fabric_ffn.sv",), {"L": 2}, "SwiGLU, two lanes", luts=True),
    Unit("residual", "fabric_residual", VEC + ("fabric_ffn.sv",), {"L": 4}, "the residual add, four lanes"),
    Unit("rotary", "fabric_rotary", VEC + ("fabric_attention.sv",), {"HD": 32, "R": 16, "L": 2}, "the rotation, two lanes"),
    Unit("rotary_table", "fabric_rotary_table", VEC + ("fabric_attention.sv",), {"R": 16}, "the rotary table", luts=True),
    Unit("attention", "fabric_attention", VEC + ("fabric_attention.sv",), {"HD": 32, "G": 1, "L": 2, "LW": 28},
         "the attention core, one head of 32, two lanes", luts=True),
    # Two lanes cannot see a fanout: the copies this design puts on a constant
    # that every lane reads are pure overhead there and a win at sixteen, so a
    # change measured only on the narrow row reads backwards.  The real row
    # takes an hour and a half in ABC, which is no way to iterate.  Eight lanes
    # over one head of 64 is the smallest elaboration where a per-lane fanout
    # is a fanout, and it maps in minutes.
    Unit("attention_lanes", "fabric_attention", VEC + ("fabric_attention.sv",),
         {"HD": 64, "G": 1, "L": 8, "LW": 28},
         "the attention core, one head of 64, eight lanes", luts=True),
    # The real elaboration, to test whether a unit's worst stage is the same at
    # any lane count: the 9B attention core is one group of four heads of 256,
    # sixteen lanes, against the representative geometry's one head of 32 and
    # two lanes.
    # Sixteen sigmoids and four exponentials, an interpolated table each:
    # flattened they are most of the 778,111 gates this unit hands ABC, and
    # its synthesis does not finish rather than does not time.  Kept as
    # hierarchy the table is mapped once, and OpenSTA walks through it either
    # way, so the timing is the same measurement.
    # Thirty-two lanes is what a 32-byte beat would allow, and the question it
    # settles is whether the clock survives it: the core measures 1,696 ps at
    # two lanes, 1,770 at eight and 1,911 at sixteen, so the trend says the
    # lane count costs, and a clock lost here is lost by every unit.  The RTL
    # takes L=32 already; only the buffer beat and two constants in the
    # sequencer hold it at sixteen.  Timing it needs neither.
    Unit("attention_l32", "fabric_attention", VEC + ("fabric_attention.sv",), {"HD": 256, "G": 4, "L": 32, "LW": 28},
         "the attention core at 9B with the lanes a 32-byte beat would allow", luts=True,
         keep_hier=("fabric_lut",)),
    Unit("attention_real", "fabric_attention", VEC + ("fabric_attention.sv",), {"HD": 256, "G": 4, "L": 16, "LW": 28},
         "the attention core at the 9B geometry: four heads of 256, sixteen lanes", luts=True,
         keep_hier=("fabric_lut",)),
    # The same units at the 9B elaboration, to see how much of each unit's
    # path the representative geometry hides.  Lanes are NL=8, CL=4,
    # ATT_L=SW_L=16; the head is 256 wide and the hidden width 4096.
    Unit("rmsnorm_real", "fabric_rmsnorm", VEC + ("fabric_norm.sv",), {"D": 4096, "XW": 16, "OW": 8, "L": 8, "SW": 44},
         "the norm at 9B: eight lanes over 4096", luts=True),
    Unit("swiglu_real", "fabric_swiglu", VEC + ("fabric_ffn.sv",), {"L": 16}, "SwiGLU at 9B: sixteen lanes", luts=True),
    Unit("residual_real", "fabric_residual", VEC + ("fabric_ffn.sv",), {"L": 8}, "the residual add at 9B: eight lanes"),
    Unit("conv_silu_real", "fabric_conv_silu", VEC + ("fabric_recurrent.sv",), {"K": 4, "L": 4},
         "the conv and SiLU at 9B: four lanes", luts=True),
    Unit("rotary_real", "fabric_rotary", VEC + ("fabric_attention.sv",), {"HD": 256, "R": 64, "L": 16},
         "the rotation at 9B: sixteen lanes of a 256-wide head"),
    Unit("columns_real", "fabric_columns", VEC + ("fabric_tile.sv",), {"ROWS": 4096, "COLS": 64, "WB": 4, "AB": 8, "P": 2, "ACC": 24},
         "the tile's column datapath at 64 columns"),
    Unit("index_scan", "fabric_index_scan", ("fabric_memory.sv", "fabric_norm.sv") + VEC, {"IDIM": 32, "RPB": 8},
         "the index scan, 32 codes", luts=True),
    Unit("index_scan_real", "fabric_index_scan", ("fabric_memory.sv", "fabric_norm.sv") + VEC, {"IDIM": 128, "RPB": 8},
         "the index scan at 9B: 128 codes", luts=True),
    # The state engine's V lanes are independent, so a narrower V is
    # representative of the path unless something shared fans out to all of
    # them.  The sweep says which: at the real K of 128, V of 16, 32 and 64.
    Unit("delta_state8_v16", "fabric_delta_state8", VEC + ("fabric_recurrent.sv",), {"K": 128, "V": 16, "YSH": 9},
         "the state engine, a 128-row state, sixteen lanes", luts=True),
    Unit("delta_state8_v32", "fabric_delta_state8", VEC + ("fabric_recurrent.sv",), {"K": 128, "V": 32, "YSH": 9},
         "the state engine, a 128-row state, 32 lanes", luts=True),
    Unit("delta_state8_v64", "fabric_delta_state8", VEC + ("fabric_recurrent.sv",), {"K": 128, "V": 64, "YSH": 9},
         "the state engine, a 128-row state, 64 lanes", luts=True),
    Unit("delta_state8_real", "fabric_delta_state8", VEC + ("fabric_recurrent.sv",),
         {"K": 128, "V": 128, "VL": 16, "YSH": 9},
         "the int8 state engine at 9B: a 128 x 128 state, sixteen lanes over eight slices", luts=True),
    Unit("delta_state8_wide", "fabric_delta_state8", VEC + ("fabric_recurrent.sv",),
         {"K": 128, "V": 128, "VL": 128, "YSH": 9},
         "the same state unsliced, which ABC cannot map", luts=True),
    Unit("kv_append_real", "fabric_kv_append", ("fabric_memory.sv", "fabric_norm.sv") + VEC,
         {"HD": 256, "NKV": 4, "IDIM": 128, "BS": 16, "W": 512, "KV_BITS": 4},
         "the append at 9B: four heads of 256", luts=True),
    Unit("rotary_table_real", "fabric_rotary_table", VEC + ("fabric_attention.sv",), {"R": 64},
         "the rotary table at 9B: 64 rotary dimensions", luts=True),
    Unit("topk_real", "fabric_topk", ("fabric_memory.sv", "fabric_norm.sv") + VEC, {"K": 32}, "top-K of 32", luts=True),
    Unit("topk", "fabric_topk", ("fabric_memory.sv", "fabric_norm.sv") + VEC, {"K": 8}, "top-K of eight", luts=True),
    Unit("kv_append", "fabric_kv_append", ("fabric_memory.sv", "fabric_norm.sv") + VEC,
         {"HD": 32, "NKV": 1, "IDIM": 32, "BS": 4, "W": 16, "KV_BITS": 4}, "the append, one head of 32", luts=True),
    Unit("mem_arbiter", "fabric_mem_arbiter", ("fabric_memory.sv", "fabric_norm.sv") + VEC, {"N": 4}, "the memory arbiter, four requesters", luts=True),
    Unit("vector_buffer", "fabric_vb", VEC + ("fabric_engine.sv",),
         {"BYTES": 1 << 16, "NR": 26, "NW": 19, "AW": 16, "NB": 8, "BSH": 13, "RCAP2": 5, "RCAP3": 1, "WCAP2": 2,
          # folded onto the crossbar ports the colouring needs across every
          # program shape -- single token, stream of two, chunk of three.
          "NPR": 11, "NPW": 8,
          "RMAP0": 5305655462704809316, "RMAP1": 2259224409,
          "WMAP0": 5143726797662605616, "WMAP1": 5},
         "the vector buffer's crossbar, 26 reads and 19 writes folded onto 11 and 8 over eight banks (the banks are macros)", noshare=True),
    # The crossbar as the lanes elaborate it at 9B (``engine.Layout`` over four
    # lanes of the die's four layers): every logical port its own, since any
    # step of one lane may run beside any of another's, over forty banks,
    # the memory unit's ports 32 bytes wide.
    Unit("vector_buffer_lanes", "fabric_vb", VEC + ("fabric_engine.sv",),
         {"BYTES": 10485760, "NR": 25, "NW": 20, "AW": 25, "NB": 40, "BSH": 18,
          "RCAP2": 31168951325, "RCAP3": 1074791425, "WCAP2": 533096546800,
          "FOLD": 0, "NPR": 25, "NPW": 20, "WIDE_R": 17, "WIDE_W": 12},
         "the lanes' crossbar at 9B, 25 reads and 20 writes unfolded over forty banks (the banks are macros)", noshare=True),
    # Four lanes, each with its own store and 128 buffer ids (a lane's
    # programs use 85 at the 9B geometry).  The store is a macro, so its
    # depth costs the logic nothing.  The lane is mapped once: flat, the
    # four took two and a half hours; kept, three minutes.
    Unit("sequencer", "fabric_sequencer", ("fabric_sram.sv", "fabric_sequencer.sv"),
         {"NU": 10, "NE": 4, "LN": 4, "DEPTH": 512, "NID": 128},
         "the token sequencer, four lanes, each a 512-step program store and 128 buffer ids",
         files=(("program.hex", _program_image(512)),), keep_hier=("fabric_seq_lane",)),
]


def path_cells(report: str) -> list[str]:
    """The cell types along the worst path, in order, from OpenSTA's full report."""
    start = report.find("Startpoint:")
    end = report.find("data arrival time", start)
    section = report[start:end] if start >= 0 and end >= 0 else ""
    return re.findall(r"\((\w+)\)", section)


def run_unit(unit: Unit, lib_name: str, liberty: Path, target_ps: int, sta: Path | None, keep: Path | None = None,
             abc_target_ps: int | None = None) -> dict:
    """``target_ps`` is the clock the slack is reported against; ``abc_target_ps``
    is what ABC is asked for, which is not the same number.

    ABC area-recovers every path its own delay model scores as slack-positive,
    and that model is more optimistic than the timer that grades the result, so
    it trades away paths that turn out to be critical.  Asking for less than the
    clock buys that back.  On the state engine, at the same RTL:

        asked 1600 -> 1,777 ps   asked 1200 -> 1,511 ps (+0.3% area)
        asked  900 -> 1,230 ps (+3.9%)   asked 600 -> 1,218 ps (+5.2%)

    900 is that unit's knee, and only that unit's: the curve is not monotonic
    and the best target differs per unit.  The index scan is best at 1200 and
    falls off a cliff below it -- 1,574 ps there against 1,965 at 1000, for
    19 percent more gates -- so a target good for one unit is 25 percent worse
    for another.  The module docstring has both sweeps.  Sweep before trusting.

    Defaults to ``target_ps`` so an unchanged caller gets the
    behaviour it had.
    """
    abc_target_ps = target_ps if abc_target_ps is None else abc_target_ps
    t0 = time.time()
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        data = []
        if unit.luts:
            L.write_luts(work)
            data = sorted(work.glob("lut_*.hex"))
        for name, contents in unit.files:
            (work / name).write_text(contents)
            data.append(work / name)
        netlist = (keep / f"{unit.name}_{lib_name}.v") if keep else (work / "netlist.v")
        try:
            synth = synthesize(liberty, top=unit.top, target_ps=abc_target_ps, keep_netlist=netlist,
                               sources=[RTL_DIR / name for name in unit.sources], params=unit.params, data_files=data,
                               noshare=unit.noshare, keep_hier=unit.keep_hier)
        except Exception as error:  # noqa: BLE001 - the report says what failed
            return {"unit": unit.name, "library": lib_name, "error": str(error)[-1500:], "seconds": time.time() - t0}
        out = {"unit": unit.name, "library": lib_name, "top": unit.top, "params": unit.params, "note": unit.note,
               "cells": synth.cells, "flops": synth.flops, "area_um2": synth.area_um2, "abc_delay_ps": synth.abc_delay_ps,
               "buffered": synth.buffered, "target_ps": target_ps, "abc_target_ps": abc_target_ps,
               "seconds": time.time() - t0}
        nand2 = nand2_area(liberty)
        if nand2:
            out["nand2_equiv"] = synth.area_um2 / nand2
        if unit.keep_hier and Path(netlist).exists():
            heavy = heavy_inputs(Path(netlist).read_text(), unit.top)
            if heavy:                                            # unbuffered: the timing below is not the design's
                out["heavy_inputs"] = heavy
        # The memories are blackboxes to synthesis; the timing tools get a liberty
        # for the ones this design has, so their address and data paths are timed.
        macros, libs = [], [liberty]
        design = Path(netlist).with_suffix(".json")
        if design.exists() and lib_name in sram.PROCESSES:
            macros = sram.macros_from_json(design)
            if macros:
                process = sram.PROCESSES[lib_name]
                libs.append(sram.write_liberty(macros, process, work / "sram.lib", reference=liberty))
                out["sram"] = sram.inventory(macros, process)
                out["area_um2"] += out["sram"]["area_um2"]
                if nand2:
                    out["nand2_equiv"] = out["area_um2"] / nand2
        if sta is not None and netlist.exists():
            try:
                timing = run_sta(sta, libs, netlist, top=unit.top, period_ps=target_ps)
                out.update(critical_path_ps=timing.critical_path_ps, worst_slack_ps=timing.worst_slack_ps,
                           max_frequency_mhz=timing.max_frequency_mhz, startpoint=timing.startpoint, endpoint=timing.endpoint,
                           path_cells=path_cells(timing.report))
            except Exception as error:  # noqa: BLE001
                out["sta_error"] = str(error)[-1500:]
        out["seconds"] = time.time() - t0
    return out


def _job(args):
    return run_unit(*args)


def report_markdown(results: list[dict]) -> str:
    libs = sorted({r["library"] for r in results})
    lines = ["| Unit | " + " | ".join(f"{lib}: path (ps) | {lib}: NAND2-eq" for lib in libs) + " |",
             "| --- | " + " | ".join("---: | ---:" for _ in libs) + " |"]
    for unit in [u for u in UNITS if any(r["unit"] == u.name for r in results)]:
        row = [unit.name]
        for lib in libs:
            r = next((r for r in results if r["unit"] == unit.name and r["library"] == lib), None)
            if r is None or "error" in r:
                row += ["failed", ""]
            else:
                path = r.get("critical_path_ps", r.get("abc_delay_ps"))
                mark = "" if r.get("buffered", True) else " (unbuffered)"
                row += [(f"{path:.0f}" if path else "?") + mark, f"{r.get('nand2_equiv', 0):,.0f}"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", action="append", required=True,
                        help="name=liberty.lib:target_ps[:abc_target_ps] -- the clock the slack is\n"
                             "reported against, and optionally what ABC is asked for, which wants to\n"
                             "be lower (see run_unit)")
    parser.add_argument("--sta", type=Path, default=None)
    parser.add_argument("--units", default=None, help="comma-separated unit names (default all)")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("fabric/results/synth_units.json"))
    parser.add_argument("--keep", type=Path, default=None, help="directory for the mapped netlists")
    args = parser.parse_args()
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=True)
    libs = []
    for spec in args.lib:
        name, rest = spec.split("=", 1)
        path, _, targets = rest.rpartition(":")
        # A second number is ABC's; with one, ABC is asked for the clock, as before.
        if ":" in path and not Path(path).exists():
            path, _, first = path.rpartition(":")
            target, abc_target = int(first), int(targets)
        else:
            target, abc_target = int(targets), None
        libs.append((name, Path(path), target, abc_target))
    units = [u for u in UNITS if args.units is None or u.name in args.units.split(",")]
    jobs = [(u, name, path, target, args.sta, args.keep, abc) for u in units
            for name, path, target, abc in libs]
    # A rerun of some units replaces their rows in the existing file and keeps the others.
    results = [r for r in (json.loads(args.out.read_text()) if args.out.exists() else [])
               if not any(r["unit"] == u.name and r["library"] == name for u in units for name, *_ in libs)]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for result in pool.map(_job, jobs):
            results.append(result)
            status = result.get("error", "")[:80] or (
                f"{result.get('critical_path_ps', result.get('abc_delay_ps', 0)) or 0:.0f} ps, "
                f"{result.get('nand2_equiv', 0):,.0f} NAND2-eq"
                + ("" if result.get("buffered", True) else "  (UNBUFFERED: ABC's timing script aborted)"))
            print(f"{result['unit']:14s} {result['library']:10s} {result['seconds']:6.0f} s  {status}", flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, indent=2))
    print()
    print(report_markdown(results))


if __name__ == "__main__":
    main()
