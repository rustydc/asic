"""The die's clock from its units' synthesis at the 9B elaboration, and the
die's throughput at that clock.

Each unit of a 9B die is synthesized on its own at the geometry the die
elaborates it at (``synth_units``, the ``*_real`` rows, the lanes'
sequencer and vector buffer).  The slowest sets the stage, and the stage
is read across to 28 nm by FO4: NanGate 45 measures 20.54 ps on a
twenty-stage fanout-of-four inverter chain, and a 28 nm FO4 is 15 to 18
ps, so a path of P ps on NanGate 45 is P x 15/20.54 to P x 18/20.54 at 28
nm.  The middle of that is the clock.  It is pre-layout, one corner, no
wires; ``WIRE_MARGIN`` is the usual allowance for them, reported beside it.

The throughput is ``sequencer.lane_rate`` at that clock: four lanes each
taking a context's token a link gap after the last, on sixteen PSRAMs at
their datasheet timing.  The memory's timing is absolute, so the cycles a
token costs change with the clock and the rate is not proportional to it.

    python -m fabric.clock --results fabric/results/synth_units.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

NANGATE45_FO4_PS = 20.54
FO4_28NM_PS = (15.0, 18.0)
WIRE_MARGIN = 0.3            # the 20 to 40 percent pre-layout paths grow by in place and route

# The units of a 9B die, at the geometry the die elaborates them at.
DIE_UNITS = {
    "columns_real": "the tiles' column datapath, 64 columns",
    "rmsnorm_real": "the norm, eight lanes over 4096",
    "conv_silu_real": "the causal conv and SiLU, four lanes",
    "head_gates": "the per-head gates",
    "delta_state8_real": "the int8 state engine, 128 x 128, sixteen lanes over eight slices",
    "swiglu_real": "SwiGLU, sixteen lanes",
    "residual_real": "the residual add, eight lanes",
    "rotary_real": "the rotation, sixteen lanes of a 256-wide head",
    "rotary_table_real": "the rotary table, 64 rotary dimensions",
    "attention_real": "the attention core, four heads of 256, sixteen lanes",
    "index_scan_real": "the index scan, 128 codes",
    "topk_real": "top-K of 32",
    "kv_append_real": "the append, four heads of 256",
    "mem_arbiter": "the memory arbiter, four requesters",
    "vector_buffer_lanes": "the buffer's crossbar, 25 reads and 20 writes over 40 banks",
    "sequencer": "the sequencer, four lanes of 128 ids, the lane mapped once",
}


@dataclass
class Clock:
    unit: str                # the slowest unit
    path_ps: float           # its critical path on NanGate 45
    fo4: float
    mhz: float               # at the middle of the 28 nm FO4 range
    mhz_range: tuple[float, float]
    mhz_wired: float         # with WIRE_MARGIN on the path
    missing: list[str]       # die units with no result: the clock is a floor on the period without them


def best(results: list[dict], library: str = "nangate45") -> dict[str, dict]:
    """Each unit's best mapped result on ``library``: the shortest critical
    path of any ABC target it was run at.  A result whose kept hierarchy has
    an unbuffered heavy input (``synth.heavy_inputs``) times that net rather
    than the design, and is left out."""
    out: dict[str, dict] = {}
    for r in results:
        if r.get("library") != library or not r.get("critical_path_ps") or r.get("heavy_inputs"):
            continue
        if r["unit"] not in out or r["critical_path_ps"] < out[r["unit"]]["critical_path_ps"]:
            out[r["unit"]] = r
    return out


def to_28nm_mhz(path_ps: float, fo4_ps: float) -> float:
    return 1e6 / (path_ps / NANGATE45_FO4_PS * fo4_ps)


def clock(results: list[dict], units: dict[str, str] = DIE_UNITS) -> Clock:
    rows = best(results)
    have = {u: rows[u] for u in units if u in rows}
    slow = max(have.values(), key=lambda r: r["critical_path_ps"])
    p = slow["critical_path_ps"]
    mid = sum(FO4_28NM_PS) / 2
    return Clock(slow["unit"], p, p / NANGATE45_FO4_PS, to_28nm_mhz(p, mid),
                 (to_28nm_mhz(p, FO4_28NM_PS[1]), to_28nm_mhz(p, FO4_28NM_PS[0])),
                 to_28nm_mhz(p * (1 + WIRE_MARGIN), mid), [u for u in units if u not in rows])


@dataclass
class Rate:
    mhz: float
    position: int            # the token's position in its context
    decode: float            # tokens/s, four lanes of single tokens
    one_lane: float          # tokens/s, one lane: a context alone on the die
    lane_ms: float           # a lane's token, done to done, in ms: what one context waits a die
    prefill: float           # tokens/s, four lanes of eight-token chunks


def rates(mhz: float, positions: tuple[int, ...] | None = None, devices: int = 16) -> list[Rate]:
    from fabric import sequencer as S
    from fabric.memory import MemoryMap
    from fabric.tile import TileSpec
    from fixed_llm_poc import ASICLMConfig
    cfg = ASICLMConfig.qwen3_5_9b()
    mm, spec = MemoryMap.from_config(cfg), TileSpec()
    t = S.Timing(core_mhz=mhz, devices=devices, memory_mode="pipelined", pushout=0)
    out = []
    for pos in positions or (4095, mm.context_tokens - 1):
        rec = S.recurrent_program(cfg, None, spec, mm, t, slots=S.LANE_SLOTS)
        glob = S.global_program(cfg, None, spec, mm, pos, t)
        four = S.lane_rate(rec, glob, 4, t, tokens=10)
        one = S.lane_rate(rec, glob, 1, t)
        rec8 = S.recurrent_program(cfg, None, spec, mm, t, chunk=8, slots=S.LANE_SLOTS)
        glob8 = S.global_program(cfg, None, spec, mm, pos, t, chunk=8)
        pre = S.lane_rate(rec8, glob8, 4, t, per_job=8)
        out.append(Rate(mhz, pos + 1, four.tokens_per_s, one.tokens_per_s, four.lane_cycles / mhz / 1e3, pre.tokens_per_s))
    return out


def report_markdown(results: list[dict], units: dict[str, str] = DIE_UNITS) -> str:
    rows, c = best(results), clock(results, units)
    mid = sum(FO4_28NM_PS) / 2
    lines = ["| Unit | Geometry | NanGate 45 (ns) | FO4 | 28 nm (MHz) | NAND2-eq |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for u, note in sorted(units.items(), key=lambda kv: -rows[kv[0]]["critical_path_ps"] if kv[0] in rows else 0):
        r = rows.get(u)
        if r is None:
            lines.append(f"| {u} | {note} | not mapped | | | |")
            continue
        p = r["critical_path_ps"]
        lines.append(f"| {u} | {note} | {p / 1000:.2f} | {p / NANGATE45_FO4_PS:.1f} | {to_28nm_mhz(p, mid):.0f} "
                     f"| {r.get('nand2_equiv', 0):,.0f} |")
    lines += ["", f"The clock: {c.unit} at {c.path_ps:.0f} ps, {c.fo4:.1f} FO4: {c.mhz:.0f} MHz at 28 nm "
              f"({c.mhz_range[0]:.0f} to {c.mhz_range[1]:.0f}), {c.mhz_wired:.0f} with {WIRE_MARGIN:.0%} for wires."]
    if c.missing:
        lines.append(f"Not mapped, so not in it: {', '.join(c.missing)}.")
    return "\n".join(lines)


def rates_markdown(rs: list[Rate]) -> str:
    lines = ["| Clock | Position | Decode, 4 lanes | One lane | A lane's token | Prefill, 4 lanes of 8 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rs:
        lines.append(f"| {r.mhz:.0f} MHz | {r.position:,} | {r.decode:,.0f} tok/s | {r.one_lane:,.0f} | {r.lane_ms:.2f} ms "
                     f"| {r.prefill:,.0f} tok/s |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, nargs="+", default=[Path(__file__).parent / "results" / "synth_units.json"])
    parser.add_argument("--out", type=Path, default=None, help="write the clock and the rates as JSON")
    args = parser.parse_args()
    results = [r for path in args.results for r in json.loads(path.read_text())]
    c = clock(results)
    print(report_markdown(results))
    print()
    rs = [r for mhz in (c.mhz_wired, c.mhz) for r in rates(mhz)]
    print(rates_markdown(rs))
    if args.out:
        args.out.write_text(json.dumps({"clock": asdict(c), "rates": [asdict(r) for r in rs]}, indent=2))


if __name__ == "__main__":
    main()
