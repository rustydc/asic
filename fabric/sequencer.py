"""The token sequencer: the program that runs a layer's tiles, vector
units and DMAs in order for one token, the merge of several tokens'
programs into one stream, and the controller that executes it.

A layer is a fixed dataflow graph over a few dozen on-chip vector buffers:
the residual, the pass input, the pass outputs, the per-head state slots.
This module writes that graph out as a *program*: a list of steps, each a
command to one unit (a tile pass, a norm, a state-engine head, a DMA) with
the buffers it consumes and produces.  Three things are derived from the
same list, so they cannot drift apart:

* ``run_program`` executes the steps on the integer model of ``layer.py``
  and must reproduce ``recurrent_layer_int`` / ``global_layer_int`` bit for
  bit, which proves the program's data flow (slices, head loops, buffer
  reuse) is complete and in a legal order;
* ``schedule`` runs a list scheduler over the steps with each unit's
  throughput and the memory port's bandwidth, giving cycles per token and
  where the time goes, under the controller's exact issue rules so the RTL
  testbench's cycle count must equal it;
* ``emit_program`` writes the steps as the controller's program image.

Dependencies come from buffer names.  A step consumes the buffers it
reads and produces the ones it writes; a write waits for the buffer's
outstanding writers and readers, a read for its outstanding writers.  A
produce marked ``+name`` is a *contribution* to a whole vector by one of
several parallel producers (a head's slice of ``y_norm``): it waits only
for the readers of the previous version, and a consumer of the whole waits
for every contribution.

Several tokens of different contexts run at once by ``stream``: each
token's program has its buffers renamed with a token suffix (the state
slots stay shared, they are the engines' physical buffers), and the
programs are merged into one issue order by simulating which step could
issue first, so one token's memory steps run under another's passes.

The controller (``rtl/fabric_sequencer.sv``) is a microcoded issue engine:
in program order, when the head step's consumed buffers have no
outstanding writer, its produced buffers no outstanding writer or reader
(readers only, for a contribution), and its unit reports the addressed
engine free, it sends the command and moves on; a completing unit returns
the step's tag and the controller releases the step's buffers.  Units are
black boxes to it, which is what lets one testbench check the controller
against stub units of programmed duration.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from fabric import hpi
from fabric import layer as L
from fabric.memory import BEAT, MemoryMap, _beats, eligible_blocks
from fabric.tile import TileSpec, write_hex

# --------------------------------------------------------------------------
# The units and their throughput (from the RTL's lanes and latencies)
# --------------------------------------------------------------------------

MAX_CONSUME = 7              # buffers a step may consume (the program word's fields)
MAX_PRODUCE = 2              # buffers a step may produce
MAX_IDS = 255                # buffer ids per program image (0xFF is "none")
UNITS: dict[str, tuple[int, int]] = {   # name -> (id, engines)
    "tiles": (0, 1), "norm": (1, 2), "conv": (2, 1), "gates": (3, 1), "delta": (4, 4),
    "swiglu": (5, 1), "residual": (6, 1), "rotary": (7, 2), "attn": (8, 4), "mem": (9, 1),
}
SHARED_PREFIX = "s_slot"     # buffers shared by every token in flight: the state engines' slots


@dataclasses.dataclass(frozen=True)
class Timing:
    """Lanes and latencies of the units; the memory port in bytes per core cycle."""
    core_mhz: float = 800.0
    rows_per_cycle: int = 2
    pass_latency: int = 16
    l_norm: int = 16
    norm_latency: int = 7
    l_conv: int = 8
    conv_latency: int = 7
    gates_latency: int = 9
    delta_latency: int = 6           # row out follows row in by K + 4; two passes over K rows
    l_vec: int = 8                   # swiglu, residual, rotary, silu
    swiglu_latency: int = 6
    residual_latency: int = 1
    silu_latency: int = 4
    rotary_latency: int = 3
    l_attn: int = 64
    attn_exp_stall: int = 4          # in_ready drops after a key row
    attn_out_latency: int = 12
    port_bytes_per_cycle: float = 16e9 / 800e6   # sixteen devices at 250 MHz DDR x16 against the core clock

    def norm(self, d: int) -> int:
        return 2 * -(-d // self.l_norm) + self.norm_latency

    def tile_pass(self, rows_in: int) -> int:
        return -(-rows_in // self.rows_per_cycle) + self.pass_latency

    def memory(self, nbytes: int, burst_bytes: int) -> int:
        """Cycles the port is busy moving nbytes in bursts of burst_bytes, at the HPI burst efficiency."""
        return int(math.ceil(nbytes / (self.port_bytes_per_cycle * hpi.efficiency(max(1, burst_bytes // BEAT)))))


# --------------------------------------------------------------------------
# Steps, dependencies, programs
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Step:
    name: str
    unit: str
    engine: int
    src: tuple[str, ...]             # buffers consumed
    dst: tuple[str, ...]             # buffers produced; "+name" contributes to a whole vector
    cycles: int
    func: Callable[[dict], None] | None = None
    nbytes: int = 0                  # memory traffic, for the report
    token: int | None = None         # set on a stream's steps: their private buffers carry the token suffix
    deps: list[int] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        self.cycles = max(1, int(self.cycles))       # a command occupies its unit for at least a cycle
        assert len(self.src) <= MAX_CONSUME and len(self.dst) <= MAX_PRODUCE, self.name


def _plain(name: str) -> str:
    return name[1:] if name.startswith("+") else name


class Linker:
    """The dependency rules, applied step by step: per buffer the writers of
    its current version and the readers since."""

    def __init__(self) -> None:
        self.writers: dict[str, list[int]] = {}
        self.readers: dict[str, list[int]] = {}

    def deps_for(self, step: Step) -> list[int]:
        deps: set[int] = set()
        for name in step.src:
            deps.update(self.writers.get(name, []))
        for name in step.dst:
            plain = _plain(name)
            deps.update(self.readers.get(plain, []))
            if not name.startswith("+"):
                deps.update(self.writers.get(plain, []))
        return sorted(deps)

    def commit(self, step: Step, index: int) -> None:
        for name in step.src:
            self.readers.setdefault(name, []).append(index)
        for name in step.dst:
            plain = _plain(name)
            if name.startswith("+") and not self.readers.get(plain):
                self.writers.setdefault(plain, []).append(index)     # another contribution to the current version
            else:
                self.writers[plain] = [index]                        # a new version
                self.readers[plain] = []


def link(steps: list[Step]) -> None:
    """Fill each step's dependencies in program order."""
    lk = Linker()
    for i, step in enumerate(steps):
        step.deps = lk.deps_for(step)
        lk.commit(step, i)


def _slot(engine: int, k: int) -> str:
    """The physical state buffer of an engine: two per engine, so the next head's read overlaps this one."""
    return f"{SHARED_PREFIX}[{engine * 2 + k % 2}]"


def recurrent_program(cfg, c: L.RecurrentConsts | None, spec: TileSpec, mm: MemoryMap, t: Timing = Timing()) -> list[Step]:
    """One token through a recurrent layer.  Inputs in the environment:
    ``x`` (int16 residual), ``s_mem[h]`` (the state rows of each head) and,
    for the int8 state, ``scale_mem[h]`` (its scale beat), ``hist_mem`` (the
    conv history).  With ``c`` None the program has its shape and timing
    but cannot be run."""
    nk, nv, hk, hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    d, kd, vd = cfg.hidden_size, nk * hk, nv * hv
    conv_dim, ffn = 2 * kd + vd, cfg.layer_intermediate_size(0)
    n_delta = UNITS["delta"][1]
    repeat = nv // nk
    int8_state = mm.state_bits == 8
    head_bytes = hk * hv * mm.state_bits // 8 + (BEAT if int8_state else 0)   # the rows and the scale beat
    steps: list[Step] = []

    def add(name, unit, src, dst, cycles, func=None, engine=0, nbytes=0):
        steps.append(Step(name, unit, engine, tuple(src), tuple(dst), int(cycles), func, nbytes))

    def fabric(q):
        return lambda x: L._fabric(q, x, spec)

    hist_bytes = _beats(conv_dim * (cfg.linear_conv_kernel - 1)) * BEAT
    add("dma.hist_rd", "mem", (), ("hist",), t.memory(hist_bytes, 2048), lambda e: e.__setitem__("hist", e["hist_mem"]), nbytes=hist_bytes)
    add("norm.h", "norm", ("x",), ("A",), t.norm(d), lambda e: e.__setitem__("A", L._norm(e["x"], c.norm)))

    def in_proj(e):                                     # one buffer P1: qkv | z | b, a accumulators
        _, e["qkv"] = fabric(c.in_proj_qkv)(e["A"])
        _, e["z"] = fabric(c.in_proj_z)(e["A"])
        e["b_acc"], _ = fabric(c.in_proj_b)(e["A"])
        e["a_acc"], _ = fabric(c.in_proj_a)(e["A"])
    add("pass.in_proj", "tiles", ("A",), ("P1",), t.tile_pass(d), in_proj)

    def conv(e):
        e["conv"], e["hist_next"] = L.conv_silu_int(e["hist"], e["qkv"], c.conv_w, c.conv_mult_in, c.conv_sh_in,
                                                    c.conv_mult_out, c.conv_sh_out)
    add("conv", "conv", ("P1", "hist"), ("conv", "hist_next"), -(-conv_dim // t.l_conv) + t.conv_latency, conv)
    add("dma.hist_wr", "mem", ("hist_next",), (), t.memory(hist_bytes, 2048), lambda e: e.__setitem__("hist_mem", e["hist_next"]), nbytes=hist_bytes)
    add("gates", "gates", ("P1",), ("gates",), nv + t.gates_latency,
        lambda e: e.update(zip(("decay", "beta"), L.head_gates_int(e["a_acc"][:nv], e["b_acc"][:nv], c.gate_mult_a, c.gate_sh_a,
                                                                    c.gate_mult_b, c.gate_sh_b, c.a_coef, c.dt_bias))))
    for i in range(nk):
        for which, off in (("q", 0), ("k", kd)):
            def unit_norm(e, i=i, off=off, which=which):
                e[f"{which}_unit[{i}]"] = L._norm(e["conv"][off + i * hk:off + (i + 1) * hk], c.unit_norm)
            add(f"norm.{which}[{i}]", "norm", ("conv",), (f"{which}_unit[{i}]",), t.norm(hk), unit_norm, engine=i % UNITS["norm"][1])
    # The heads over the state engines, each engine alternating two state slots.
    for h in range(nv):
        e_id, k = h % n_delta, h // n_delta
        slot = _slot(e_id, k)
        def s_rd(e, h=h, slot=slot):                       # the slot holds the rows and, for int8, the scale beat
            e[slot] = (e["s_mem"][h], tuple(int(x) for x in e["scale_mem"][h])) if int8_state else e["s_mem"][h]
        add(f"dma.s_rd[{h}]", "mem", (), (slot,), t.memory(head_bytes, 2048), s_rd, nbytes=head_bytes)

        def delta(e, h=h, slot=slot):
            v = e["conv"][2 * kd + h * hv:2 * kd + (h + 1) * hv]
            k_unit, q_unit = e[f"k_unit[{h // repeat}]"], e[f"q_unit[{h // repeat}]"]
            if int8_state:
                t_new, *scale_new, e[f"y[{h}]"] = L.delta_state_int8(e[slot][0], k_unit, v, q_unit, int(e["decay"][h]),
                                                                     int(e["beta"][h]), *e[slot][1])
                e[slot] = (t_new, tuple(scale_new))
            else:
                e[slot], e[f"y[{h}]"] = L.delta_state_int(e[slot], k_unit, v, q_unit, int(e["decay"][h]), int(e["beta"][h]))
        add(f"delta[{h}]", "delta", (slot, f"q_unit[{h // repeat}]", f"k_unit[{h // repeat}]", "conv", "gates"),
            (slot, f"y[{h}]"), 2 * hk + hk + 4 + t.delta_latency, delta, engine=e_id)

        def s_wr(e, h=h, slot=slot):
            if int8_state:
                e["s_mem"][h], e["scale_mem"][h] = e[slot]
            else:
                e["s_mem"][h] = e[slot]
        add(f"dma.s_wr[{h}]", "mem", (slot,), (), t.memory(head_bytes, 2048), s_wr, nbytes=head_bytes)

        def gnorm(e, h=h):
            gate = L.silu_fixed(L.requant(e["z"][h * hv:(h + 1) * hv], c.z_mult, c.z_shift, 16))
            e[f"y_norm[{h}]"] = L._norm(e[f"y[{h}]"], c.gated_norm, gain=gate)
        add(f"gnorm[{h}]", "norm", (f"y[{h}]", "P1"), ("+y_norm",), -(-hv // t.l_vec) + t.silu_latency + t.norm(hv), gnorm,
            engine=h % UNITS["norm"][1])

    def out_proj(e):
        y_norm = np.concatenate([e[f"y_norm[{h}]"] for h in range(nv)])
        _, e["mixer"] = fabric(c.out_proj)(y_norm)
    add("pass.out_proj", "tiles", ("y_norm",), ("mixer",), t.tile_pass(vd), out_proj)
    add("residual.1", "residual", ("x", "mixer"), ("x1",), -(-d // t.l_vec) + t.residual_latency,
        lambda e: e.__setitem__("x1", L.residual_int(e["x"], e["mixer"], c.res_mult, c.res_shift)))
    _ffn_steps(add, c.ffn if c is not None else None, spec, d, ffn, t)
    link(steps)
    return steps


def _ffn_steps(add, f: L.FfnConsts | None, spec: TileSpec, d: int, ffn: int, t: Timing) -> None:
    add("norm.h2", "norm", ("x1",), ("A2",), t.norm(d), lambda e: e.__setitem__("A2", L._norm(e["x1"], f.norm)))

    def gate_up(e):
        _, e["gate"] = L._fabric(f.gate_proj, e["A2"], spec)
        _, e["up"] = L._fabric(f.up_proj, e["A2"], spec)
    add("pass.gate_up", "tiles", ("A2",), ("GU",), t.tile_pass(d), gate_up)
    add("swiglu", "swiglu", ("GU",), ("act",), -(-ffn // t.l_vec) + t.swiglu_latency,
        lambda e: e.__setitem__("act", L.swiglu_int(e["gate"], e["up"], f.mult_g, f.sh_g, f.mult_o, f.sh_o)))
    add("pass.down", "tiles", ("act",), ("ffn",), t.tile_pass(ffn),
        lambda e: e.__setitem__("ffn", L._fabric(f.down_proj, e["act"], spec)[1]))
    add("residual.2", "residual", ("x1", "ffn"), ("x2",), -(-d // t.l_vec) + t.residual_latency,
        lambda e: e.__setitem__("x2", L.residual_int(e["x1"], e["ffn"], f.res_mult, f.res_shift)))


def global_program(cfg, c: L.GlobalConsts | None, spec: TileSpec, mm: MemoryMap, pos: int, t: Timing = Timing()) -> list[Step]:
    """One token through a global layer at position ``pos``.  Inputs in the
    environment: ``x``, and the memory side as ``k_rows``/``v_rows``
    ``[kv_heads, N, hd]`` (the window then the retrieved blocks, as the
    record reader would stream them)."""
    nh, nkv, hd, rd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rotary_dim
    d, ffn, group = cfg.hidden_size, cfg.layer_intermediate_size(cfg.global_layer_offset), nh // nkv
    steps: list[Step] = []

    def add(name, unit, src, dst, cycles, func=None, engine=0, nbytes=0):
        steps.append(Step(name, unit, engine, tuple(src), tuple(dst), int(cycles), func, nbytes))

    add("norm.h", "norm", ("x",), ("A",), t.norm(d), lambda e: e.__setitem__("A", L._norm(e["x"], c.norm)))

    def qkv(e):                                         # one buffer P1: q, gate | k | v | index_q | index_k
        _, qg = L._fabric(c.q_proj, e["A"], spec)
        qg = qg.reshape(nh, 2 * hd)
        e["q_raw"], e["gate"] = qg[:, :hd], qg[:, hd:]
        e["k_raw"] = L._fabric(c.k_proj, e["A"], spec)[1].reshape(nkv, hd)
        e["v"] = L._fabric(c.v_proj, e["A"], spec)[1].reshape(nkv, hd)
        _, e["index_q"] = L._fabric(c.index_q, e["A"], spec)
        _, e["index_k"] = L._fabric(c.index_k, e["A"], spec)
    add("pass.qkv", "tiles", ("A",), ("P1",), t.tile_pass(d), qkv)
    add("rotary.table", "rotary", (), ("rot",), rd // 2 + t.rotary_latency,
        lambda e: e.__setitem__("rot", L.rotary_table_int(pos, c.inv_freq)))
    add("norm.index_q", "norm", ("P1",), ("iq",), t.norm(cfg.index_dim),
        lambda e: e.__setitem__("index_q_unit", L._norm(e["index_q"], c.unit_norm)))
    rot_cycles = t.norm(hd) + -(-hd // t.l_vec) + t.rotary_latency
    for n in range(nkv):
        def krot(e, n=n):
            e[f"k[{n}]"] = L.rotary_int(L._norm(e["k_raw"][n], c.k_norm), *e["rot"], rd, c.rot_mult_k, c.rot_sh_k)
        add(f"rotary.k[{n}]", "rotary", ("P1", "rot"), ("+k",), rot_cycles, krot, engine=n % UNITS["rotary"][1])
    for h in range(nh):
        def qrot(e, h=h):
            e[f"q[{h}]"] = L.rotary_int(L._norm(e["q_raw"][h], c.q_norm), *e["rot"], rd, c.rot_mult_q, c.rot_sh_q)
        add(f"rotary.q[{h}]", "rotary", ("P1", "rot"), (f"+qg[{h // group}]",), rot_cycles, qrot, engine=h % UNITS["rotary"][1])
    # The memory side: append this token's records, scan the index, then stream rows to the cores.
    append_bytes = nkv * mm.kv_record_bytes + (nkv * mm.kv_record_bytes + mm.index_record_bytes) // mm.block
    add("mem.append", "mem", ("k", "P1"), (), t.memory(append_bytes, mm.kv_record_bytes), nbytes=append_bytes)
    # The scan reads the index a page of records per request; the rows are the
    # window (head-major, page bursts) and one mean record per selected block.
    eligible = eligible_blocks(pos, mm.local_window, mm.block)
    scan_bytes = eligible * mm.index_record_bytes
    add("mem.scan", "mem", ("iq",), ("sel",), t.memory(scan_bytes, mm.index_burst_records * mm.index_record_bytes),
        lambda e: e.__setitem__("selected", None), nbytes=scan_bytes)
    n_window, n_blocks = min(pos + 1, mm.local_window), min(cfg.top_blocks, eligible)
    rows = n_window + n_blocks
    row_cycles = 2 * -(-hd // t.l_attn) + t.attn_exp_stall
    for n in range(nkv):
        heads = list(range(n * group, (n + 1) * group))
        window_bytes, block_bytes = n_window * mm.kv_record_bytes, n_blocks * mm.kv_record_bytes
        add(f"mem.rows[{n}]", "mem", ("sel",), (f"rows[{n}]",),
            t.memory(window_bytes, mm.window_burst_records * mm.kv_record_bytes) + t.memory(block_bytes, mm.kv_record_bytes),
            lambda e, n=n: e.__setitem__(f"rows[{n}]", (e["k_rows"][n], e["v_rows"][n])), nbytes=window_bytes + block_bytes)

        def attn(e, n=n, heads=heads):
            q = np.stack([e[f"q[{h}]"] for h in heads])
            k_rows, v_rows = e[f"rows[{n}]"]
            e[f"att[{n}]"] = L.attention_int(q, e["gate"][heads], k_rows, v_rows, mult_s=c.mult_s, sh_s=c.sh_s,
                                             mult_gate=c.mult_gate, sh_gate=c.sh_gate, mult_o=c.mult_o, sh_o=c.sh_o)
        add(f"attn[{n}]", "attn", (f"qg[{n}]", "P1", f"rows[{n}]"), ("+att",),
            rows * row_cycles + group * (hd // t.l_attn) + t.attn_out_latency, attn, engine=n % UNITS["attn"][1])

    def o_proj(e):
        att = np.concatenate([e[f"att[{n}]"] for n in range(nkv)]).reshape(nh * hd)
        _, e["mixer"] = L._fabric(c.o_proj, att, spec)
    add("pass.o_proj", "tiles", ("att",), ("mixer",), t.tile_pass(nh * hd), o_proj)
    add("residual.1", "residual", ("x", "mixer"), ("x1",), -(-d // t.l_vec) + t.residual_latency,
        lambda e: e.__setitem__("x1", L.residual_int(e["x"], e["mixer"], c.res_mult, c.res_shift)))
    _ffn_steps(add, c.ffn if c is not None else None, spec, d, ffn, t)
    link(steps)
    return steps


# --------------------------------------------------------------------------
# Streams of tokens
# --------------------------------------------------------------------------

def _shared(name: str) -> bool:
    return _plain(name).startswith(SHARED_PREFIX)


def _renamed(name: str, token: int) -> str:
    return name if _shared(name) else f"{name}@{token}"


def retarget(steps: list[Step], token: int) -> list[Step]:
    """A copy of a token's program with its private buffers renamed for that token."""
    out = []
    for s in steps:
        out.append(Step(s.name, s.unit, s.engine, tuple(_renamed(n, token) for n in s.src),
                        tuple(_renamed(n, token) for n in s.dst), s.cycles, s.func, s.nbytes, token))
    return out


class _View(dict):
    """The environment as one token sees it: its private names carry the token suffix."""

    def __init__(self, env: dict, token: int) -> None:
        super().__init__()
        self.env, self.token = env, token

    def _key(self, name: str) -> str:
        return _renamed(name, self.token)

    def __getitem__(self, name: str):
        return self.env[self._key(name)]

    def __setitem__(self, name: str, value) -> None:
        self.env[self._key(name)] = value

    def update(self, pairs) -> None:                    # type: ignore[override]
        for name, value in dict(pairs).items():
            self[name] = value


def interleave(programs: list[list[Step]]) -> list[Step]:
    """Merge token programs (already retargeted) into one issue order: at
    each pick, the program whose next step could issue earliest under the
    controller's rules goes next, the older token on a tie.  The merged
    list carries its dependencies."""
    lk = Linker()
    merged: list[Step] = []
    issue: list[int] = []
    end: list[int] = []
    free: dict[tuple[str, int], int] = {}
    ptr = [0] * len(programs)
    while any(p < len(prog) for p, prog in zip(ptr, programs)):
        best = None
        for j, prog in enumerate(programs):
            if ptr[j] >= len(prog):
                continue
            step = prog[ptr[j]]
            deps = lk.deps_for(step)
            t0 = issue[-1] + 1 if issue else 0
            for d in deps:
                t0 = max(t0, end[d] + 1)
            t0 = max(t0, free.get((step.unit, step.engine), 0))
            if best is None or (t0, j) < (best[0], best[1]):
                best = (t0, j, step, deps)
        t0, j, step, deps = best
        step.deps = deps
        merged.append(step)
        issue.append(t0)
        end.append(t0 + step.cycles)
        free[(step.unit, step.engine)] = t0 + step.cycles + 1
        lk.commit(step, len(merged) - 1)
        ptr[j] += 1
    return merged


def stream(program: list[Step], tokens: int) -> list[Step]:
    """`tokens` consecutive tokens of one layer, of different contexts, as one merged program."""
    return interleave([retarget(program, k) for k in range(tokens)])


def run_program(steps: list[Step], inputs: dict) -> dict:
    """Execute the steps in program order on the integer model; returns the
    environment.  A stream's inputs carry the token suffix (``x@0``)."""
    env = dict(inputs)
    views: dict[int, _View] = {}
    for step in steps:
        if step.func is not None:
            step.func(env if step.token is None else views.setdefault(step.token, _View(env, step.token)))
    return env


# --------------------------------------------------------------------------
# Scheduling
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Schedule:
    issue: list[int]
    end: list[int]
    steps: list[Step]

    @property
    def cycles(self) -> int:
        return max(self.end) if self.end else 0

    def busy(self, unit: str) -> int:
        return sum(s.cycles for s in self.steps if s.unit == unit)

    def report(self, t: Timing = Timing()) -> str:
        total = self.cycles
        lines = [f"{total} cycles per token ({total / t.core_mhz:.1f} us at {t.core_mhz:.0f} MHz)",
                 "| Unit | Busy cycles | Share |", "| --- | ---: | ---: |"]
        for unit in UNITS:
            b = self.busy(unit)
            if b:
                lines.append(f"| {unit} | {b} | {100 * b / total / UNITS[unit][1]:.0f}% |")
        traffic = sum(s.nbytes for s in self.steps)
        lines.append(f"\nmemory traffic {traffic / 1024:.0f} KB")
        return "\n".join(lines)


def schedule(steps: list[Step]) -> Schedule:
    """In-order list scheduling: a step issues at the earliest cycle after
    the previous issue that is past the end of each dependency and of the
    engine's previous step."""
    issue, end = [0] * len(steps), [0] * len(steps)
    free: dict[tuple[str, int], int] = {}
    for i, step in enumerate(steps):
        t0 = issue[i - 1] + 1 if i else 0
        for d in step.deps:
            t0 = max(t0, end[d] + 1)
        t0 = max(t0, free.get((step.unit, step.engine), 0))
        issue[i], end[i] = t0, t0 + step.cycles
        free[(step.unit, step.engine)] = end[i] + 1          # an engine finishing at e takes its next command at e + 1
    return Schedule(issue, end, steps)


def token_interval(program: list[Step], tokens: int = 3) -> int:
    """The steady-state cycles per token of a stream: the last token's cost on top of the others."""
    a = schedule(stream(program, tokens)).cycles
    b = schedule(stream(program, tokens - 1)).cycles
    return a - b


# --------------------------------------------------------------------------
# The program image for the controller
# --------------------------------------------------------------------------

def buffer_ids(steps: list[Step]) -> dict[str, int]:
    ids: dict[str, int] = {}
    for step in steps:
        for name in step.src + tuple(_plain(n) for n in step.dst):
            if name not in ids:
                ids[name] = len(ids)
    assert len(ids) <= MAX_IDS, f"{len(ids)} buffers, the program word holds {MAX_IDS}"
    return ids


def encode(steps: list[Step]) -> list[int]:
    """Each step as a 256-bit word: unit, engine, last, the length the unit
    is given, an argument, source and destination fields (unused by the
    stubs), then up to seven consumed buffer ids, two produced ids and
    their contribution bits (0xFF is no buffer)."""
    ids = buffer_ids(steps)
    words = []
    for i, step in enumerate(steps):
        assert step.cycles < (1 << 16) and step.engine < 16
        w = UNITS[step.unit][0] | (step.engine << 4) | (int(i == len(steps) - 1) << 8) | (step.cycles << 16)
        consume = [ids[n] for n in step.src] + [0xFF] * (MAX_CONSUME - len(step.src))
        produce = [ids[_plain(n)] for n in step.dst] + [0xFF] * (MAX_PRODUCE - len(step.dst))
        contrib = sum(int(n.startswith("+")) << k for k, n in enumerate(step.dst))
        for k, c in enumerate(consume):
            w |= c << (96 + 8 * k)
        for k, p in enumerate(produce):
            w |= p << (152 + 8 * k)
        w |= contrib << 168
        words.append(w)
    return words


def emit_program(directory: Path, steps: list[Step]) -> dict:
    """Write ``program.hex`` and the schedule the RTL must reproduce; returns the testbench parameters."""
    write_hex(directory / "program.hex", encode(steps), 256)
    sched = schedule(steps)
    (directory / "expected_issue.txt").write_text("".join(f"{i} {a} {b}\n" for i, (a, b) in enumerate(zip(sched.issue, sched.end))))
    params = {"N": len(steps), "EXPECTED_CYCLES": sched.cycles}
    for name, (uid, engines) in UNITS.items():
        params[f"E{uid}"] = engines
    (directory / "params.json").write_text(json.dumps(params))
    return params


def check_trace(steps: list[Step], trace: str) -> list[str]:
    """Check a testbench trace (``tag unit engine issue done`` per line)
    against the program: every dependency ended before the issue, no engine
    ran two steps at once, and the program order was kept.  Tags are the
    step index modulo 256; the trace is in completion order."""
    rows = [tuple(int(v) for v in line.split()) for line in trace.strip().splitlines() if line.strip()]
    problems = []
    if len(rows) != len(steps):
        return [f"{len(rows)} trace rows for {len(steps)} steps"]
    # Issue order recovers the absolute index: the k-th issue is step k.
    by_issue = sorted(rows, key=lambda r: r[3])
    by_index = {}
    for k, (tag, unit, engine, t_issue, t_done) in enumerate(by_issue):
        if tag != k % 256:
            problems.append(f"issue {k} carried tag {tag}")
        by_index[k] = (unit, engine, t_issue, t_done)
    for i, step in enumerate(steps):
        unit, engine, t_issue, t_done = by_index[i]
        if unit != UNITS[step.unit][0] or engine != step.engine:
            problems.append(f"step {i} ran on unit {unit} engine {engine}")
        for d in step.deps:
            if by_index[d][3] > t_issue:
                problems.append(f"step {i} issued at {t_issue} before dependency {d} ended at {by_index[d][3]}")
        if i and by_index[i - 1][2] > t_issue:
            problems.append(f"step {i} issued before step {i - 1}")
    busy: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for tag, unit, engine, t_issue, t_done in rows:
        busy.setdefault((unit, engine), []).append((t_issue, t_done))
    for key, spans in busy.items():
        spans.sort()
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            if b0 < a1:
                problems.append(f"unit {key[0]} engine {key[1]} overlapped at {b0} < {a1}")
    return problems


# --------------------------------------------------------------------------
# The full-size picture
# --------------------------------------------------------------------------

def report_markdown(cfg, mm: MemoryMap, pos: int, t: Timing = Timing()) -> str:
    """The two layers' schedules at a model's geometry, one token at a time and as a stream."""
    spec = TileSpec()
    out = [f"# Token sequencer schedule: hidden {cfg.hidden_size}, position {pos}", ""]
    for title, steps in (("Recurrent layer", recurrent_program(cfg, None, spec, mm, t)),
                         ("Global layer", global_program(cfg, None, spec, mm, pos, t))):
        sched = schedule(steps)
        interval = token_interval(steps)
        out += [f"## {title}: {len(steps)} steps", "", sched.report(t), "",
                f"stream of tokens: {interval} cycles per token ({interval / t.core_mhz:.1f} us), "
                f"{100 * sched.busy('mem') / interval:.0f}% of the memory port", ""]
    return "\n".join(out)


def main() -> None:
    from fixed_llm_poc import ASICLMConfig
    cfg = ASICLMConfig.qwen3_5_9b()
    mm = MemoryMap.from_config(cfg)
    print(report_markdown(cfg, mm, mm.context_tokens - 1))
    print(report_markdown(cfg, mm, 4095))


if __name__ == "__main__":
    main()
