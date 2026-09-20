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

MAX_CONSUME = 6              # buffers a step may consume (the program word's fields)
MAX_PRODUCE = 2              # buffers a step may produce
MAX_IDS = 255                # buffer ids per program image (0xFF is "none")
UNITS: dict[str, tuple[int, int]] = {   # name -> (id, engines)
    "tiles": (0, 1), "norm": (1, 2), "conv": (2, 1), "gates": (3, 1), "delta": (4, 4),
    "swiglu": (5, 1), "residual": (6, 1), "rotary": (7, 2), "attn": (8, 4), "mem": (9, 1),
}
SHARED_PREFIX = "s_slot"     # buffers shared by every token in flight: the state engines' slots
MEM_PREFIX = "m_"            # names of buffers in the memory image (the rest live in the vector buffer)

# Operand conventions of the layer engine's adapters (rtl/fabric_engine.sv).
NL = 8                       # lanes of the norm, swiglu and residual adapters
CL = 4                       # channels per beat of the conv adapter
HIST_REC = 4                 # bytes per channel of the conv history in the vector buffer (kernel - 1 used)
SLOT_HEADER = BEAT           # the state slot: one beat of scale, exponent, peak, saturated count, then the rows
MEM_RD, MEM_WR = 0, 1        # the memory unit's operations (arg[3:0]): memory to vector buffer, vector buffer to memory
ADDR_BITS = 30               # the program word's address operands (src, dst, a2, a3)
# Vector-buffer ports each adapter has.  A step names every buffer it consumes,
# but a one-port unit reads them in turn -- the state engine takes its slot, its
# unit vector, the conv output and the gates through one port -- so those cannot
# want the same bank in the same cycle.  These must match ``R_*`` and ``W_*`` in
# rtl/fabric_engine.sv.
RD_PORTS = {"tiles": 1, "norm": 2, "conv": 2, "gates": 2, "delta": 1,
            "swiglu": 2, "residual": 2, "rotary": 1, "attn": 1, "mem": 1}
WR_PORTS = {"tiles": 1, "norm": 1, "conv": 2, "gates": 1, "delta": 1,
            "swiglu": 1, "residual": 1, "rotary": 1, "attn": 1, "mem": 1}
MEM_APPEND, MEM_SCAN, MEM_ROWS = 2, 3, 4   # the global layer's: append the token, scan the index, stream a head's rows
MEM_PAGE_SHIFT = 7           # a memory page (the map's alignment) in beats: the context base travels as a page number
ROT_TABLE, ROT_HEAD = 0, 1   # the rotary unit's operations (arg[3:0]); arg[7:4] the head kind, 0 q and 1 k
NORM_INT16 = 1 << 8          # the norm's input elements are int16 (else int8)
NORM_GATED = 1 << 9          # the norm's gain is silu of the requantized int8 vector at arg[31:16]
NORM_RESIDUAL, NORM_UNIT, NORM_GATE, NORM_FFN = 0, 1, 2, 3    # the norm's constant sets


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
    ops: dict | None = None          # the command's operands for the layer engine (see ``operands``)

    def __post_init__(self) -> None:
        self.cycles = max(1, int(self.cycles))       # a command occupies its unit for at least a cycle
        assert len(self.src) <= MAX_CONSUME and len(self.dst) <= MAX_PRODUCE, self.name


def _plain(name: str) -> str:
    return name[1:] if name.startswith("+") else name


def operands(**fields) -> dict:
    """The command fields of a step for the layer engine: the address
    operands ``src``, ``dst``, ``a2`` and ``a3`` (30 bits), the argument
    ``arg`` (32 bits) and ``len`` (16 bits), each an int, a reference
    ``(buffer, byte_offset)`` resolved against the engine's layout (a third
    element shifts the address right, for a page number), or a list of
    ``(shift, value)`` parts OR-ed together.  Vector-buffer references are
    byte addresses, memory references (``m_...``) beat addresses."""
    return {k: v for k, v in fields.items() if v is not None}


def _rename_value(value, token: int):
    if isinstance(value, tuple):
        return (_renamed(value[0], token),) + value[1:]
    if isinstance(value, list):
        return [(sh, _rename_value(v, token)) for sh, v in value]
    return value


def resolve_value(value, layout) -> int:
    """An operand value as an integer, references looked up in ``layout.address``."""
    if value is None:
        return 0
    if isinstance(value, tuple):
        return (layout.address(value[0]) + value[1]) >> (value[2] if len(value) > 2 else 0)
    if isinstance(value, list):
        return sum(resolve_value(v, layout) << sh for sh, v in value)
    return int(value)


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


def _chunk_env(chunk: int):
    """Accessors of a chunk's per-token values in the environment: with one
    token the plain arrays, with more the token's slice of a ``[T, ...]``
    array (or entry of a list)."""
    def get(e, name, t):
        return e[name][t] if chunk > 1 else e[name]

    def put(e, name, t, value):
        if chunk == 1:
            e[name] = value
            return
        try:
            holder = e[name]
        except KeyError:
            holder = np.zeros((chunk,) + np.shape(value), dtype=np.int64) if isinstance(value, np.ndarray) else [None] * chunk
            e[name] = holder
        holder[t] = value
    return get, put


def _contrib(name: str, chunk: int) -> str:
    """A per-token step's produce: the whole buffer for one token, a contribution to it in a chunk."""
    return name if chunk == 1 else "+" + name


def recurrent_program(cfg, c: L.RecurrentConsts | None, spec: TileSpec, mm: MemoryMap, t: Timing = Timing(),
                      chunk: int = 1) -> list[Step]:
    """One token through a recurrent layer, or a chunk of ``chunk``
    consecutive tokens of one context (prefill): the passes carry the whole
    chunk at once, the vector units and state engines take the tokens in
    turn, and each head's state is read and written once per chunk.

    Inputs in the environment: ``x`` (int16 residual; ``[chunk, d]`` for a
    chunk), ``s_mem[h]`` (the state rows of each head) and, for the int8
    state, ``scale_mem[h]`` (its scale beat), ``hist_mem`` (the conv
    history).  With ``c`` None the program has its shape and timing but
    cannot be run.  Every step carries its operands for the layer engine
    (``recurrent_layout``)."""
    nk, nv, hk, hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    d, kd, vd = cfg.hidden_size, nk * hk, nv * hv
    conv_dim, ffn = 2 * kd + vd, cfg.layer_intermediate_size(0)
    n_delta = UNITS["delta"][1]
    repeat = nv // nk
    int8_state = mm.state_bits == 8
    head_bytes = hk * hv * mm.state_bits // 8 + (BEAT if int8_state else 0)   # the rows and the scale beat
    lay = recurrent_layout(cfg, spec, mm, chunk)
    off_z, off_b, off_a, p1 = lay["off_z"], lay["off_b"], lay["off_a"], lay["p1"]
    get, put = _chunk_env(chunk)
    T = chunk
    steps: list[Step] = []

    def add(name, unit, src, dst, cycles, func=None, engine=0, nbytes=0, ops=None):
        steps.append(Step(name, unit, engine, tuple(src), tuple(dst), int(cycles), func, nbytes, ops=ops))

    def fabric(q):
        return lambda x: L._fabric(q, x, spec)

    def hist_name(i: int) -> str:
        return "hist" if i == 0 else ("hist_next" if i == T else f"hist[{i}]")

    def tok(name: str, i: int) -> str:                    # a per-token step's name
        return name if chunk == 1 else f"{name}<{i}>"

    hist_bytes = _beats(conv_dim * (cfg.linear_conv_kernel - 1)) * BEAT
    hist_beats = lay["sizes"]["hist"] // BEAT
    add("dma.hist_rd", "mem", (), ("hist",), t.memory(hist_bytes, 2048), lambda e: e.__setitem__("hist", e["hist_mem"]), nbytes=hist_bytes,
        ops=operands(src=("m_hist", 0), dst=("hist", 0), arg=MEM_RD, len=hist_beats))
    for i in range(T):
        add(tok("norm.h", i), "norm", ("x",), (_contrib("A", chunk),), t.norm(d), lambda e, i=i: put(e, "A", i, L._norm(get(e, "x", i), c.norm)),
            ops=operands(src=("x", i * 2 * d), dst=("A", i * d), arg=NORM_RESIDUAL | NORM_INT16, len=d // NL))

    def in_proj(e):                                     # one buffer P1 per token: qkv | z | b, a accumulators
        for i in range(T):
            a = get(e, "A", i)
            put(e, "qkv", i, fabric(c.in_proj_qkv)(a)[1])
            put(e, "z", i, fabric(c.in_proj_z)(a)[1])
            put(e, "b_acc", i, fabric(c.in_proj_b)(a)[0])
            put(e, "a_acc", i, fabric(c.in_proj_a)(a)[0])
    add("pass.in_proj", "tiles", ("A",), ("P1",), t.tile_pass(d), in_proj,
        ops=operands(src=("A", 0), dst=("P1", 0), arg=[(0, 0), (8, -(-d // spec.rows)), (16, T)], a2=d, a3=p1))

    for i in range(T):
        def conv(e, i=i):
            y, nxt = L.conv_silu_int(e[hist_name(i)], get(e, "qkv", i), c.conv_w, c.conv_mult_in, c.conv_sh_in,
                                     c.conv_mult_out, c.conv_sh_out)
            put(e, "conv", i, y)
            e[hist_name(i + 1)] = nxt
        add(tok("conv", i), "conv", ("P1", hist_name(i)), (_contrib("conv", chunk), hist_name(i + 1)),
            -(-conv_dim // t.l_conv) + t.conv_latency, conv,
            ops=operands(src=("P1", i * p1), dst=("conv", i * conv_dim), a2=(hist_name(i), 0), a3=(hist_name(i + 1), 0), len=conv_dim // CL))
    add("dma.hist_wr", "mem", ("hist_next",), (), t.memory(hist_bytes, 2048), lambda e: e.__setitem__("hist_mem", e["hist_next"]), nbytes=hist_bytes,
        ops=operands(src=("hist_next", 0), dst=("m_hist", 0), arg=MEM_WR, len=hist_beats))
    for i in range(T):
        def gates(e, i=i):
            decay, beta = L.head_gates_int(get(e, "a_acc", i)[:nv], get(e, "b_acc", i)[:nv], c.gate_mult_a, c.gate_sh_a,
                                           c.gate_mult_b, c.gate_sh_b, c.a_coef, c.dt_bias)
            put(e, "decay", i, decay)
            put(e, "beta", i, beta)
        add(tok("gates", i), "gates", ("P1",), (_contrib("gates", chunk),), nv + t.gates_latency, gates,
            ops=operands(src=("P1", i * p1 + off_b), a2=("P1", i * p1 + off_a), dst=("gates", i * lay["gates"]), len=nv))
    for i in range(T):
        for j in range(nk):
            for which, off in (("q", 0), ("k", kd)):
                def unit_norm(e, i=i, j=j, off=off, which=which):
                    put(e, f"{which}_unit[{j}]", i, L._norm(get(e, "conv", i)[off + j * hk:off + (j + 1) * hk], c.unit_norm))
                add(tok(f"norm.{which}[{j}]", i), "norm", ("conv",), (f"+qk_unit[{j}]",), t.norm(hk), unit_norm,
                    engine=j % UNITS["norm"][1],
                    ops=operands(src=("conv", i * conv_dim + off + j * hk), dst=(f"qk_unit[{j}]", i * 2 * hk + (hk if which == "k" else 0)),
                                 arg=NORM_UNIT, len=hk // NL))
    # The heads over the state engines, each engine alternating two state slots; a chunk's tokens run on the slot in turn.
    slot_beats = lay["sizes"][_slot(0, 0)] // BEAT
    for h in range(nv):
        e_id, k = h % n_delta, h // n_delta
        slot = _slot(e_id, k)
        def s_rd(e, h=h, slot=slot):                       # the slot holds the rows and, for int8, the scale beat
            e[slot] = (e["s_mem"][h], tuple(int(x) for x in e["scale_mem"][h])) if int8_state else e["s_mem"][h]
        add(f"dma.s_rd[{h}]", "mem", (), (slot,), t.memory(head_bytes, 2048), s_rd, nbytes=head_bytes,
            ops=operands(src=(f"m_s[{h}]", 0), dst=(slot, 0), arg=MEM_RD, len=slot_beats))
        for i in range(T):
            def delta(e, h=h, slot=slot, i=i):
                v = get(e, "conv", i)[2 * kd + h * hv:2 * kd + (h + 1) * hv]
                k_unit, q_unit = get(e, f"k_unit[{h // repeat}]", i), get(e, f"q_unit[{h // repeat}]", i)
                decay, beta = int(get(e, "decay", i)[h]), int(get(e, "beta", i)[h])
                if int8_state:
                    t_new, *scale_new, y = L.delta_state_int8(e[slot][0], k_unit, v, q_unit, decay, beta, *e[slot][1])
                    e[slot] = (t_new, tuple(scale_new))
                else:
                    e[slot], y = L.delta_state_int(e[slot], k_unit, v, q_unit, decay, beta)
                put(e, f"y[{h}]", i, y)
            add(tok(f"delta[{h}]", i), "delta", (slot, f"qk_unit[{h // repeat}]", "conv", "gates"),
                (slot, _contrib(f"y[{h}]", chunk)), 2 * hk + hk + 4 + t.delta_latency, delta, engine=e_id,
                ops=operands(src=(f"qk_unit[{h // repeat}]", i * 2 * hk), dst=(f"y[{h}]", i * 2 * hv), a2=("conv", i * conv_dim + 2 * kd + h * hv),
                             a3=(slot, 0), arg=("gates", i * lay["gates"] + 4 * h)))

        def s_wr(e, h=h, slot=slot):
            if int8_state:
                e["s_mem"][h], e["scale_mem"][h] = e[slot]
            else:
                e["s_mem"][h] = e[slot]
        add(f"dma.s_wr[{h}]", "mem", (slot,), (), t.memory(head_bytes, 2048), s_wr, nbytes=head_bytes,
            ops=operands(src=(slot, 0), dst=(f"m_s[{h}]", 0), arg=MEM_WR, len=slot_beats))
        for i in range(T):
            def gnorm(e, h=h, i=i):
                gate = L.silu_fixed(L.requant(get(e, "z", i)[h * hv:(h + 1) * hv], c.z_mult, c.z_shift, 16))
                put(e, f"y_norm[{h}]", i, L._norm(get(e, f"y[{h}]", i), c.gated_norm, gain=gate))
            add(tok(f"gnorm[{h}]", i), "norm", (f"y[{h}]", "P1"), ("+y_norm",), -(-hv // t.l_vec) + t.silu_latency + t.norm(hv), gnorm,
                engine=h % UNITS["norm"][1],
                ops=operands(src=(f"y[{h}]", i * 2 * hv), dst=("y_norm", i * vd + h * hv), arg=NORM_GATE | NORM_INT16 | NORM_GATED,
                             a2=("P1", i * p1 + off_z + h * hv), len=hv // NL))

    def out_proj(e):
        for i in range(T):
            y_norm = np.concatenate([get(e, f"y_norm[{h}]", i) for h in range(nv)])
            put(e, "mixer", i, fabric(c.out_proj)(y_norm)[1])
    add("pass.out_proj", "tiles", ("y_norm",), ("mixer",), t.tile_pass(vd), out_proj,
        ops=operands(src=("y_norm", 0), dst=("mixer", 0), arg=[(0, 1), (8, -(-vd // spec.rows)), (16, T)], a2=vd, a3=d))
    for i in range(T):
        add(tok("residual.1", i), "residual", ("x", "mixer"), (_contrib("x1", chunk),), -(-d // t.l_vec) + t.residual_latency,
            lambda e, i=i: put(e, "x1", i, L.residual_int(get(e, "x", i), get(e, "mixer", i), c.res_mult, c.res_shift)),
            ops=operands(src=("x", i * 2 * d), a2=("mixer", i * d), arg=0, dst=("x1", i * 2 * d), len=d // NL))
    _ffn_steps(add, c.ffn if c is not None else None, spec, d, ffn, t, chunk)
    link(steps)
    return steps


def recurrent_layout(cfg, spec: TileSpec, mm: MemoryMap, chunk: int = 1) -> dict:
    """The recurrent program's buffers as the layer engine holds them: byte
    sizes of every vector-buffer name and memory-image name (``m_...``), the
    offsets inside one token's pass-1 output ``P1`` (``qkv | z | b words | a
    words``) and its size ``p1``, and the bytes of one token's gates.  A
    chunk's buffers hold its tokens' vectors in turn.  The int8 state's slot
    is a header beat then ``K`` rows of ``V`` bytes; the conv history keeps
    ``HIST_REC`` bytes per channel, one buffer per token boundary."""
    nk, nv, hk, hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    d, kd, vd = cfg.hidden_size, nk * hk, nv * hv
    conv_dim, ffn = 2 * kd + vd, cfg.layer_intermediate_size(0)
    assert cfg.linear_conv_kernel - 1 <= HIST_REC
    align = lambda n: -(-n // BEAT) * BEAT
    off_z = 2 * kd + vd
    off_b = align(off_z + vd)
    off_a = align(off_b + 4 * nv)
    p1 = align(off_a + 4 * nv)
    gates = align(4 * nv)
    slot = (SLOT_HEADER + hk * hv) if mm.state_bits == 8 else 2 * hk * hv
    T = chunk
    sizes = {"x": T * 2 * d, "A": T * d, "P1": T * p1, "hist": align(conv_dim * HIST_REC), "conv": T * conv_dim,
             "hist_next": align(conv_dim * HIST_REC), "gates": T * gates, "y_norm": T * vd, "mixer": T * d, "x1": T * 2 * d,
             "A2": T * d, "GU": T * 2 * ffn, "act": T * ffn, "ffn": T * d, "x2": T * 2 * d, "m_hist": align(conv_dim * HIST_REC)}
    for i in range(1, T):
        sizes[f"hist[{i}]"] = align(conv_dim * HIST_REC)
    for i in range(nk):
        sizes[f"qk_unit[{i}]"] = T * 2 * hk                    # a token's unit q then its unit k
    for j in range(2 * UNITS["delta"][1]):
        sizes[_slot(j // 2, j % 2)] = slot
    for h in range(nv):
        sizes[f"y[{h}]"] = T * 2 * hv
        sizes[f"m_s[{h}]"] = slot
    return {"sizes": sizes, "off_z": off_z, "off_b": off_b, "off_a": off_a, "p1": p1, "gates": gates}


def _ffn_steps(add, f: L.FfnConsts | None, spec: TileSpec, d: int, ffn: int, t: Timing, chunk: int = 1) -> None:
    get, put = _chunk_env(chunk)
    T = chunk

    def tok(name: str, i: int) -> str:
        return name if chunk == 1 else f"{name}<{i}>"

    for i in range(T):
        add(tok("norm.h2", i), "norm", ("x1",), (_contrib("A2", chunk),), t.norm(d), lambda e, i=i: put(e, "A2", i, L._norm(get(e, "x1", i), f.norm)),
            ops=operands(src=("x1", i * 2 * d), dst=("A2", i * d), arg=NORM_FFN | NORM_INT16, len=d // NL))

    def gate_up(e):
        for i in range(T):
            put(e, "gate_ffn", i, L._fabric(f.gate_proj, get(e, "A2", i), spec)[1])
            put(e, "up_ffn", i, L._fabric(f.up_proj, get(e, "A2", i), spec)[1])
    add("pass.gate_up", "tiles", ("A2",), ("GU",), t.tile_pass(d), gate_up,
        ops=operands(src=("A2", 0), dst=("GU", 0), arg=[(0, 2), (8, -(-d // spec.rows)), (16, T)], a2=d, a3=2 * ffn))
    for i in range(T):
        add(tok("swiglu", i), "swiglu", ("GU",), (_contrib("act", chunk),), -(-ffn // t.l_vec) + t.swiglu_latency,
            lambda e, i=i: put(e, "act", i, L.swiglu_int(get(e, "gate_ffn", i), get(e, "up_ffn", i), f.mult_g, f.sh_g, f.mult_o, f.sh_o)),
            ops=operands(src=("GU", i * 2 * ffn), a2=("GU", i * 2 * ffn + ffn), arg=0, dst=("act", i * ffn), len=ffn // NL))

    def down(e):
        for i in range(T):
            put(e, "ffn", i, L._fabric(f.down_proj, get(e, "act", i), spec)[1])
    add("pass.down", "tiles", ("act",), ("ffn",), t.tile_pass(ffn), down,
        ops=operands(src=("act", 0), dst=("ffn", 0), arg=[(0, 3), (8, -(-ffn // spec.rows)), (16, T)], a2=ffn, a3=d))
    for i in range(T):
        add(tok("residual.2", i), "residual", ("x1", "ffn"), (_contrib("x2", chunk),), -(-d // t.l_vec) + t.residual_latency,
            lambda e, i=i: put(e, "x2", i, L.residual_int(get(e, "x1", i), get(e, "ffn", i), f.res_mult, f.res_shift)),
            ops=operands(src=("x1", i * 2 * d), a2=("ffn", i * d), arg=1, dst=("x2", i * 2 * d), len=d // NL))


def global_program(cfg, c: L.GlobalConsts | None, spec: TileSpec, mm: MemoryMap, pos: int, t: Timing = Timing(),
                   chunk: int = 1) -> list[Step]:
    """One token through a global layer at position ``pos``, or a chunk of
    ``chunk`` consecutive tokens from ``pos`` (prefill): the passes carry
    the chunk, everything else takes the tokens in turn.  Inputs in the
    environment: ``x`` (``[chunk, d]`` for a chunk), and the memory side as
    ``k_rows``/``v_rows`` ``[kv_heads, N, hd]`` (the window then the
    retrieved blocks, as the record reader would stream them; a list per
    token for a chunk).  Every step carries its operands for the layer
    engine (``global_layout``); the memory steps name the context's memory
    image ``m_ctx`` by page and the position."""
    nh, nkv, hd, rd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rotary_dim
    d, ffn, group, idim = cfg.hidden_size, cfg.layer_intermediate_size(cfg.global_layer_offset), nh // nkv, cfg.index_dim
    lay = global_layout(cfg, spec, mm, chunk)
    off_k, off_v, off_iq, off_ik, p1 = lay["off_k"], lay["off_v"], lay["off_iq"], lay["off_ik"], lay["p1"]
    rows_bytes, sel_bytes = lay["rows"], lay["sel"]
    ctx = ("m_ctx", 0, MEM_PAGE_SHIFT)                     # the context's memory page, in a3
    get, put = _chunk_env(chunk)
    T = chunk
    steps: list[Step] = []

    def add(name, unit, src, dst, cycles, func=None, engine=0, nbytes=0, ops=None):
        steps.append(Step(name, unit, engine, tuple(src), tuple(dst), int(cycles), func, nbytes, ops=ops))

    def tok(name: str, i: int) -> str:
        return name if chunk == 1 else f"{name}<{i}>"

    for i in range(T):
        add(tok("norm.h", i), "norm", ("x",), (_contrib("A", chunk),), t.norm(d), lambda e, i=i: put(e, "A", i, L._norm(get(e, "x", i), c.norm)),
            ops=operands(src=("x", i * 2 * d), dst=("A", i * d), arg=NORM_RESIDUAL | NORM_INT16, len=d // NL))

    def qkv(e):                                         # one buffer P1 per token: q, gate | k | v | index_q | index_k
        for i in range(T):
            a = get(e, "A", i)
            qg = L._fabric(c.q_proj, a, spec)[1].reshape(nh, 2 * hd)
            put(e, "q_raw", i, qg[:, :hd])
            put(e, "gate", i, qg[:, hd:])
            put(e, "k_raw", i, L._fabric(c.k_proj, a, spec)[1].reshape(nkv, hd))
            put(e, "v", i, L._fabric(c.v_proj, a, spec)[1].reshape(nkv, hd))
            put(e, "index_q", i, L._fabric(c.index_q, a, spec)[1])
            put(e, "index_k", i, L._fabric(c.index_k, a, spec)[1])
    add("pass.qkv", "tiles", ("A",), ("P1",), t.tile_pass(d), qkv,
        ops=operands(src=("A", 0), dst=("P1", 0), arg=[(0, 0), (8, -(-d // spec.rows)), (16, T)], a2=d, a3=p1))
    for i in range(T):
        add(tok("rotary.table", i), "rotary", (), (_contrib("rot", chunk),), rd // 2 + t.rotary_latency,
            lambda e, i=i: put(e, "rot", i, L.rotary_table_int(pos + i, c.inv_freq)),
            ops=operands(dst=("rot", i * 2 * rd), arg=ROT_TABLE, a3=pos + i, len=rd // 2))
        add(tok("norm.index_q", i), "norm", ("P1",), (_contrib("iq", chunk),), t.norm(idim),
            lambda e, i=i: put(e, "index_q_unit", i, L._norm(get(e, "index_q", i), c.unit_norm)),
            ops=operands(src=("P1", i * p1 + off_iq), dst=("iq", i * idim), arg=NORM_UNIT, len=idim // NL))
    rot_cycles = t.norm(hd) + -(-hd // t.l_vec) + t.rotary_latency
    for i in range(T):
        for n in range(nkv):
            def krot(e, n=n, i=i):
                put(e, f"k[{n}]", i, L.rotary_int(L._norm(get(e, "k_raw", i)[n], c.k_norm), *get(e, "rot", i), rd, c.rot_mult_k, c.rot_sh_k))
            add(tok(f"rotary.k[{n}]", i), "rotary", ("P1", "rot"), ("+k",), rot_cycles, krot, engine=n % UNITS["rotary"][1],
                ops=operands(src=("P1", i * p1 + off_k + n * hd), dst=("k", i * nkv * hd + n * hd), arg=ROT_HEAD | (1 << 4),
                             a2=("rot", i * 2 * rd), len=hd // NL))
        for h in range(nh):
            def qrot(e, h=h, i=i):
                put(e, f"q[{h}]", i, L.rotary_int(L._norm(get(e, "q_raw", i)[h], c.q_norm), *get(e, "rot", i), rd, c.rot_mult_q, c.rot_sh_q))
            add(tok(f"rotary.q[{h}]", i), "rotary", ("P1", "rot"), (f"+qg[{h // group}]",), rot_cycles, qrot, engine=h % UNITS["rotary"][1],
                ops=operands(src=("P1", i * p1 + h * 2 * hd), dst=(f"qg[{h // group}]", i * group * hd + (h % group) * hd), arg=ROT_HEAD,
                             a2=("rot", i * 2 * rd), len=hd // NL))
    # The memory side, token by token: append this token's records, scan the index, then stream rows to the cores.
    row_cycles = 2 * -(-hd // t.l_attn) + t.attn_exp_stall
    for i in range(T):
        p = pos + i
        append_bytes = nkv * mm.kv_record_bytes + (nkv * mm.kv_record_bytes + mm.index_record_bytes) // mm.block + 2 * mm.sums_bytes
        add(tok("mem.append", i), "mem", ("k", "P1"), (), t.memory(append_bytes, mm.kv_record_bytes), nbytes=append_bytes,
            ops=operands(src=("k", i * nkv * hd), dst=("P1", i * p1 + off_ik), a2=("P1", i * p1 + off_v), a3=ctx,
                         arg=[(0, MEM_APPEND), (4, p)]))
        # The scan reads the index a page of records per request; the rows are the
        # window (head-major, page bursts) and one mean record per selected block.
        eligible = eligible_blocks(p, mm.local_window, mm.block)
        scan_bytes = eligible * mm.index_record_bytes
        add(tok("mem.scan", i), "mem", ("iq",), (_contrib("sel", chunk),), t.memory(scan_bytes, mm.index_burst_records * mm.index_record_bytes),
            lambda e, i=i: put(e, "selected", i, None), nbytes=scan_bytes,
            ops=operands(src=("iq", i * idim), dst=("sel", i * sel_bytes), a3=ctx, arg=[(0, MEM_SCAN), (4, p)]))
        n_window, n_blocks = min(p + 1, mm.local_window), min(cfg.top_blocks, eligible)
        rows = n_window + n_blocks
        for n in range(nkv):
            heads = list(range(n * group, (n + 1) * group))
            window_bytes, block_bytes = n_window * mm.kv_record_bytes, n_blocks * mm.kv_record_bytes

            def mem_rows(e, n=n, i=i):
                k_rows, v_rows = (e["k_rows"][i], e["v_rows"][i]) if chunk > 1 else (e["k_rows"], e["v_rows"])
                put(e, f"rows[{n}]", i, (k_rows[n], v_rows[n]))
            add(tok(f"mem.rows[{n}]", i), "mem", ("sel",), (_contrib(f"rows[{n}]", chunk),),
                t.memory(window_bytes, mm.window_burst_records * mm.kv_record_bytes) + t.memory(block_bytes, mm.kv_record_bytes),
                mem_rows, nbytes=window_bytes + block_bytes,
                ops=operands(src=("sel", i * sel_bytes), dst=(f"rows[{n}]", i * rows_bytes), a2=n, a3=ctx, arg=[(0, MEM_ROWS), (4, p)], len=rows))

            def attn(e, n=n, heads=heads, i=i):
                q = np.stack([get(e, f"q[{h}]", i) for h in heads])
                k_rows, v_rows = get(e, f"rows[{n}]", i)
                put(e, f"att[{n}]", i, L.attention_int(q, get(e, "gate", i)[heads], k_rows, v_rows, mult_s=c.mult_s, sh_s=c.sh_s,
                                                       mult_gate=c.mult_gate, sh_gate=c.sh_gate, mult_o=c.mult_o, sh_o=c.sh_o))
            add(tok(f"attn[{n}]", i), "attn", (f"qg[{n}]", "P1", f"rows[{n}]"), ("+att",),
                rows * row_cycles + group * (hd // t.l_attn) + t.attn_out_latency, attn, engine=n % UNITS["attn"][1],
                ops=operands(src=(f"qg[{n}]", i * group * hd), dst=("att", i * nh * hd + n * group * hd),
                             a2=("P1", i * p1 + n * group * 2 * hd + hd), a3=(f"rows[{n}]", i * rows_bytes), len=rows))

    def o_proj(e):
        for i in range(T):
            att = np.concatenate([get(e, f"att[{n}]", i) for n in range(nkv)]).reshape(nh * hd)
            put(e, "mixer", i, L._fabric(c.o_proj, att, spec)[1])
    add("pass.o_proj", "tiles", ("att",), ("mixer",), t.tile_pass(nh * hd), o_proj,
        ops=operands(src=("att", 0), dst=("mixer", 0), arg=[(0, 1), (8, -(-(nh * hd) // spec.rows)), (16, T)], a2=nh * hd, a3=d))
    for i in range(T):
        add(tok("residual.1", i), "residual", ("x", "mixer"), (_contrib("x1", chunk),), -(-d // t.l_vec) + t.residual_latency,
            lambda e, i=i: put(e, "x1", i, L.residual_int(get(e, "x", i), get(e, "mixer", i), c.res_mult, c.res_shift)),
            ops=operands(src=("x", i * 2 * d), a2=("mixer", i * d), arg=0, dst=("x1", i * 2 * d), len=d // NL))
    _ffn_steps(add, c.ffn if c is not None else None, spec, d, ffn, t, chunk)
    link(steps)
    return steps


def global_layout(cfg, spec: TileSpec, mm: MemoryMap, chunk: int = 1) -> dict:
    """The global program's buffers as the layer engine holds them: byte
    sizes of every vector-buffer name and of the context's memory image
    ``m_ctx`` (the map's window, block store, index and block sums), the
    offsets inside one token's pass-1 output ``P1`` (``q, gate per head | k
    | v | index_q | index_k``) and its size ``p1``, and one token's bytes of
    ``sel`` and of a head's ``rows``.  ``rot`` holds the sines then the
    cosines of the rotary frequencies as int16; ``sel`` the count then the
    ids of the selected blocks as int16; ``rows[n]`` the head's key and
    value records as int8.  A chunk's buffers hold its tokens' vectors in
    turn."""
    nh, nkv, hd, rd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rotary_dim
    d, ffn, group, idim = cfg.hidden_size, cfg.layer_intermediate_size(cfg.global_layer_offset), nh // nkv, cfg.index_dim
    align = lambda n: -(-n // BEAT) * BEAT
    off_k = nh * 2 * hd
    off_v = off_k + nkv * hd
    off_iq = off_v + nkv * hd
    off_ik = off_iq + idim
    p1 = align(off_ik + idim)
    sel = 2 * (1 + cfg.top_blocks)
    rows = (mm.local_window + cfg.top_blocks) * 2 * hd
    T = chunk
    sizes = {"x": T * 2 * d, "A": T * d, "P1": T * p1, "rot": T * 2 * rd, "iq": T * idim, "k": T * nkv * hd,
             "sel": T * sel, "att": T * nh * hd, "mixer": T * d, "x1": T * 2 * d, "A2": T * d, "GU": T * 2 * ffn, "act": T * ffn,
             "ffn": T * d, "x2": T * 2 * d, "m_ctx": mm.context_bytes}
    for n in range(nkv):
        sizes[f"qg[{n}]"] = T * group * hd
        sizes[f"rows[{n}]"] = T * rows
    return {"sizes": sizes, "off_k": off_k, "off_v": off_v, "off_iq": off_iq, "off_ik": off_ik, "p1": p1, "sel": sel, "rows": rows}


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
        ops = None if s.ops is None else {k: _rename_value(v, token) for k, v in s.ops.items()}
        out.append(Step(s.name, s.unit, s.engine, tuple(_renamed(n, token) for n in s.src),
                        tuple(_renamed(n, token) for n in s.dst), s.cycles, s.func, s.nbytes, token, ops=ops))
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


def encode(steps: list[Step], layout=None) -> list[int]:
    """Each step as a 256-bit word: unit, engine, last, the length the unit
    is given, a 32-bit argument, four 30-bit address operands (source,
    destination, two more), then up to six consumed buffer ids, two
    produced ids and their contribution bits (0xFF is no buffer).  Without
    a layout the length is the step's cycles and the operand fields are
    zero (the stub units); with one (``engine.Layout``) the fields are the
    step's operands."""
    ids = buffer_ids(steps)
    words = []
    for i, step in enumerate(steps):
        assert step.cycles < (1 << 16) and step.engine < 16
        ops = step.ops if layout is not None and step.ops is not None else {}
        length = resolve_value(ops["len"], layout) if "len" in ops else step.cycles
        fields = {k: resolve_value(ops.get(k), layout) for k in ("arg", "src", "dst", "a2", "a3")}
        assert length < (1 << 16) and fields["arg"] < (1 << 32), step.name
        assert all(fields[k] < (1 << ADDR_BITS) for k in ("src", "dst", "a2", "a3")), step.name
        w = UNITS[step.unit][0] | (step.engine << 4) | (int(i == len(steps) - 1) << 8) | (length << 16)
        w |= fields["arg"] << 32
        for k, name in enumerate(("src", "dst", "a2", "a3")):
            w |= fields[name] << (64 + ADDR_BITS * k)
        consume = [ids[n] for n in step.src] + [0xFF] * (MAX_CONSUME - len(step.src))
        produce = [ids[_plain(n)] for n in step.dst] + [0xFF] * (MAX_PRODUCE - len(step.dst))
        contrib = sum(int(n.startswith("+")) << k for k, n in enumerate(step.dst))
        base = 64 + 4 * ADDR_BITS
        for k, c in enumerate(consume):
            w |= c << (base + 8 * k)
        for k, p in enumerate(produce):
            w |= p << (base + 8 * MAX_CONSUME + 8 * k)
        w |= contrib << (base + 8 * MAX_CONSUME + 8 * MAX_PRODUCE)
        assert w < (1 << 256)
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

def report_markdown(cfg, mm: MemoryMap, pos: int, t: Timing = Timing(), chunk: int = 8) -> str:
    """The two layers' schedules at a model's geometry, one token at a time, as a stream, and as a prefill chunk."""
    spec = TileSpec()
    out = [f"# Token sequencer schedule: hidden {cfg.hidden_size}, position {pos}", ""]
    for title, steps, chunked in (("Recurrent layer", recurrent_program(cfg, None, spec, mm, t), recurrent_program(cfg, None, spec, mm, t, chunk)),
                                  ("Global layer", global_program(cfg, None, spec, mm, pos, t), global_program(cfg, None, spec, mm, pos, t, chunk))):
        sched = schedule(steps)
        interval = token_interval(steps)
        pre = schedule(chunked)
        out += [f"## {title}: {len(steps)} steps", "", sched.report(t), "",
                f"stream of tokens: {interval} cycles per token ({interval / t.core_mhz:.1f} us), "
                f"{100 * sched.busy('mem') / interval:.0f}% of the memory port", "",
                f"prefill, chunk of {chunk}: {pre.cycles // chunk} cycles per token ({pre.cycles / chunk / t.core_mhz:.1f} us), "
                f"{sum(s.nbytes for s in chunked) // chunk // 1024} KB of memory traffic per token, {len(chunked)} steps", ""]
    return "\n".join(out)


def main() -> None:
    from fixed_llm_poc import ASICLMConfig
    cfg = ASICLMConfig.qwen3_5_9b()
    mm = MemoryMap.from_config(cfg)
    print(report_markdown(cfg, mm, mm.context_tokens - 1))
    print(report_markdown(cfg, mm, 4095))


if __name__ == "__main__":
    main()
