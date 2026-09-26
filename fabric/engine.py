"""The layer engine: the sequencer wired to the real units.

``rtl/fabric_engine.sv`` puts the controller of ``sequencer.py`` in front
of the tile array, the vector units, the state engines and a memory port,
all sharing one byte-addressed vector buffer.  Each unit sits behind an
*adapter* that turns a command (source and destination addresses, two
arguments, a length) into the unit's start pulses and beat streams and
returns the step's tag when its last result is in the buffer.  This module
writes everything that engine loads: the program image with its operands
resolved to addresses (``Layout``), the tiles' ROM images and the pass
table, the constant tables of the vector units, the initial vector buffer
and memory images, and the results the run must reproduce bit for bit
(``run_program`` on the same steps).

Both layers run: the recurrent program with the state and history in
memory, the global program with the context's window, block store, index
and block sums.  The memory behind the port is ``fabric_memory.sv``'s
behavioural model in the testbench; the HPI bridge that replaces it on the
die is checked on its own (``tb_mem_bridge``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fabric import hpi
from fabric import layer as L
from fabric import sequencer as S
from fabric.memory import BEAT, MemoryMap
from fabric.tile import TileSpec, compile_matrix, write_hex

NPASS = 4                    # the recurrent layer's passes: in_proj, out_proj, gate_up, down
# rtl/fabric_engine.sv's ATT_L: the attention cores' and the record reader's
# lanes, which must divide the head because they stream HD / L beats with no
# ragged last one.  `sequencer.Timing.attn_lanes` is the same rule.
TAB_BITS = 56                # a pass-table entry


def _refs(value) -> list[str]:
    if isinstance(value, tuple):
        return [value[0]]
    if isinstance(value, list):
        return [n for _, v in value for n in _refs(v)]
    return []


PAGE_BEATS = 1 << S.MEM_PAGE_SHIFT


# --------------------------------------------------------------------------
# Banking the vector buffer
# --------------------------------------------------------------------------

def live_together(steps: list[S.Step]) -> list[set[int]]:
    """For each step, the later steps that can be in flight with it.

    The controller issues in program order, so step *j* runs beside step *i*
    only if every step between them could issue while *i* was running: one
    that waits for *i*, directly or through another step, or that wants the
    engine *i* is on, stops *j* and everything after it.  This is the
    relation the buffers' banking has to survive, and it does not depend on
    how long any unit takes, which is why it can be decided here rather than
    read off a simulation."""
    after: list[set[int]] = []
    deep: list[set[int]] = []
    for i, step in enumerate(steps):
        d = set(step.deps)
        for k in step.deps:
            d |= deep[k]
        deep.append(d)
    for i, step in enumerate(steps):
        port, live = (step.unit, step.engine), set()
        for j in range(i + 1, len(steps)):
            if i in deep[j] or (steps[j].unit, steps[j].engine) == port:
                break
            live.add(j)
        after.append(live)
    return after


# Which vector buffer each of an adapter's ports holds while it runs a
# command.  The two-port units take one operand each -- and for the gates and
# SwiGLU both operands are one buffer at two offsets, which is two reads of one
# bank that no assignment can separate -- while the pass adapter takes a port
# per token of a chunk, all of the same buffer.  Every other unit has a single
# port that visits the command's buffers in turn, so its own reads never
# collide.  These follow rtl/fabric_engine.sv's address assignments.
RD_FIELDS = {"norm": ("src", "a2"), "conv": ("src", "a2"), "gates": ("src", "a2"),
             "swiglu": ("src", "a2"), "residual": ("src", "a2")}
WR_FIELDS = {"conv": ("dst", "a3")}


def _vb_ref(ops: dict, field: str) -> str | None:
    names = [n for n in _refs(ops.get(field)) if not n.startswith(S.MEM_PREFIX)]
    return names[0] if names else None


def ports(step: S.Step, chunk: int = 1) -> tuple[list[str], list[str], list[str], list[str]]:
    """A step's ports: the buffers its fixed read ports hold, the buffers its
    one shared read port visits, and the same for writes."""
    ops = step.ops or {}
    if step.unit == "tiles":
        rd_fixed, rd_shared = [_vb_ref(ops, "src")] * chunk, []
    elif step.unit in RD_FIELDS:
        rd_fixed, rd_shared = [_vb_ref(ops, f) for f in RD_FIELDS[step.unit]], []
    else:
        rd_fixed, rd_shared = [], [n for n in step.src if not n.startswith(S.MEM_PREFIX)]
    if step.unit in WR_FIELDS:
        wr_fixed, wr_shared = [_vb_ref(ops, f) for f in WR_FIELDS[step.unit]], []
    else:
        wr_fixed = []
        wr_shared = [S._plain(n) for n in step.dst if not S._plain(n).startswith(S.MEM_PREFIX)]
        own = _vb_ref(ops, "dst")                    # the memory unit's append writes back
        if own and own not in wr_shared:
            wr_shared.append(own)
    return ([n for n in rd_fixed if n], rd_shared, [n for n in wr_fixed if n], wr_shared)


# The crossbar's logical read and write ports, in the order rtl/fabric_engine.sv
# wires them: unit -> (ports per engine, engines).  ``None`` ports is a port
# per token of a chunk: the pass adapter reads one row per token at once
# (`R_CONV = R_TILES + TMAX` in the RTL), so every read port after it moves
# when the chunk does.  It writes its result a beat at a time, through one
# port, so the write side does not move.
RD_PORT_MAP = (("norm", 2, 2), ("tiles", None, 1), ("conv", 2, 1), ("gates", 2, 1), ("delta", 1, 4),
               ("swiglu", 2, 1), ("residual", 2, 1), ("mem", 1, 1), ("rotary", 1, 2), ("attn", 1, 4), ("link", 1, 1))
WR_PORT_MAP = (("norm", 1, 2), ("tiles", 1, 1), ("conv", 2, 1), ("gates", 1, 1), ("delta", 1, 4),
               ("swiglu", 1, 1), ("residual", 1, 1), ("mem", 1, 1), ("rotary", 1, 2), ("attn", 1, 4), ("link", 1, 1))


def _port_index(table, chunk: int = 1) -> tuple[dict, int]:
    index, base = {}, 0
    for unit, per, engines in table:
        if per is None:
            per = chunk
        for engine in range(engines):
            for k in range(per):
                index[(unit, engine, k)] = base + engine * per + k
        base += per * engines
    return index, base


def port_colours(programs: list[S.Step] | list[list[S.Step]], write: bool = False, chunk: int = 1) -> tuple[list[int], int]:
    """Which crossbar port each of the engine's logical ports may share.

    The face is sized by construction -- every adapter operand got a port of
    its own, 26 reads and 19 writes at a chunk of three -- but the controller
    issues in order, so most of them can never be asking at once.  Two logical
    ports may share a crossbar port when no command of one can be in flight
    with a command of the other; a port an adapter reads beside its own never can.
    That is the same relation the banks are coloured by, over ports instead of
    buffers, and it is what makes the crossbar a fraction of the face: the
    fold is an or of the sharers' addresses, and the answer comes back on one
    set of wires that all of them read.

    Several programs may be given, and the face then survives all of them:
    the conflicts are their union, which is what an engine built for more
    than one program shape needs."""
    shapes = programs if programs and isinstance(programs[0], list) else [programs]
    index, total = _port_index(WR_PORT_MAP if write else RD_PORT_MAP, chunk)
    conflict: dict[int, set[int]] = {p: set() for p in range(total)}

    def edge(a: int, b: int) -> None:
        if a != b:
            conflict[a].add(b)
            conflict[b].add(a)

    def used(step: S.Step) -> list[int]:
        # A write port is asserted on the beat it writes, so a command that
        # writes no buffer of the vector buffer never holds one.  A read port
        # is the adapter's busy line (`rd_en = !ready` in the RTL), so a
        # command holds it for its whole run even when it reads nothing --
        # the rotary's table pass is one -- and the colouring has to say so.
        rf, rs, wf, ws = ports(step, chunk)
        if write and not (wf or ws):
            return []
        fixed = wf if write else rf
        n = len(fixed) if fixed else 1
        # The memory unit's engines are its commands in flight, not adapters of
        # their own: they share its one pair of ports.
        engine = 0 if step.unit == "mem" else step.engine
        return [index[(step.unit, engine, k)] for k in range(n) if (step.unit, engine, k) in index]

    for steps in shapes:
        live = live_together(steps)
        for i, step in enumerate(steps):
            for a in used(step):
                for j in live[i]:
                    for b in used(steps[j]):
                        edge(a, b)
    for (unit, engine, _), a in index.items():           # an adapter's own ports are asked for together
        for (unit2, engine2, _), b in index.items():
            if unit == unit2 and engine == engine2:
                edge(a, b)
            if unit == "link":                           # and the ring link's beside any of them: it has a port of its own
                edge(a, b)
    colour: dict[int, int] = {}
    for p in sorted(range(total), key=lambda q: -len(conflict[q])):
        taken = {colour[q] for q in conflict[p] if q in colour}
        colour[p] = next(c for c in range(total) if c not in taken)
    return [colour[p] for p in range(total)], max(colour.values()) + 1


def bank_conflicts(steps: list[S.Step], chunk: int = 1) -> dict[str, set[str]]:
    """Buffers that cannot share a bank.

    A bank has one write port, so two buffers written at once want different
    banks, and a read and a write of one bank in a cycle are free.  Reads are
    not a colouring question alone: two ports may want one buffer, which no
    assignment can separate, so the read ports are counted per bank afterwards
    by ``bank_reads`` rather than forbidden here."""
    graph: dict[str, set[str]] = {}

    def edge(a: str, b: str) -> None:
        graph.setdefault(a, set())
        graph.setdefault(b, set())
        if a != b:
            graph[a].add(b)
            graph[b].add(a)

    for step in steps:
        rf, rs, wf, ws = ports(step, chunk)
        for n in rf + rs + wf + ws:
            graph.setdefault(n, set())
        for group in (rf, wf):                       # a unit's own ports are concurrent
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    edge(group[a], group[b])
    live = live_together(steps)
    for i, step in enumerate(steps):
        rf, rs, wf, ws = ports(step, chunk)
        for j in live[i]:
            rf2, rs2, wf2, ws2 = ports(steps[j], chunk)
            for a in rf + rs:
                for b in rf2 + rs2:
                    edge(a, b)
            for a in wf + ws:
                for b in wf2 + ws2:
                    edge(a, b)
    return graph


def colour_banks(graph: dict[str, set[str]], sizes=None) -> dict[str, int]:
    """A bank per buffer, largest degree first, and among the banks a buffer
    may take, the one holding the fewest bytes: the banks are one memory
    each and the widest sets the address stride, so a balanced colouring is
    a smaller buffer.  Greedy is within a bank of the floor on every program
    measured, and the assignment is checked in the RTL, so a better
    colouring would buy nothing but a smaller number."""
    order = sorted(graph, key=lambda n: (-len(graph[n]), n))
    at: dict[str, int] = {}
    for name in order:                                   # first for the number of banks
        used = {at[m] for m in graph[name] if m in at}
        at[name] = next(b for b in range(len(used) + 1) if b not in used)
    banks = max(at.values()) + 1 if at else 1
    at, load = {}, {b: 0 for b in range(banks)}           # then for the balance, over that many
    for name in order:
        used = {at[m] for m in graph[name] if m in at}
        free = [b for b in load if b not in used]
        if not free:                                      # balance can cost a bank; take it
            free = [len(load)]
            load[free[0]] = 0
        at[name] = min(free, key=lambda b: (load[b], b))
        load[at[name]] += sizes(name) if sizes else 1
    return at


def bank_ports(steps: list[S.Step], at: dict[str, int], chunk: int = 1) -> tuple[list[int], list[int]]:
    """Read and write ports each bank needs: the most its buffers are asked
    for at once, over every set of steps that may be in flight together.  A
    fixed port charges the bank of the buffer it holds; a shared port charges
    one access to any bank its command's buffers are in, since which one it is
    on a given cycle is not known here.

    More than one is wanted only where two ports meet on one buffer, which no
    assignment of buffers to banks can undo.  Reading, that is the pass output
    the gates take two slices of.  Writing, it is a vector several engines
    contribute to at once -- the rotary heads' keys, the attention heads'
    output -- where each writes its own slice of one buffer."""
    nb = max(at.values()) + 1 if at else 1
    reads, writes = [1] * nb, [1] * nb
    live = live_together(steps)
    for i in range(len(steps)):
        run = [i]
        for j in sorted(live[i]):
            if all(j in live[k] for k in run):
                run.append(j)                            # a set that may all be in flight at once
        rb: dict[int, int] = {}
        wb: dict[int, int] = {}
        for k in run:
            rf, rs, wf, ws = ports(steps[k], chunk)
            for name in rf:
                rb[at[name]] = rb.get(at[name], 0) + 1
            for b in {at[n] for n in rs}:                # the one shared port, wherever it is
                rb[b] = rb.get(b, 0) + 1
            for name in wf:
                wb[at[name]] = wb.get(at[name], 0) + 1
            for b in {at[n] for n in ws}:
                wb[b] = wb.get(b, 0) + 1
        for b, n in rb.items():
            reads[b] = max(reads[b], n)
        for b, n in wb.items():
            writes[b] = max(writes[b], n)
    return reads, writes


class Layout:
    """Addresses of a program's buffers: vector-buffer names to byte
    offsets and memory-image names (``m_...``) to beat addresses, in order
    of first reference, each buffer beat-aligned in the vector buffer and
    page-aligned in memory.  A stream's names carry their token suffix;
    sizes are looked up by the plain name.

    A token's memory names are one region, its context's slot: page-aligned
    and contiguous, from ``base_page`` on.  The program carries a memory
    operand as its offset in the region and the engine adds the slot's page
    at run time (``slot_pages``), so one program image serves a context in
    any slot."""

    def __init__(self, steps: list[S.Step], sizes: dict[str, int], chunk: int = 1, base_page: int = 0,
                 programs: list[list[S.Step]] | None = None, lanes: list[list[S.Step]] | None = None) -> None:
        """``programs``, if given, are every program the engine will run from
        this one buffer placement -- a die's recurrent and global layers --
        and the buffers are placed and banked over all of them, so a name is
        at one address in each; the memory names are ``steps``'s own.

        ``lanes``, if given, are the lanes' programs (``sequencer.chain`` of
        each one's runs, its names carrying its suffix): each lane has banks
        of its own, since any step of another lane may run beside any of its
        steps, and every lane's are coloured alike, over the plain names of
        all of them."""
        self.sizes, self.chunk = sizes, chunk
        self.vb: dict[str, int] = {}
        self.mem: dict[str, int] = {}
        names: list[str] = []
        regions: dict[str, list[str]] = {}
        for step in steps:
            for value in (step.ops or {}).values():
                for name in _refs(value):
                    if name.startswith(S.MEM_PREFIX):
                        region = regions.setdefault(_suffix(name), [])
                        if name not in region:
                            region.append(name)
                    elif name not in names:
                        names.append(name)
        self.region: dict[str, int] = {}             # token suffix -> the first beat of its slot
        mem_next = base_page * PAGE_BEATS
        for sfx, region in regions.items():
            self.region[sfx] = mem_next
            for name in region:
                beats = -(-self.size(name) // BEAT)
                self.mem[name] = mem_next
                mem_next += -(-beats // PAGE_BEATS) * PAGE_BEATS
        self.mem_beats = mem_next
        if programs or lanes:                            # the names in the programs' order, so every layout of the set agrees
            names = []
        programs = programs or lanes or [steps]
        for prog in programs:
            for step in prog:
                for value in (step.ops or {}).values():
                    for name in _refs(value):
                        if not name.startswith(S.MEM_PREFIX) and name not in names:
                            names.append(name)
        if lanes:
            self._place_lanes(lanes, names)
        else:
            self._place(programs, names)

    def slot_pages(self) -> dict[int, int]:
        """Each token in flight's slot, as a page: what the engine is started with."""
        return {_token_index(sfx): start // PAGE_BEATS for sfx, start in self.region.items()}

    def operand(self, name: str) -> int:
        """What the program carries: a memory name's offset in its slot, a buffer's address."""
        if name.startswith(S.MEM_PREFIX):
            return self.mem[name] - self.region[_suffix(name)]
        return self.vb[name]

    def _place(self, programs: list[list[S.Step]], names: list[str]) -> None:
        """The vector buffer in banks: each buffer takes one, and a buffer's
        bank is the high bits of its address, so an adapter's byte address
        carries it and no port needs a bank field of its own.  The banks are
        the same size, since the stride has to be a shift, and the colouring
        balances them so that size is the largest bank and not the sum."""
        graph: dict[str, set[str]] = {}
        for prog in programs:                            # the programs never run together: their conflicts, each
            for name, edges in bank_conflicts(prog, self.chunk).items():
                graph.setdefault(name, set()).update(edges)
        self.bank = colour_banks(graph, lambda n: -(-self.size(n) // BEAT) * BEAT)
        for name in names:                               # a buffer no step's operands reach still needs a bank
            self.bank.setdefault(name, 0)
        self.banks = max(self.bank.values()) + 1 if self.bank else 1
        self.bank_reads, self.bank_writes = [0] * self.banks, [0] * self.banks
        for prog in programs:
            reads, writes = bank_ports(prog, self.bank, self.chunk)
            self.bank_reads = [max(a, b) for a, b in zip(self.bank_reads, reads)]
            self.bank_writes = [max(a, b) for a, b in zip(self.bank_writes, writes)]
        fill = [0] * self.banks
        for name in names:
            b = self.bank[name]
            self.vb[name] = b, fill[b]                   # resolved once the stride is known
            fill[b] += -(-self.size(name) // BEAT) * BEAT
        self.bank_shift = max(1, max(fill, default=1) - 1).bit_length()
        self.bank_bytes = 1 << self.bank_shift
        self.vb = {n: (b << self.bank_shift) + off for n, (b, off) in self.vb.items()}
        self.vb_bytes = self.banks << self.bank_shift

    def _place_lanes(self, lanes: list[list[S.Step]], names: list[str]) -> None:
        """The lanes' banks: one colouring of the plain names over every
        lane's program, and lane l on its own copy of the banks."""
        plain = lambda n: n.rsplit("@", 1)[0]
        lane_of = lambda n: int(n.rsplit("@", 1)[1])
        graph: dict[str, set[str]] = {}
        for prog in lanes:
            for name, edges in bank_conflicts(prog, self.chunk).items():
                graph.setdefault(plain(name), set()).update(plain(e) for e in edges if plain(e) != plain(name))
        at = colour_banks(graph, lambda n: -(-self.size(n) // BEAT) * BEAT)
        per = max(at.values()) + 1 if at else 1
        self.bank = {n: at.get(plain(n), 0) + per * lane_of(n) for n in names}
        self.banks = per * len(lanes)
        self.bank_reads, self.bank_writes = [0] * self.banks, [0] * self.banks
        for prog in lanes:
            reads, writes = bank_ports(prog, self.bank, self.chunk)
            self.bank_reads = [max(a, b) for a, b in zip(self.bank_reads, reads + [0] * (self.banks - len(reads)))]
            self.bank_writes = [max(a, b) for a, b in zip(self.bank_writes, writes + [0] * (self.banks - len(writes)))]
        fill = [0] * self.banks
        for name in names:
            b = self.bank[name]
            self.vb[name] = b, fill[b]
            fill[b] += -(-self.size(name) // BEAT) * BEAT
        self.bank_shift = max(1, max(fill, default=1) - 1).bit_length()
        self.bank_bytes = 1 << self.bank_shift
        self.vb = {n: (b << self.bank_shift) + off for n, (b, off) in self.vb.items()}
        self.vb_bytes = self.banks << self.bank_shift

    def size(self, name: str) -> int:
        return self.sizes[name.split("@")[0]]

    def sram(self) -> dict:
        """The memories the vector buffer is: per bank, the words its two
        halves hold and the ports they carry.  A bank is two memories because
        a sixteen-byte read at any byte address straddles two words, so the
        even words are in one and the odd in the other; the bits are counted
        once per read port, since that is how a read port is bought."""
        fill = [0] * self.banks
        for name in self.vb:
            fill[self.bank[name]] += -(-self.size(name) // BEAT) * BEAT
        banks = []
        for b in range(self.banks):
            words = -(-fill[b] // 16)                    # what the bank holds, not the stride it is addressed by
            banks.append({"bank": b, "half_words": -(-words // 2), "used_bytes": fill[b],
                          "reads": self.bank_reads[b], "writes": self.bank_writes[b],
                          "bits": words * 128 * self.bank_reads[b]})
        return {"banks": banks, "macros": 2 * self.banks, "word_bits": 128,
                "buffer_bytes": sum(fill), "address_bytes": self.vb_bytes,
                "sram_bits": sum(b["bits"] for b in banks)}

    def cap_mask(self, need: list[int], ports: int) -> int:
        """The banks that need at least ``ports`` of a kind, as a bit mask for
        the RTL's check."""
        assert self.banks <= 64, f"{self.banks} banks, the mask holds 64"
        assert max(need) <= 3, f"a bank wants {max(need)} ports of one kind"
        return sum(1 << b for b, n in enumerate(need) if n >= ports)

    def address(self, name: str) -> int:
        return self.mem[name] if name.startswith(S.MEM_PREFIX) else self.vb[name]


# --------------------------------------------------------------------------
# Byte packing
# --------------------------------------------------------------------------

def _suffix(name: str) -> str:
    return "" if "@" not in name else "@" + name.split("@")[1]


def _token_index(sfx: str) -> int:
    return int(sfx[1:]) if sfx else 0


def _int8(values) -> bytes:
    return np.asarray(values, dtype=np.int64).astype(np.int8).tobytes()


def _int16(values) -> bytes:
    return np.asarray(values, dtype=np.int64).astype("<i2").tobytes()


def _hist_bytes(hist: np.ndarray) -> bytes:
    """The conv history ``[channels, kernel - 1]`` at HIST_REC bytes per channel, oldest first."""
    rec = np.zeros((hist.shape[0], S.HIST_REC), dtype=np.int8)
    rec[:, :hist.shape[1]] = np.asarray(hist, dtype=np.int64).astype(np.int8)
    return rec.tobytes()


def _slot_bytes(rows: np.ndarray, scale) -> bytes:
    """A state slot: the header beat (g, e, peak, nsat) then the int8 rows."""
    g, e, peak, nsat = (int(v) for v in scale)
    header = bytes([g & 0xFF, (g >> 8) & 0xFF, e & 0xFF, peak & 0xFF, nsat & 0xFF, (nsat >> 8) & 0xFF]) + bytes(BEAT - 6)
    return header + _int8(rows)


def _unpack_slot(raw: bytes, k: int, v: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    g = raw[0] | (raw[1] << 8)
    e = raw[2] - 256 if raw[2] >= 128 else raw[2]
    nsat = raw[4] | (raw[5] << 8)
    rows = np.frombuffer(raw[BEAT:BEAT + k * v], dtype=np.int8).astype(np.int64).reshape(k, v)
    return rows, (g, e, raw[3], nsat)


def _write_bytes(path: Path, data: bytes) -> None:
    write_hex(path, data, 8)


def _write_beats(path: Path, data: bytes) -> None:
    assert len(data) % BEAT == 0
    write_hex(path, [int.from_bytes(data[i:i + BEAT], "little") for i in range(0, len(data), BEAT)], 8 * BEAT)


def read_hex_bytes(path: Path, width_bytes: int) -> bytes:
    """A ``$writememh`` image (one word per line, comments and addresses skipped) as bytes, little-endian words."""
    out = bytearray()
    for line in path.read_text().splitlines():
        line = line.split("//")[0].strip()
        if not line or line.startswith("@"):
            continue
        out += int(line.replace("x", "0").replace("z", "0"), 16).to_bytes(width_bytes, "little")   # an unwritten byte reads as zero
    return bytes(out)


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------

def _passes(c) -> list:
    """The four passes' matrices with the byte offset of each in its pass output, and whether raw accumulators are wanted."""
    ffn = c.ffn.gate_proj.out_features
    common = [[(c.ffn.gate_proj, 0, False), (c.ffn.up_proj, ffn, False)], [(c.ffn.down_proj, 0, False)]]
    if isinstance(c, L.RecurrentConsts):
        lay = _layout_of(c)
        return [[(c.in_proj_qkv, 0, False), (c.in_proj_z, lay["off_z"], False), (c.in_proj_b, lay["off_b"], True),
                 (c.in_proj_a, lay["off_a"], True)], [(c.out_proj, 0, False)]] + common
    lay = _layout_of(c)
    return [[(c.q_proj, 0, False), (c.k_proj, lay["off_k"], False), (c.v_proj, lay["off_v"], False),
             (c.index_q, lay["off_iq"], False), (c.index_k, lay["off_ik"], False)], [(c.o_proj, 0, False)]] + common


_LAYOUTS: dict[int, dict] = {}


def _layout_of(c) -> dict:
    return _LAYOUTS[id(c)]


def _write_rom(path: Path, words: np.ndarray, spec: TileSpec) -> None:
    """A tile's ROM image, one row per line, column 0 in the lowest nibble."""
    if spec.weight_bits == 4 and spec.cols % 2 == 0:
        packed = (words[:, 0::2] | (words[:, 1::2] << 4)).astype(np.uint8)        # byte c holds columns 2c and 2c + 1
        path.write_text("".join(packed[r, ::-1].tobytes().hex() + "\n" for r in range(words.shape[0])))
    else:
        write_hex(path, [sum(int(words[r, col]) << (col * spec.weight_bits) for col in range(spec.cols)) for r in range(words.shape[0])],
                  spec.cols * spec.weight_bits)


def _tiles(directory: Path, c, cfg, spec: TileSpec, layers: list | None = None) -> int:
    """The tile ROM images, the flat requantizer tables and the pass table;
    returns the tile count.  With ``layers`` (a die's layers' constants, in
    order) the array holds every layer's tiles, one after another, and layer
    L's passes are 4 L to 4 L + 3, which the engine's layer register picks."""
    mult, shift, table, ranges = [], [], [], []
    t = 0
    for layer, lc in enumerate(layers or [c]):
        first = t
        for p, matrices in enumerate(_passes(lc)):
            ranges.append(t)
            for q, base, raw in matrices:
                cm = compile_matrix(q, spec)
                start = t
                for tile in cm.tiles:
                    _write_rom(directory / f"tile_{t}.hex", tile.rom_words(spec), spec)
                    mult += list(tile.mult)
                    shift += list(tile.shift)
                    chain = 0xFFF if tile.row_block == 0 else t - cm.col_blocks
                    assert chain == 0xFFF or cm.tiles[chain - start].col_block == tile.col_block
                    last = tile.row_block == cm.row_blocks - 1
                    nbytes = tile.valid_cols * (4 if raw else 1)
                    dst_off = base + tile.col_block * spec.cols * (4 if raw else 1)
                    assert nbytes < 256 and dst_off < (1 << 24) and cm.row_blocks < 16
                    table.append((4 * layer + p) | (tile.row_block << 4) | (chain << 8) | (int(last) << 20) | (int(raw) << 21)
                                 | (nbytes << 22) | (dst_off << 30))
                    t += 1
            ranges[-1] |= (t - 1) << 12                 # the pass's first tile and its last
        # The timing model works the same array out from the config alone, and a
        # pass's cost is partly its tiles, so the two must agree.
        assert t - first == sum(-(-i // spec.rows) * -(-o // spec.cols)
                                for p in S.pass_matrices(cfg, isinstance(lc, L.RecurrentConsts)) for i, o, _ in p), t
    assert t < 0xFFF and len(ranges) <= 16
    write_hex(directory / "tiles_mult.hex", mult, spec.scale_bits)
    write_hex(directory / "tiles_shift.hex", shift, spec.shift_bits)
    write_hex(directory / "passes.hex", table, TAB_BITS)
    write_hex(directory / "pass_ranges.hex", ranges + [0] * (16 - len(ranges)), 24)
    return t


def _consts(directory: Path, c, cfg, sw: int) -> None:
    """The constant tables of every unit; a layer kind's absent units get zero tables of the right size."""
    def norm_entry(n: L.Norm | None, gmult: int = 0, gshift: int = 0) -> int:
        if n is None:
            return 0
        assert n.eps_int < (1 << sw)
        return int(n.mult) | (int(n.shift) << 16) | (gmult << 22) | (gshift << 38) | (int(n.eps_int) << 44)
    f = c.ffn
    recurrent = isinstance(c, L.RecurrentConsts)
    nv, hd, rd = cfg.linear_num_value_heads, cfg.head_dim, cfg.rotary_dim
    conv_dim = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + nv * cfg.linear_value_head_dim
    gated = c.gated_norm if recurrent else None
    write_hex(directory / "norm_consts.hex",
              [norm_entry(c.norm), norm_entry(c.unit_norm), norm_entry(gated, *((c.z_mult, c.z_shift) if recurrent else (0, 0))),
               norm_entry(f.norm)], 44 + sw)
    if recurrent:
        write_hex(directory / "conv_taps.hex", [sum((int(v) & 0xFF) << (8 * j) for j, v in enumerate(row)) for row in c.conv_w], 8 * cfg.linear_conv_kernel)
        write_hex(directory / "conv_consts.hex",
                  [int(mi) | (int(si) << 16) | (int(mo) << 22) | (int(so) << 38)
                   for mi, si, mo, so in zip(c.conv_mult_in, c.conv_sh_in, c.conv_mult_out, c.conv_sh_out)], 44)
        write_hex(directory / "gates_consts.hex",
                  [int(ma) | (int(sa) << 16) | (int(mb) << 22) | (int(sb) << 38) | (int(ac) << 44) | ((int(db) & 0xFFFF) << 60)
                   for ma, sa, mb, sb, ac, db in zip(c.gate_mult_a, c.gate_sh_a, c.gate_mult_b, c.gate_sh_b, c.a_coef, c.dt_bias)], 76)
        write_hex(directory / "rot_consts.hex", [0, 0], 44 + sw)
        write_hex(directory / "rot_gains.hex", [0] * (2 * hd), 16)
        write_hex(directory / "inv_freq.hex", [0] * (rd // 2), 32)
        write_hex(directory / "attn_consts.hex", [0], 66)
    else:
        write_hex(directory / "conv_taps.hex", [0] * conv_dim, 8 * cfg.linear_conv_kernel)
        write_hex(directory / "conv_consts.hex", [0] * conv_dim, 44)
        write_hex(directory / "gates_consts.hex", [0] * nv, 76)
        # The head norms keep their weight as a gain table; the rotary requantizer rides in the gain fields.
        write_hex(directory / "rot_consts.hex", [norm_entry(c.q_norm, c.rot_mult_q, c.rot_sh_q), norm_entry(c.k_norm, c.rot_mult_k, c.rot_sh_k)], 44 + sw)
        write_hex(directory / "rot_gains.hex", list(c.q_norm.gain) + list(c.k_norm.gain), 16)
        write_hex(directory / "inv_freq.hex", list(c.inv_freq), 32)
        write_hex(directory / "attn_consts.hex",
                  [c.mult_s | (c.sh_s << 16) | (c.mult_gate << 22) | (c.sh_gate << 38) | (c.mult_o << 44) | (c.sh_o << 60)], 66)
    write_hex(directory / "swiglu_consts.hex", [f.mult_g | (f.sh_g << 16) | (f.mult_o << 22) | (f.sh_o << 38)] + [0] * 15, 44)
    write_hex(directory / "residual_consts.hex", [c.res_mult | (c.res_shift << 16), f.res_mult | (f.res_shift << 16)] + [0] * 14, 22)


def _geometry(cfg, spec: TileSpec, mm: MemoryMap, layout: "Layout", programs: list[list[S.Step]], chunk: int, nt: int,
              model_tiles: bool = False) -> dict:
    """The engine's elaboration: its units' geometry, the global layer's
    memory map, the tile array and the vector buffer's banks and ports."""
    d, nk, nv = cfg.hidden_size, cfg.linear_num_key_heads, cfg.linear_num_value_heads
    hk, hv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    regions = mm.regions()
    return {"D": d, "NK": nk, "NV": nv, "HK": hk, "HV": hv, "KK": cfg.linear_conv_kernel,
            "CONV": 2 * nk * hk + nv * hv, "NH": cfg.num_attention_heads, "NKV": cfg.num_key_value_heads, "HD": cfg.head_dim,
            "RD": cfg.rotary_dim, "IDIM": cfg.index_dim, "W": mm.local_window, "BS": mm.block, "TOP": cfg.top_blocks,
            "KV_BITS": mm.kv_bits, "REC_BYTES": mm.kv_record_bytes, "RPB": mm.index_burst_records, "MAXR": mm.window_burst_records,
            "WINDOW_OFF": regions["window0"][0], "BLOCK_OFF": regions["blocks0"][0], "INDEX_OFF": regions["index0"][0],
            "SUMS_OFF": regions["sums0"][0], "ATT_L": S.Timing().head_lanes(cfg.head_dim), "SW_L": S.Timing().l_vec,
            "ROWS": spec.rows, "COLS": spec.cols, "P": spec.rows_per_cycle, "NT": nt, "TMAX": chunk,
            "MODEL_TILES": int(model_tiles), "AW": max(16, (max(layout.vb_bytes, 1) - 1).bit_length() + 1),
            "WB": spec.weight_bits, "ACC": spec.acc_bits, "SB": spec.scale_bits, "SHB": spec.shift_bits, "SW": L.sw_for(16, d),
            "YSH": L.ysh_for(hk), "VB_BYTES": layout.vb_bytes,
            "VB_BANKS": layout.banks, "VB_BANK_SHIFT": layout.bank_shift,
            "VB_RCAP2": layout.cap_mask(layout.bank_reads, 2),
            "VB_RCAP3": layout.cap_mask(layout.bank_reads, 3),
            "VB_WCAP2": layout.cap_mask(layout.bank_writes, 2),
            **port_params(programs, chunk)}


def port_params(programs: list[S.Step] | list[list[S.Step]], chunk: int = 1) -> dict:
    """The crossbar's port face: how many ports it needs and which one each of
    the engine's logical ports folds onto, four bits each."""
    out = {}
    for kind, write in (("R", False), ("W", True)):
        colours, count = port_colours(programs, write=write, chunk=chunk)
        out[f"VB_NP{kind}"] = count
        for half in (0, 1):
            word = 0
            for k, c in enumerate(colours[half*16:(half+1)*16]):
                word |= c << (4 * k)
            out[f"VB_{kind}MAP{half}"] = word
    return out


class EngineRun:
    """One run of a layer on the engine: the files in ``directory``, the
    testbench parameters and the expected results.  ``inputs`` are
    ``run_program``'s (a stream's carry the token suffix); for the global
    layer ``memory`` maps each token's ``m_ctx`` to its initial image and
    the image expected after the run (``GlobalContextMemory`` before and
    after the token's append).

    With ``ring`` the tokens come in as packets on the die's ring link, a
    lane each, and the engine is started by ``fabric_die_link`` rather than
    the testbench: the buffer starts with no input in it, and what leaves
    on the link is checked against the model's packets (``ring_out.hex``).

    Each token in flight's position is the engine's to know at run time,
    as its slot is: a global program's memory, rotary-table and attention
    commands carry the token's place in its chunk, and the engine adds the
    position it was started with.  The positions are the programs' own
    (their memory steps say which), and go in as ``POS<k>``, or on the
    packets through the ring.

    With ``lanes`` the steps are the lanes': each lane's runs (programs
    retargeted to it, ``sequencer.retarget(..., private=True)``), pushed
    lane by lane, the lanes' tokens in flight being their own numbers.  The
    engine runs them together and must take the model's cycles
    (``sequencer.schedule_lanes``)."""

    def __init__(self, directory: Path, cfg, c, spec: TileSpec, mm: MemoryMap, steps: list[S.Step], inputs: dict,
                 memory: dict[str, tuple[bytes, bytes]] | None = None, ndev: int = 0, model_tiles: bool = False,
                 first: bool = False, base_page: int = 0, ring: bool = False,
                 lanes: list[list[list[S.Step]]] | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        progs = [S.chain(runs) for runs in lanes] if lanes else None
        if progs:
            steps = [s for p in progs for s in p]
        self.cfg, self.steps, self.mm, self.ndev = cfg, steps, mm, ndev
        self.recurrent = isinstance(c, L.RecurrentConsts)
        self.nv, self.hk, self.hv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
        d, nk = cfg.hidden_size, cfg.linear_num_key_heads
        conv_dim, ffn = 2 * nk * self.hk + self.nv * self.hv, cfg.layer_intermediate_size(0)
        xs = [v for k, v in inputs.items() if k.split("@")[0] == "x"]
        self.chunk = xs[0].shape[0] if np.ndim(xs[0]) == 2 else 1          # tokens per pass: the residual's shape says
        lay = S.recurrent_layout(cfg, spec, mm, self.chunk) if self.recurrent else S.global_layout(cfg, spec, mm, self.chunk)
        _LAYOUTS[id(c)] = lay
        self.layout = Layout(steps, lay["sizes"], self.chunk, base_page, lanes=progs)
        self.suffixes = sorted({"" if "@" not in key else "@" + key.split("@")[1] for key in inputs})
        self.positions = {s.token or 0: s.ops["position"] for s in steps if s.ops and "position" in s.ops}
        # Images: the vector buffer holds each token's x, the memory its context.
        vb = bytearray(self.layout.vb_bytes)
        mem = bytearray(self.layout.mem_beats * BEAT)
        self.expected_memory: dict[str, bytes] = {}
        for sfx in self.suffixes:
            if not ring:                                             # through the ring, the packet brings it
                self._place(vb, "x" + sfx, _int16(inputs["x" + sfx]))
            if self.recurrent:
                assert mm.state_bits == 8, "the engine holds the int8 state"
                self._place(mem, "m_hist" + sfx, _hist_bytes(inputs["hist_mem" + sfx]))
                for h in range(self.nv):
                    self._place(mem, f"m_s[{h}]" + sfx, _slot_bytes(inputs["s_mem" + sfx][h], inputs["scale_mem" + sfx][h]))
            else:
                before, after = memory["m_ctx" + sfx]
                self._place(mem, "m_ctx" + sfx, before)
                self.expected_memory["m_ctx" + sfx] = after
        _write_bytes(directory / "vb_init.hex", bytes(vb))
        _write_beats(directory / "mem_init.hex", bytes(mem))
        # With ndev the memory is the HPI devices: the image striped over them, a page per device in turn.
        hpi_params = {"USE_HPI": 0}
        if ndev:
            self.device_bytes = max(2 * hpi.STRIPE_BYTES, 1 << (-(-len(mem) // (ndev * hpi.STRIPE_BYTES)) * hpi.STRIPE_BYTES - 1).bit_length())
            striped = hpi.StripedImage(ndev, self.device_bytes)
            striped.write(0, bytes(mem))
            striped.to_hex(directory / "devs.hex")
            mrs = hpi.mode_registers()
            hpi_params = {"USE_HPI": 1, "NDEV": ndev, "DEV_WORDS": self.device_bytes // 2, "MR0": mrs[0], "MR4": mrs[4], "MR8": mrs[8]}
        # The expected results: the same steps on the integer model.
        copies = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in inputs.items()}
        self.expected = S.run_program(steps, copies)
        # The program and everything the units load.
        L.write_luts(directory)
        # The lanes' stores and the runs pushed, lane by lane: {set, page, layer, lane, pc, steps}.
        if progs:
            pushes, start = [], []
            for l, runs in enumerate(lanes):
                write_hex(directory / S.lane_store_name(l), S.encode(progs[l], self.layout, limit=S.LANE_IDS), 256)
                start.append(len(pushes))
                pc = 0
                for run in runs:
                    pushes.append((1 << (57 + l)) | (l << 32) | (pc << 16) | len(run))
                    pc += len(run)
            self.schedule_cycles = S.schedule_lanes(progs, start).cycles
        else:
            write_hex(directory / "program.hex", S.encode(steps, self.layout, limit=S.LANE_IDS), 256)
            pushes = [(0xF << 57) | len(steps)]
            self.schedule_cycles = S.schedule(steps).cycles
        write_hex(directory / "runs.hex", pushes, 64)
        nt = _tiles(directory, c, cfg, spec)
        sw = L.sw_for(16, d)
        _consts(directory, c, cfg, sw)
        geometry = _geometry(cfg, spec, mm, self.layout, [steps], self.chunk, nt, model_tiles)
        if progs:                                                    # nothing folds: see fabric_vb
            geometry.update({"VB_FOLD": 0, "VB_NPR": _port_index(RD_PORT_MAP, self.chunk)[1],
                             "VB_NPW": _port_index(WR_PORT_MAP, self.chunk)[1]})
        self.params = {"N": len(steps), "RUNS": len(pushes), "LANES_USED": len(lanes) if lanes else 1, "FFN": ffn,
                       **geometry,
                       "MEM_BEATS": self.layout.mem_beats,
                       "SCHEDULE_CYCLES": self.schedule_cycles, **hpi_params,
                       # The tokens in flight: each one's slot, and FIRST for all of them or none.
                       **{f"SLOT{k}": page for k, page in self.layout.slot_pages().items()},
                       "FIRST": ((1 << len(self.layout.region)) - 1) if first else 0,
                       **{f"POS{k}": pos for k, pos in self.positions.items()}}
        self.progs, self.ring = progs, ring
        if ring:
            assert progs, "through the ring the tokens are lanes"
            self.params.update(self._ring(directory, inputs, first, base_page))
        (directory / "params.json").write_text(json.dumps(self.params))

    def expected_cycles(self, directory: Path) -> int:
        """The model's count for the run.  Through the ring the lanes start
        when the link pushes them, which the testbench records."""
        if not self.ring:
            return self.schedule_cycles
        return S.schedule_lanes(self.progs, ring_starts(directory, len(self.progs))).cycles

    def _ring(self, directory: Path, inputs: dict, first: bool, base_page: int) -> dict:
        """The packets in, a lane per token in flight, and the ones the model
        says leave; the die's program table; the link's parameters.  A lane's
        slot number is where its context's region is in the memory image: the
        regions are a slot each, the same size, from ``base_page`` on.  The
        packets take the lanes in order, all being free."""
        from fabric import controller as C
        starts = sorted(self.layout.region.values())
        sizes = {b - a for a, b in zip(starts, starts[1:] + [self.layout.mem_beats])}
        assert len(sizes) == 1 and min(sizes) % PAGE_BEATS == 0, "the lanes' regions differ in size"
        per_slot = min(sizes) // PAGE_BEATS
        lanes = sorted(self.suffixes, key=_token_index)
        assert [_token_index(sfx) for sfx in lanes] == list(range(len(lanes))), lanes
        page_base = base_page % per_slot
        slot = {sfx: (self.layout.region[sfx] // PAGE_BEATS - page_base) // per_slot for sfx in lanes}
        flags = C.FLAG_FIRST if first else 0
        words = lambda packet: [int.from_bytes(packet[i:i + 4], "little") for i in range(0, len(packet), 4)]
        ins, lengths, outs = [], [], []
        for k, sfx in enumerate(lanes):
            x = np.asarray(inputs["x" + sfx], dtype=np.int64).astype(np.int16)
            position = self.positions.get(k, 0)
            packet = C.pack_item(C.WorkItem(slot[sfx], position, x, flags))
            ins += words(packet)
            lengths.append(len(packet) // 4)
            y = np.asarray(self.expected["x2" + sfx], dtype=np.int64).reshape(x.shape).astype(np.int16)
            outs += words(C.pack_item(C.WorkItem(slot[sfx], position, y, flags)))
        addresses = 0
        for k, sfx in enumerate(lanes):
            addresses |= self.layout.vb["x" + sfx] << (32 + 24 * k)
            addresses |= self.layout.vb["x2" + sfx] << (128 + 24 * k)
        steps = {len(p) for p in self.progs}
        assert len(steps) == 1, "every lane's store holds the program at the same steps"
        table = [0] * 4                                              # {kind, chunked}: the program at step 0
        table[int(self.chunk > 1)] = addresses | (steps.pop() << 16)
        write_hex(directory / "die_table.hex", table, 256)
        write_hex(directory / "ring_in.hex", ins, 32)
        write_hex(directory / "ring_len.hex", lengths, 16)
        write_hex(directory / "ring_out.hex", outs, 32)
        return {"RING": 1, "RING_PACKETS": len(lanes), "RING_IN": len(ins), "RING_OUT": len(outs), "RING_LANES": len(lanes),
                "RING_CHUNK": self.chunk, "RING_SLOT_PAGES": per_slot, "RING_PAGE_BASE": page_base}

    def ring_problems(self, directory: Path) -> list[str]:
        """What left on the ring against the model's packets: all of them, each
        context's in order (``die.order_problems``)."""
        from fabric import die
        want = die.packets_of(int(w, 16) for w in (directory / "ring_out.hex").read_text().split())
        got = die.packets_of(int(w, 16) for w in (directory / "ring_got.hex").read_text().split())
        return die.order_problems(want, got)

    def _place(self, image: bytearray, name: str, data: bytes) -> None:
        base = self.layout.address(name) * (BEAT if name.startswith(S.MEM_PREFIX) else 1)
        assert len(data) <= self.layout.size(name), name
        image[base:base + len(data)] = data

    def check(self, directory: Path) -> list[str]:
        """Compare the testbench's final images with the expected results; a list of mismatches."""
        vb = read_hex_bytes(directory / "vb_out.hex", 1)
        if self.ndev:
            words = read_hex_bytes(directory / "devs_out.hex", 2)
            striped = hpi.StripedImage(self.ndev, self.device_bytes)
            for dev in range(self.ndev):
                striped.devices[dev][:] = words[dev * self.device_bytes:(dev + 1) * self.device_bytes]
            mem = striped.read(0, self.layout.mem_beats * BEAT)
        else:
            mem = read_hex_bytes(directory / "mem_out.hex", BEAT)
        problems = []

        def take(image: bytes, name: str, n: int) -> bytes:
            base = self.layout.address(name) * (BEAT if name.startswith(S.MEM_PREFIX) else 1)
            return image[base:base + n]

        for sfx in self.suffixes:
            x2 = np.frombuffer(take(vb, "x2" + sfx, 2 * self.cfg.hidden_size * self.chunk), dtype="<i2").astype(np.int64)
            want = np.asarray(self.expected["x2" + sfx]).reshape(-1)
            if not np.array_equal(x2, want):
                problems.append(f"x2{sfx}: {int((x2 != want).sum())} of {len(x2)} elements differ")
            if not self.recurrent:
                want = self.expected_memory["m_ctx" + sfx]
                got = take(mem, "m_ctx" + sfx, len(want))
                if got != want:
                    first = next(i for i in range(len(want)) if got[i] != want[i])
                    region = next((name for name, (off, size) in self.mm.regions().items() if name != "_total" and off <= first < off + size), "?")
                    problems.append(f"memory{sfx}: first difference at byte {first} ({region})")
                continue
            hist = np.frombuffer(take(mem, "m_hist" + sfx, self.layout.size("m_hist")), dtype=np.int8).astype(np.int64)
            hist = hist.reshape(-1, S.HIST_REC)[:, :self.cfg.linear_conv_kernel - 1]
            if not np.array_equal(hist, self.expected["hist_mem" + sfx]):
                problems.append(f"hist{sfx} differs")
            for h in range(self.nv):
                rows, scale = _unpack_slot(take(mem, f"m_s[{h}]" + sfx, self.layout.size(f"m_s[{h}]")), self.hk, self.hv)
                if not np.array_equal(rows, self.expected["s_mem" + sfx][h]):
                    problems.append(f"state{sfx} head {h}: {int((rows != self.expected['s_mem' + sfx][h]).sum())} elements differ")
                if tuple(scale) != tuple(int(v) for v in self.expected["scale_mem" + sfx][h]):
                    problems.append(f"scale{sfx} head {h}: {scale} expected {tuple(self.expected['scale_mem' + sfx][h])}")
        problems += self.check_banks(directory)
        if self.ring:
            problems += self.ring_problems(directory)
        return problems

    def check_banks(self, directory: Path) -> list[str]:
        """What each bank was really asked for against what the colouring
        gave it.  Over is a broken run and the RTL says so itself; under is
        only the analysis being careful, which it is entitled to be."""
        path = directory / "ports.txt"
        if not path.exists():
            return []
        problems = []
        for line in path.read_text().splitlines():
            f = line.split()
            if f[0] != "bank":
                continue
            b, reads, writes = (int(v) for v in f[1:4])
            # A bank has a port of each kind whatever the program asks of it:
            # the ring link loads the input and takes the output through them.
            gave_r, gave_w = max(1, self.layout.bank_reads[b]), max(1, self.layout.bank_writes[b])
            if reads > gave_r:
                problems.append(f"bank {b}: {reads} reads at once, the colouring gave it {gave_r}")
            if writes > gave_w:
                problems.append(f"bank {b}: {writes} writes at once, the colouring gave it {gave_w}")
        return problems


class _LaneOperands:
    """A lane's operands: its memory names' offsets in their part of the
    slot, from the program's own layout, and its buffers' addresses, from the
    lanes'."""

    def __init__(self, mem: Layout, vb: Layout) -> None:
        self.mem, self.vb = mem, vb

    def operand(self, name: str) -> int:
        if name.startswith(S.MEM_PREFIX):
            return self.mem.operand(name.rsplit("@", 1)[0])
        return self.vb.vb[name]

    address = operand


def ring_starts(directory: Path, lanes: int) -> list[int]:
    """Each lane's first push, from the first lane's, as the testbench
    recorded them (``pushes.txt``): when the model's lanes may start."""
    first: dict[int, int] = {}
    for line in (directory / "pushes.txt").read_text().splitlines():
        cycle, lane = (int(v) for v in line.split())
        first.setdefault(lane, cycle)
    return [first[l] - first[0] for l in range(lanes)]


class DieRun:
    """A layer die on the engine: its three recurrent layers and its global
    layer, one token after another of one context through the ring.

    The engine holds all four layers' weights and constants, banked by
    layer; it runs two programs, the recurrent one three times and the
    global one once, each writing its output over its input
    (``sequencer.in_place``), so a layer's output is the next layer's input
    where it lies.  The vector buffer is placed over both programs at once,
    so a buffer is at one address in each and one elaboration serves both.
    A slot is the four layers' state one after another -- three recurrent
    regions and the global context -- and the die link starts each layer
    with its part of the slot, its program and its layer.

    ``consts`` are the four layers' compiled constants, in order, all at one
    residual scale (the packet's).  ``tokens`` are ``(position, x, first)``
    for one context in slot ``slot``: they go in as packets back to back,
    and each runs the four layers before the next, since two tokens of a
    context cannot run at once -- the second waits on the link, then takes
    the next lane.  Every lane's store holds the two programs at the same
    steps, each lane's copy on its own buffers.  The expected results are the
    chained integer model's: the packets out and every layer's state after
    the last token."""

    def __init__(self, directory: Path, cfg, consts: list, spec: TileSpec, mm_r: MemoryMap, mm_g: MemoryMap,
                 tokens: list[tuple[int, np.ndarray, bool]], slot: int = 3, page_base: int = 5, seed: int = 11,
                 lanes: int = 2) -> None:
        from fabric import controller as C
        from fabric.memory import GlobalContextMemory
        directory.mkdir(parents=True, exist_ok=True)
        assert len(consts) == 4 and all(isinstance(c, L.RecurrentConsts) for c in consts[:3])
        assert isinstance(consts[3], L.GlobalConsts) and len({c.s_h for c in consts}) == 1, "one residual scale"
        self.cfg, self.consts, self.mm_r, self.mm_g = cfg, consts, mm_r, mm_g
        nv, hk, hv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
        lay_r, lay_g = S.recurrent_layout(cfg, spec, mm_r), S.global_layout(cfg, spec, mm_g)
        for c in consts[:3]:
            _LAYOUTS[id(c)] = lay_r
        _LAYOUTS[id(consts[3])] = lay_g
        # The two programs, in place; a layer's part of the slot laid out by
        # its program, and the lanes' buffers over every lane's four layers.
        self.rec = S.in_place(S.recurrent_program(cfg, consts[0], spec, mm_r, slots=S.LANE_SLOTS))
        self.glob = S.in_place(S.global_program(cfg, consts[3], spec, mm_g, tokens[0][0]))
        sizes = dict(lay_r["sizes"])
        for name, n in lay_g["sizes"].items():
            sizes[name] = max(n, sizes.get(name, 0))
        self.lr, self.lg = Layout(self.rec, sizes), Layout(self.glob, sizes)
        self.lanes = lanes
        lane_runs = [[S.retarget(prog, l, private=True) for prog in (self.rec, self.glob)] for l in range(lanes)]
        self.lv = Layout([st for runs in lane_runs for run in runs for st in run], sizes,
                         lanes=[S.chain([r, r, r, g]) for r, g in lane_runs])
        # A slot: three recurrent regions, then the global one.
        pages_r, pages_g = self.lr.mem_beats // PAGE_BEATS, self.lg.mem_beats // PAGE_BEATS
        self.layer_page = [0, pages_r, 2 * pages_r, 3 * pages_r]
        self.slot_pages = 3 * pages_r + pages_g
        self.slot, self.page_base = slot, page_base
        start = (page_base + slot * self.slot_pages) * PAGE_BEATS * BEAT
        mem = bytearray(start + self.slot_pages * PAGE_BEATS * BEAT)
        rng = np.random.default_rng(seed)
        self.start = start

        def region(layer: int) -> int:
            return start + self.layer_page[layer] * PAGE_BEATS * BEAT

        # The slot another context left: the recurrent state and history are
        # garbage, which FIRST must not read; the global region is garbage
        # where FIRST must not read it (the block sums, the window beyond the
        # tokens) and zero where the model starts from zero.
        first_pos = tokens[0][0]
        assert tokens[0][2] and first_pos == 0, "the slot starts fresh"
        for layer in range(3):
            n = pages_r * PAGE_BEATS * BEAT
            mem[region(layer):region(layer) + n] = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        store = GlobalContextMemory(mm_g, cfg.top_blocks)
        regions = mm_g.regions()
        g0 = region(3) + self.lg.operand("m_ctx") * BEAT
        before = bytearray(store.image.data)
        off, size = regions["sums0"]
        before[off:off + size] = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
        woff, _ = regions["window0"]
        rec = mm_g.kv_record_bytes
        junk = {}
        for n in range(cfg.num_key_value_heads):
            for p in range(len(tokens), mm_g.local_window):
                a = woff + (n * mm_g.local_window + p) * rec
                junk[a] = rng.integers(0, 256, rec, dtype=np.uint8).tobytes()
                before[a:a + rec] = junk[a]
        mem[g0:g0 + len(before)] = before

        # The chain on the integer model: a token through the four layers.
        s = [np.zeros((nv, hk, hv), dtype=np.int64) for _ in range(3)]
        sc = [np.tile([L.ONE_U, 0, 0, 0], (nv, 1)).astype(np.int64) for _ in range(3)]
        conv_dim = 2 * cfg.linear_num_key_heads * hk + nv * hv
        hist = [np.zeros((conv_dim, cfg.linear_conv_kernel - 1), dtype=np.int64) for _ in range(3)]
        zero = np.zeros((cfg.num_key_value_heads, 1, cfg.head_dim), dtype=np.int64)
        self.outputs, packets_in, packets_out = [], [], []
        for pos, x, first in tokens:
            h = np.asarray(x, dtype=np.int64)
            for layer in range(3):
                prog = S.recurrent_program(cfg, consts[layer], spec, mm_r, first=first)
                env = S.run_program(prog, {"x": h, "s_mem": s[layer].copy(), "scale_mem": sc[layer].copy(), "hist_mem": hist[layer].copy()})
                h, s[layer], sc[layer], hist[layer] = env["x2"], env["s_mem"], env["scale_mem"], env["hist_mem"]
            own = L.global_layer_int(consts[3], cfg, spec, h, pos, zero, zero)
            store.append(pos, own["k"], own["v"], own["index_k"])
            got = store.retrieve(pos, own["index_q_unit"])
            env = S.run_program(S.global_program(cfg, consts[3], spec, mm_g, pos, first=first),
                                {"x": h, "k_rows": got["k_rows"], "v_rows": got["v_rows"]})
            h = np.asarray(env["x2"], dtype=np.int64)
            self.outputs.append(h)
            flags = C.FLAG_FIRST if first else 0
            packets_in.append(C.pack_item(C.WorkItem(slot, pos, np.asarray(x).astype(np.int16), flags)))
            packets_out.append(C.pack_item(C.WorkItem(slot, pos, h.astype(np.int16), flags)))
        after = bytearray(store.image.data)
        for a, data in junk.items():
            after[a:a + rec] = data
        self.expected = {"s": s, "scale": sc, "hist": hist, "ctx": bytes(after)}
        self.region = region

        # The images and the tables.
        _write_bytes(directory / "vb_init.hex", bytes(self.lv.vb_bytes))
        _write_beats(directory / "mem_init.hex", bytes(mem))
        L.write_luts(directory)
        # A lane's store: the recurrent program, then the global one, on one
        # numbering of the buffers -- the link pushes all four layers at once
        # and the lane runs them back to back.
        for l, (r, g) in enumerate(lane_runs):
            ids = S.buffer_ids(r + g)
            words = (S.encode(r, _LaneOperands(self.lr, self.lv), ids, S.LANE_IDS)
                     + S.encode(g, _LaneOperands(self.lg, self.lv), ids, S.LANE_IDS))
            write_hex(directory / S.lane_store_name(l), words, 256)
        nt = _tiles(directory, consts[0], cfg, spec, layers=consts)
        sw = L.sw_for(16, cfg.hidden_size)
        tables: dict[str, list[str]] = {}
        for layer, c in enumerate(consts):                 # each unit's constants, a layer's bank after another's
            sub = directory / f"layer{layer}"
            sub.mkdir(exist_ok=True)
            _consts(sub, c, cfg, sw)
            for f in sub.iterdir():
                tables.setdefault(f.name, []).append(f.read_text())
        for name, parts in tables.items():
            (directory / name).write_text("".join(parts))
        words = lambda packet: [int.from_bytes(packet[i:i + 4], "little") for i in range(0, len(packet), 4)]
        ins = [w for p in packets_in for w in words(p)]
        outs = [w for p in packets_out for w in words(p)]
        write_hex(directory / "ring_in.hex", ins, 32)
        write_hex(directory / "ring_len.hex", [len(p) // 4 for p in packets_in], 16)
        write_hex(directory / "ring_out.hex", outs, 32)
        addresses = 0                                              # each lane's input and output: one buffer, in place
        for l in range(lanes):
            addresses |= self.lv.vb[f"x@{l}"] << (32 + 24 * l) | self.lv.vb[f"x@{l}"] << (128 + 24 * l)
        table = [0] * 4                                            # {kind, chunked}
        table[0] = addresses | (len(self.rec) << 16)               # the recurrent layers: step 0
        table[2] = addresses | len(self.rec) | (len(self.glob) << 16)   # the global one, after it
        write_hex(directory / "die_table.hex", table, 256)
        write_hex(directory / "die_layers.hex", [int(layer == 3) | (self.layer_page[layer] << 1) for layer in range(4)], 32)
        self.params = {"N": len(tokens) * (3 * len(self.rec) + len(self.glob)), "FFN": cfg.layer_intermediate_size(0),
                       **_geometry(cfg, spec, mm_g, self.lv, [self.rec, self.glob], 1, nt),
                       "VB_FOLD": 0, "VB_NPR": _port_index(RD_PORT_MAP)[1], "VB_NPW": _port_index(WR_PORT_MAP)[1],
                       "MEM_BEATS": len(mem) // BEAT, "SCHEDULE_CYCLES": 0, "USE_HPI": 0, "LAYERS": 4,
                       "RING": 1, "RING_PACKETS": len(tokens), "RING_IN": len(ins), "RING_OUT": len(outs), "RING_LANES": lanes,
                       "RING_CHUNK": 1, "RING_SLOT_PAGES": self.slot_pages, "RING_PAGE_BASE": page_base, "RING_LAYERS": 4}
        (directory / "params.json").write_text(json.dumps(self.params))

    def check(self, directory: Path) -> list[str]:
        """Every layer's state after the last token against the chained model's."""
        mem = read_hex_bytes(directory / "mem_out.hex", BEAT)
        problems = []
        nv, hk, hv = self.cfg.linear_num_value_heads, self.cfg.linear_key_head_dim, self.cfg.linear_value_head_dim
        for layer in range(3):
            base = self.region(layer)
            take = lambda name: mem[base + self.lr.operand(name) * BEAT:base + self.lr.operand(name) * BEAT + self.lr.size(name)]
            hist = np.frombuffer(take("m_hist"), dtype=np.int8).astype(np.int64)
            hist = hist.reshape(-1, S.HIST_REC)[:, :self.cfg.linear_conv_kernel - 1]
            if not np.array_equal(hist, self.expected["hist"][layer]):
                problems.append(f"layer {layer}: the conv history differs")
            for h in range(nv):
                rows, scale = _unpack_slot(take(f"m_s[{h}]"), hk, hv)
                if not np.array_equal(rows, self.expected["s"][layer][h]):
                    problems.append(f"layer {layer} head {h}: {int((rows != self.expected['s'][layer][h]).sum())} state elements differ")
                if tuple(scale) != tuple(int(v) for v in self.expected["scale"][layer][h]):
                    problems.append(f"layer {layer} head {h}: scale {scale}")
        g0 = self.region(3) + self.lg.operand("m_ctx") * BEAT
        want = self.expected["ctx"]
        got = mem[g0:g0 + len(want)]
        if got != want:
            first = next(i for i in range(len(want)) if got[i] != want[i])
            problems.append(f"layer 3: the context differs first at byte {first}")
        from fabric import die
        want_out = die.packets_of(int(w, 16) for w in (directory / "ring_out.hex").read_text().split())
        got_out = die.packets_of(int(w, 16) for w in (directory / "ring_got.hex").read_text().split())
        return problems + die.order_problems(want_out, got_out)
