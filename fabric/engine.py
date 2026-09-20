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
    sizes are looked up by the plain name."""

    def __init__(self, steps: list[S.Step], sizes: dict[str, int], chunk: int = 1) -> None:
        self.sizes, self.chunk = sizes, chunk
        self.vb: dict[str, int] = {}
        self.mem: dict[str, int] = {}
        names: list[str] = []
        mem_next = 0
        for step in steps:
            for value in (step.ops or {}).values():
                for name in _refs(value):
                    if name.startswith(S.MEM_PREFIX):
                        if name not in self.mem:
                            beats = -(-self.size(name) // BEAT)
                            self.mem[name] = mem_next
                            mem_next += -(-beats // PAGE_BEATS) * PAGE_BEATS
                    elif name not in names:
                        names.append(name)
        self.mem_beats = mem_next
        self._place(steps, names)

    def _place(self, steps: list[S.Step], names: list[str]) -> None:
        """The vector buffer in banks: each buffer takes one, and a buffer's
        bank is the high bits of its address, so an adapter's byte address
        carries it and no port needs a bank field of its own.  The banks are
        the same size, since the stride has to be a shift, and the colouring
        balances them so that size is the largest bank and not the sum."""
        self.bank = colour_banks(bank_conflicts(steps, self.chunk), lambda n: -(-self.size(n) // BEAT) * BEAT)
        for name in names:                               # a buffer no step's operands reach still needs a bank
            self.bank.setdefault(name, 0)
        self.banks = max(self.bank.values()) + 1 if self.bank else 1
        self.bank_reads, self.bank_writes = bank_ports(steps, self.bank, self.chunk)
        fill = [0] * self.banks
        for name in names:
            b = self.bank[name]
            self.vb[name] = b, fill[b]                   # resolved once the stride is known
            fill[b] += -(-self.size(name) // BEAT) * BEAT
        self.bank_shift = max(1, max(fill, default=1) - 1).bit_length()
        self.bank_bytes = 1 << self.bank_shift
        self.vb = {n: (b << self.bank_shift) + off for n, (b, off) in self.vb.items()}
        self.vb_bytes = self.banks << self.bank_shift

    def size(self, name: str) -> int:
        return self.sizes[name.split("@")[0]]

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


def _tiles(directory: Path, c, spec: TileSpec) -> int:
    """The tile ROM images, the flat requantizer tables and the pass table; returns the tile count."""
    mult, shift, table = [], [], []
    t = 0
    for p, matrices in enumerate(_passes(c)):
        for q, base, raw in matrices:
            cm = compile_matrix(q, spec)
            first = t
            for tile in cm.tiles:
                _write_rom(directory / f"tile_{t}.hex", tile.rom_words(spec), spec)
                mult += list(tile.mult)
                shift += list(tile.shift)
                chain = 0xFFF if tile.row_block == 0 else t - cm.col_blocks
                assert chain == 0xFFF or cm.tiles[chain - first].col_block == tile.col_block
                last = tile.row_block == cm.row_blocks - 1
                nbytes = tile.valid_cols * (4 if raw else 1)
                dst_off = base + tile.col_block * spec.cols * (4 if raw else 1)
                assert nbytes < 256 and dst_off < (1 << 24) and cm.row_blocks < 16
                table.append(p | (tile.row_block << 4) | (chain << 8) | (int(last) << 20) | (int(raw) << 21) | (nbytes << 22) | (dst_off << 30))
                t += 1
    assert t < 0xFFF
    write_hex(directory / "tiles_mult.hex", mult, spec.scale_bits)
    write_hex(directory / "tiles_shift.hex", shift, spec.shift_bits)
    write_hex(directory / "passes.hex", table, TAB_BITS)
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


class EngineRun:
    """One run of a layer on the engine: the files in ``directory``, the
    testbench parameters and the expected results.  ``inputs`` are
    ``run_program``'s (a stream's carry the token suffix); for the global
    layer ``memory`` maps each token's ``m_ctx`` to its initial image and
    the image expected after the run (``GlobalContextMemory`` before and
    after the token's append)."""

    def __init__(self, directory: Path, cfg, c, spec: TileSpec, mm: MemoryMap, steps: list[S.Step], inputs: dict,
                 memory: dict[str, tuple[bytes, bytes]] | None = None, ndev: int = 0, model_tiles: bool = False) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.cfg, self.steps, self.mm, self.ndev = cfg, steps, mm, ndev
        self.recurrent = isinstance(c, L.RecurrentConsts)
        self.nv, self.hk, self.hv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
        d, nk = cfg.hidden_size, cfg.linear_num_key_heads
        conv_dim, ffn = 2 * nk * self.hk + self.nv * self.hv, cfg.layer_intermediate_size(0)
        xs = [v for k, v in inputs.items() if k.split("@")[0] == "x"]
        self.chunk = xs[0].shape[0] if np.ndim(xs[0]) == 2 else 1          # tokens per pass: the residual's shape says
        lay = S.recurrent_layout(cfg, spec, mm, self.chunk) if self.recurrent else S.global_layout(cfg, spec, mm, self.chunk)
        _LAYOUTS[id(c)] = lay
        self.layout = Layout(steps, lay["sizes"], self.chunk)
        self.suffixes = sorted({"" if "@" not in key else "@" + key.split("@")[1] for key in inputs})
        # Images: the vector buffer holds each token's x, the memory its context.
        vb = bytearray(self.layout.vb_bytes)
        mem = bytearray(self.layout.mem_beats * BEAT)
        self.expected_memory: dict[str, bytes] = {}
        for sfx in self.suffixes:
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
        write_hex(directory / "program.hex", S.encode(steps, self.layout), 256)
        nt = _tiles(directory, c, spec)
        sw = L.sw_for(16, d)
        _consts(directory, c, cfg, sw)
        regions = mm.regions()
        self.params = {"N": len(steps), "D": d, "NK": nk, "NV": self.nv, "HK": self.hk, "HV": self.hv, "KK": cfg.linear_conv_kernel,
                       "CONV": conv_dim, "FFN": ffn, "NH": cfg.num_attention_heads, "NKV": cfg.num_key_value_heads, "HD": cfg.head_dim,
                       "RD": cfg.rotary_dim, "IDIM": cfg.index_dim, "W": mm.local_window, "BS": mm.block, "TOP": cfg.top_blocks,
                       "KV_BITS": mm.kv_bits, "REC_BYTES": mm.kv_record_bytes, "RPB": mm.index_burst_records, "MAXR": mm.window_burst_records,
                       "WINDOW_OFF": regions["window0"][0], "BLOCK_OFF": regions["blocks0"][0], "INDEX_OFF": regions["index0"][0],
                       "SUMS_OFF": regions["sums0"][0], "ATT_L": 8,
                       "ROWS": spec.rows, "COLS": spec.cols, "P": spec.rows_per_cycle, "NT": nt, "TMAX": self.chunk,
                       "MODEL_TILES": int(model_tiles), "AW": max(16, (max(self.layout.vb_bytes, 1) - 1).bit_length() + 1),
                       "WB": spec.weight_bits, "ACC": spec.acc_bits, "SB": spec.scale_bits, "SHB": spec.shift_bits, "SW": sw,
                       "YSH": L.ysh_for(self.hk), "VB_BYTES": self.layout.vb_bytes, "MEM_BEATS": self.layout.mem_beats,
                       "VB_BANKS": self.layout.banks, "VB_BANK_SHIFT": self.layout.bank_shift,
                       "VB_RCAP2": self.layout.cap_mask(self.layout.bank_reads, 2),
                       "VB_RCAP3": self.layout.cap_mask(self.layout.bank_reads, 3),
                       "VB_WCAP2": self.layout.cap_mask(self.layout.bank_writes, 2),
                       "SCHEDULE_CYCLES": S.schedule(steps).cycles, **hpi_params}
        (directory / "params.json").write_text(json.dumps(self.params))

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
            if reads > self.layout.bank_reads[b]:
                problems.append(f"bank {b}: {reads} reads at once, the colouring gave it {self.layout.bank_reads[b]}")
            if writes > self.layout.bank_writes[b]:
                problems.append(f"bank {b}: {writes} writes at once, the colouring gave it {self.layout.bank_writes[b]}")
        return problems
