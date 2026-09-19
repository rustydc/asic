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

The memory behind the port is a behavioural beat memory in the testbench;
the HPI bridge and its DMAs are checked on their own (``tb_mem_bridge``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fabric import layer as L
from fabric import sequencer as S
from fabric.memory import BEAT, MemoryMap
from fabric.tile import TileSpec, compile_matrix, write_hex

NPASS = 4                    # the recurrent layer's passes: in_proj, out_proj, gate_up, down
TAB_BITS = 48                # a pass-table entry


def _refs(value) -> list[str]:
    if isinstance(value, tuple):
        return [value[0]]
    if isinstance(value, list):
        return [n for _, v in value for n in _refs(v)]
    return []


class Layout:
    """Addresses of a program's buffers: vector-buffer names to byte
    offsets and memory-image names (``m_...``) to beat addresses, in order
    of first reference, each buffer beat-aligned.  A stream's names carry
    their token suffix; sizes are looked up by the plain name."""

    def __init__(self, steps: list[S.Step], sizes: dict[str, int]) -> None:
        self.sizes = sizes
        self.vb: dict[str, int] = {}
        self.mem: dict[str, int] = {}
        vb_next = mem_next = 0
        for step in steps:
            for value in (step.ops or {}).values():
                for name in _refs(value):
                    beats = -(-self.size(name) // BEAT)
                    if name.startswith(S.MEM_PREFIX):
                        if name not in self.mem:
                            self.mem[name] = mem_next
                            mem_next += beats
                    elif name not in self.vb:
                        self.vb[name] = vb_next
                        vb_next += beats * BEAT
        self.vb_bytes, self.mem_beats = vb_next, mem_next

    def size(self, name: str) -> int:
        return self.sizes[name.split("@")[0]]

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
        out += int(line, 16).to_bytes(width_bytes, "little")
    return bytes(out)


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------

def _tiles(directory: Path, c: L.RecurrentConsts, spec: TileSpec, lay: dict) -> int:
    """The tile ROM images, the flat requantizer tables and the pass table; returns the tile count."""
    ffn = c.ffn.gate_proj.out_features
    passes = [[(c.in_proj_qkv, 0, False), (c.in_proj_z, lay["off_z"], False), (c.in_proj_b, lay["off_b"], True),
               (c.in_proj_a, lay["off_a"], True)],
              [(c.out_proj, 0, False)],
              [(c.ffn.gate_proj, 0, False), (c.ffn.up_proj, ffn, False)],
              [(c.ffn.down_proj, 0, False)]]
    mult, shift, table = [], [], []
    t = 0
    for p, matrices in enumerate(passes):
        for q, base, raw in matrices:
            cm = compile_matrix(q, spec)
            first = t
            for tile in cm.tiles:
                words = tile.rom_words(spec)
                rows = [sum(int(words[r, col]) << (col * spec.weight_bits) for col in range(spec.cols)) for r in range(spec.rows)]
                write_hex(directory / f"tile_{t}.hex", rows, spec.cols * spec.weight_bits)
                mult += list(tile.mult)
                shift += list(tile.shift)
                chain = 0xFF if tile.row_block == 0 else t - cm.col_blocks
                assert chain == 0xFF or cm.tiles[chain - first].col_block == tile.col_block
                last = tile.row_block == cm.row_blocks - 1
                nbytes = tile.valid_cols * (4 if raw else 1)
                dst_off = base + tile.col_block * spec.cols * (4 if raw else 1)
                assert nbytes < 256 and dst_off < (1 << 16) and cm.row_blocks < 16
                table.append(p | (tile.row_block << 4) | (chain << 8) | (int(last) << 16) | (int(raw) << 17) | (nbytes << 18) | (dst_off << 26))
                t += 1
    assert t < 255
    write_hex(directory / "tiles_mult.hex", mult, spec.scale_bits)
    write_hex(directory / "tiles_shift.hex", shift, spec.shift_bits)
    write_hex(directory / "passes.hex", table, TAB_BITS)
    return t


def _consts(directory: Path, c: L.RecurrentConsts, sw: int) -> None:
    def norm_entry(n: L.Norm, gmult: int = 0, gshift: int = 0) -> int:
        assert n.gain is None or (np.ndim(n.gain) == 0 and int(n.gain) == 1)
        assert n.eps_int < (1 << sw)
        return int(n.mult) | (int(n.shift) << 16) | (gmult << 22) | (gshift << 38) | (int(n.eps_int) << 44)
    write_hex(directory / "norm_consts.hex",
              [norm_entry(c.norm), norm_entry(c.unit_norm), norm_entry(c.gated_norm, c.z_mult, c.z_shift), norm_entry(c.ffn.norm)], 44 + sw)
    kernel = c.conv_w.shape[1]
    write_hex(directory / "conv_taps.hex", [sum((int(v) & 0xFF) << (8 * j) for j, v in enumerate(row)) for row in c.conv_w], 8 * kernel)
    write_hex(directory / "conv_consts.hex",
              [int(mi) | (int(si) << 16) | (int(mo) << 22) | (int(so) << 38)
               for mi, si, mo, so in zip(c.conv_mult_in, c.conv_sh_in, c.conv_mult_out, c.conv_sh_out)], 44)
    write_hex(directory / "gates_consts.hex",
              [int(ma) | (int(sa) << 16) | (int(mb) << 22) | (int(sb) << 38) | (int(ac) << 44) | ((int(db) & 0xFFFF) << 60)
               for ma, sa, mb, sb, ac, db in zip(c.gate_mult_a, c.gate_sh_a, c.gate_mult_b, c.gate_sh_b, c.a_coef, c.dt_bias)], 76)
    f = c.ffn
    write_hex(directory / "swiglu_consts.hex", [f.mult_g | (f.sh_g << 16) | (f.mult_o << 22) | (f.sh_o << 38)] + [0] * 15, 44)
    write_hex(directory / "residual_consts.hex", [c.res_mult | (c.res_shift << 16), f.res_mult | (f.res_shift << 16)] + [0] * 14, 22)


class EngineRun:
    """One run of the recurrent layer on the engine: the files in
    ``directory``, the testbench parameters and the expected results."""

    def __init__(self, directory: Path, cfg, c: L.RecurrentConsts, spec: TileSpec, mm: MemoryMap, steps: list[S.Step],
                 inputs: dict) -> None:
        assert mm.state_bits == 8, "the engine holds the int8 state"
        directory.mkdir(parents=True, exist_ok=True)
        self.cfg, self.steps = cfg, steps
        self.nv, self.hk, self.hv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
        d, nk = cfg.hidden_size, cfg.linear_num_key_heads
        conv_dim, ffn = 2 * nk * self.hk + self.nv * self.hv, cfg.layer_intermediate_size(0)
        lay = S.recurrent_layout(cfg, spec, mm)
        self.layout = Layout(steps, lay["sizes"])
        self.suffixes = sorted({"" if "@" not in key else "@" + key.split("@")[1] for key in inputs})
        # Images: the vector buffer holds each token's x, the memory its history and state.
        vb = bytearray(self.layout.vb_bytes)
        mem = bytearray(self.layout.mem_beats * BEAT)
        for sfx in self.suffixes:
            self._place(vb, "x" + sfx, _int16(inputs["x" + sfx]))
            self._place(mem, "m_hist" + sfx, _hist_bytes(inputs["hist_mem" + sfx]))
            for h in range(self.nv):
                self._place(mem, f"m_s[{h}]" + sfx, _slot_bytes(inputs["s_mem" + sfx][h], inputs["scale_mem" + sfx][h]))
        _write_bytes(directory / "vb_init.hex", bytes(vb))
        _write_beats(directory / "mem_init.hex", bytes(mem))
        # The expected results: the same steps on the integer model.
        copies = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in inputs.items()}
        self.expected = S.run_program(steps, copies)
        # The program and everything the units load.
        L.write_luts(directory)
        write_hex(directory / "program.hex", S.encode(steps, self.layout), 256)
        nt = _tiles(directory, c, spec, lay)
        sw = L.sw_for(16, d)
        _consts(directory, c, sw)
        self.params = {"N": len(steps), "D": d, "NK": nk, "NV": self.nv, "HK": self.hk, "HV": self.hv, "KK": cfg.linear_conv_kernel,
                       "CONV": conv_dim, "FFN": ffn, "ROWS": spec.rows, "COLS": spec.cols, "P": spec.rows_per_cycle, "NT": nt,
                       "WB": spec.weight_bits, "ACC": spec.acc_bits, "SB": spec.scale_bits, "SHB": spec.shift_bits, "SW": sw,
                       "YSH": L.ysh_for(self.hk), "VB_BYTES": self.layout.vb_bytes, "MEM_BEATS": self.layout.mem_beats,
                       "SCHEDULE_CYCLES": S.schedule(steps).cycles}
        (directory / "params.json").write_text(json.dumps(self.params))

    def _place(self, image: bytearray, name: str, data: bytes) -> None:
        base = self.layout.address(name) * (BEAT if name.startswith(S.MEM_PREFIX) else 1)
        assert len(data) <= self.layout.size(name), name
        image[base:base + len(data)] = data

    def check(self, directory: Path) -> list[str]:
        """Compare the testbench's final images with the expected results; a list of mismatches."""
        vb = read_hex_bytes(directory / "vb_out.hex", 1)
        mem = read_hex_bytes(directory / "mem_out.hex", BEAT)
        problems = []

        def take(image: bytes, name: str, n: int) -> bytes:
            base = self.layout.address(name) * (BEAT if name.startswith(S.MEM_PREFIX) else 1)
            return image[base:base + n]

        for sfx in self.suffixes:
            x2 = np.frombuffer(take(vb, "x2" + sfx, 2 * self.cfg.hidden_size), dtype="<i2").astype(np.int64)
            if not np.array_equal(x2, self.expected["x2" + sfx]):
                problems.append(f"x2{sfx}: {int((x2 != self.expected['x2' + sfx]).sum())} of {len(x2)} elements differ")
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
        return problems
