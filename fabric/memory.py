"""The memory side of a layer: what a die keeps per context and how it moves.

A layer die owns, for every resident context, the Gated DeltaNet state and
convolution history of its three recurrent layers and, for its global layer,
the local window of recent keys and values, the compressed blocks of older
ones, and the index that chooses among the blocks.  All of it lives in the
die's local memory (LPDDR5X, HBM or the PSRAM modules of ``hw/``) behind one
port, and this module is the bit-exact reference for the units that use that
port as ``fabric/rtl/fabric_memory.sv`` implements them:

* ``MemoryMap``: the address map, record layouts and capacity per context;
* the recurrent side: state rows and history beats to and from the delta
  engine and the convolution (``fabric_row_dma``);
* the global side: the window append, block means, the 4-bit index key
  (``fabric_kv_append``), the index scan and top-K (``fabric_index_scan``,
  ``fabric_topk``) and the record reader that feeds the attention core
  (``fabric_record_reader``);
* ``GlobalContextMemory`` and ``GlobalContextMemoryFloat``: a context's
  global-layer stores in integer and in float, the float one reproducing
  ``fixed_llm_poc.SparseGlobalMixer`` token by token.

Retrieval semantics follow the reference model: a block is the mean of
``retrieval_block_size`` consecutive keys and values; its index key is the
L2-normalised mean of the ``index_k`` projections, quantised to 4 bits with a
per-vector absolute-maximum scale; a token's index query is its normalised
``index_q`` at the same 4 bits; a block is eligible once it ends
``local_window`` positions before the token; the ``top_blocks`` eligible
blocks by dot product join the window's rows in the softmax.

Memory port (``fabric_memory.sv``): ``DW``-bit beats (128), byte addresses
aligned to a beat, a request of up to 4095 beats, write beats following the
request, read beats returning in order, one request in flight per requester.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fabric.layer import NF, recip_fixed, rmsnorm_int, rnd_shr, sat, write_luts
from fabric.tile import write_hex

DW = 128                       # memory beat width in bits
BEAT = DW // 8                 # bytes per beat
INDEX_BITS = 4
ALIGN = 2048                   # region alignment: one device page, so records do not straddle bursts


def _beats(nbytes: int) -> int:
    return -(-nbytes // BEAT)


# ---------------------------------------------------------------------------
# Address map
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryMap:
    """Bytes per context of a layer die and the addresses of every record.

    Per context the die keeps, for each of its recurrent layers, the state
    (``v_heads x K x V`` at ``state_bits``) and the convolution history
    (``conv_dim x (kernel - 1)`` int8), and for its global layer the window
    (``local_window`` positions of a key and a value per KV head at
    ``kv_bits``), the block store (``context_tokens / block`` such records)
    and the index (one record of ``index_dim`` 4-bit codes and a scale per
    block).  Regions are laid out in that order, each aligned to a 2 KB page,
    and contexts are strided by the per-context total.
    """

    v_heads: int = 32
    k_dim: int = 128
    v_dim: int = 128
    conv_dim: int = 8192
    kernel: int = 4
    kv_heads: int = 4
    head_dim: int = 256
    local_window: int = 512
    block: int = 16
    context_tokens: int = 131072
    index_dim: int = 128
    recurrent_layers: int = 3
    global_layers: int = 1
    state_bits: int = 16
    kv_bits: int = 8

    @classmethod
    def from_config(cls, cfg, **overrides) -> "MemoryMap":
        values = dict(v_heads=cfg.linear_num_value_heads, k_dim=cfg.linear_key_head_dim, v_dim=cfg.linear_value_head_dim,
                      conv_dim=2 * cfg.linear_key_dim + cfg.linear_value_dim, kernel=cfg.linear_conv_kernel,
                      kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim, local_window=cfg.local_window,
                      block=cfg.retrieval_block_size, index_dim=cfg.index_dim, recurrent_layers=cfg.recurrent_every - 1,
                      global_layers=1)
        values.update(overrides)
        return cls(**values)

    # Records ---------------------------------------------------------------

    @property
    def state_row_bytes(self) -> int:
        return self.v_dim * self.state_bits // 8

    @property
    def state_bytes(self) -> int:
        """One recurrent layer's state for one context."""
        return self.v_heads * self.k_dim * self.state_row_bytes

    @property
    def hist_bytes(self) -> int:
        return _beats(self.conv_dim * (self.kernel - 1)) * BEAT

    @property
    def kv_half_bytes(self) -> int:
        """A key or a value of one KV head, padded to beats."""
        return _beats(self.head_dim * self.kv_bits // 8) * BEAT

    @property
    def kv_record_bytes(self) -> int:
        """Key then value of one position or block for one KV head."""
        return 2 * self.kv_half_bytes

    @property
    def window_bytes(self) -> int:
        return self.local_window * self.kv_heads * self.kv_record_bytes

    @property
    def blocks(self) -> int:
        return -(-self.context_tokens // self.block)

    @property
    def block_store_bytes(self) -> int:
        return self.blocks * self.kv_heads * self.kv_record_bytes

    @property
    def index_code_beats(self) -> int:
        return _beats(self.index_dim * INDEX_BITS // 8)

    @property
    def index_record_bytes(self) -> int:
        """The codes, then one beat holding the scale in its first byte."""
        return (self.index_code_beats + 1) * BEAT

    @property
    def index_bytes(self) -> int:
        return self.blocks * self.index_record_bytes

    # Layout ----------------------------------------------------------------

    @staticmethod
    def _align(n: int) -> int:
        return -(-n // ALIGN) * ALIGN

    def regions(self) -> dict[str, tuple[int, int]]:
        """``name -> (offset, bytes)`` within one context."""
        out: dict[str, tuple[int, int]] = {}
        offset = 0
        for layer in range(self.recurrent_layers):
            out[f"state{layer}"] = (offset, self.state_bytes)
            offset += self._align(self.state_bytes)
            out[f"hist{layer}"] = (offset, self.hist_bytes)
            offset += self._align(self.hist_bytes)
        for layer in range(self.global_layers):
            out[f"window{layer}"] = (offset, self.window_bytes)
            offset += self._align(self.window_bytes)
            out[f"blocks{layer}"] = (offset, self.block_store_bytes)
            offset += self._align(self.block_store_bytes)
            out[f"index{layer}"] = (offset, self.index_bytes)
            offset += self._align(self.index_bytes)
        out["_total"] = (0, offset)
        return out

    @property
    def context_bytes(self) -> int:
        return self.regions()["_total"][1]

    def contexts_that_fit(self, memory_bytes: int) -> int:
        return memory_bytes // self.context_bytes

    def context_base(self, ctx: int) -> int:
        return ctx * self.context_bytes

    def state_row_addr(self, ctx: int, layer: int, head: int, row: int) -> int:
        base = self.context_base(ctx) + self.regions()[f"state{layer}"][0]
        return base + (head * self.k_dim + row) * self.state_row_bytes

    def hist_addr(self, ctx: int, layer: int) -> int:
        return self.context_base(ctx) + self.regions()[f"hist{layer}"][0]

    def window_record_addr(self, ctx: int, pos: int, kv_head: int, layer: int = 0) -> int:
        base = self.context_base(ctx) + self.regions()[f"window{layer}"][0]
        return base + ((pos % self.local_window) * self.kv_heads + kv_head) * self.kv_record_bytes

    def block_record_addr(self, ctx: int, block: int, kv_head: int, layer: int = 0) -> int:
        base = self.context_base(ctx) + self.regions()[f"blocks{layer}"][0]
        return base + (block * self.kv_heads + kv_head) * self.kv_record_bytes

    def index_record_addr(self, ctx: int, block: int, layer: int = 0) -> int:
        base = self.context_base(ctx) + self.regions()[f"index{layer}"][0]
        return base + block * self.index_record_bytes

    # Traffic ---------------------------------------------------------------

    def bytes_per_token(self, pos: int, top_blocks: int) -> dict[str, int]:
        """Memory traffic of one token through the die at position ``pos``."""
        n_window = min(pos + 1, self.local_window)
        eligible = eligible_blocks(pos, self.local_window, self.block)
        return {
            "state": 2 * self.recurrent_layers * self.state_bytes,
            "history": 2 * self.recurrent_layers * self.conv_dim * (self.kernel - 1),
            "window_append": self.kv_heads * self.kv_record_bytes,
            "window_read": n_window * self.kv_heads * self.kv_record_bytes,
            "block_append": (self.kv_heads * self.kv_record_bytes + self.index_record_bytes) if (pos + 1) % self.block == 0 else 0,
            "index_scan": eligible * self.index_record_bytes,
            "block_read": min(top_blocks, eligible) * self.kv_heads * self.kv_record_bytes,
        }

    def report_markdown(self, memory_bytes: dict[str, int], top_blocks: int = 32) -> str:
        regions = self.regions()
        lines = [f"Per context ({self.context_tokens} tokens, state int{self.state_bits}, KV int{self.kv_bits}):", "",
                 "| Region | Bytes |", "| --- | ---: |"]
        for name, (_, size) in regions.items():
            if name != "_total":
                lines.append(f"| {name} | {size:,} |")
        lines.append(f"| **total** | **{self.context_bytes:,}** |")
        lines += ["", "| Memory | Contexts |", "| --- | ---: |"]
        for name, size in memory_bytes.items():
            lines.append(f"| {name} ({size // 2**20:,} MB) | {self.contexts_that_fit(size)} |")
        traffic = self.bytes_per_token(self.context_tokens - 1, top_blocks)
        lines += ["", f"Traffic per token at the end of the context: {sum(traffic.values()):,} bytes", "",
                  "| Component | Bytes |", "| --- | ---: |"]
        lines += [f"| {k} | {v:,} |" for k, v in traffic.items()]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Integer functions of the global side
# ---------------------------------------------------------------------------


def kv_quant(x: np.ndarray, bits: int) -> np.ndarray:
    """The int8 values the store holds: int8 as is, int4 as ``sat4(x / 16) * 16``."""
    x = np.asarray(x, dtype=np.int64)
    if bits == 8:
        return x
    if bits == 4:
        return sat(rnd_shr(x, 4), 4) * 16
    raise ValueError("kv_bits must be 4 or 8")


def kv_pack(x: np.ndarray, bits: int, nbytes: int) -> bytes:
    """A stored int8 vector as record bytes, element ``e`` at bit ``bits * e``."""
    x = np.asarray(x, dtype=np.int64)
    out = bytearray(nbytes)
    if bits == 8:
        out[:len(x)] = bytes(int(v) & 0xFF for v in x)
    else:
        codes = x >> 4                                  # the values are multiples of 16
        for e, c in enumerate(codes):
            out[e // 2] |= (int(c) & 0xF) << (4 * (e % 2))
    return bytes(out)


def kv_unpack(data: bytes, bits: int, n: int) -> np.ndarray:
    if bits == 8:
        return np.frombuffer(bytes(data[:n]), dtype=np.int8).astype(np.int64)
    codes = np.array([(data[e // 2] >> (4 * (e % 2))) & 0xF for e in range(n)], dtype=np.int64)
    return np.where(codes >= 8, codes - 16, codes) * 16


def block_mean_int(total: np.ndarray, block: int) -> np.ndarray:
    """Mean of ``block`` int8 rows from their sum, ``block`` a power of two."""
    if block & (block - 1):
        raise ValueError("block size must be a power of two")
    return rnd_shr(np.asarray(total, dtype=np.int64), int(math.log2(block)))


def index_unit(total_or_vec: np.ndarray, block: int = 1) -> np.ndarray:
    """An int8 unit vector (scale 2^-7) from the sum of ``block`` index projections."""
    mean = block_mean_int(total_or_vec, block) if block > 1 else np.asarray(total_or_vec, dtype=np.int64)
    return rmsnorm_int(mean, 1, 1, NF - 7, xw=8, ow=8)


def index_codes(u: np.ndarray) -> tuple[np.ndarray, int]:
    """4-bit codes of an int8 unit vector with its absolute maximum as scale:
    ``code = round((u / scale + 1) * 15 / 2)`` via the reciprocal unit."""
    u = np.asarray(u, dtype=np.int64)
    scale = max(int(np.abs(u).max()), 1)
    r, lz = recip_fixed(2 * scale, 9)
    codes = rnd_shr(15 * (u + scale) * r, 24 - lz)
    return np.clip(codes, 0, 15), scale


def index_codes_float(x: np.ndarray, levels: int = 16) -> tuple[np.ndarray, float]:
    """The reference's fake quantiser: codes and the absmax scale of a float vector."""
    scale = max(float(np.abs(x).max()), 1e-6)
    normalized = np.clip(x / scale, -1.0, 1.0)
    return np.rint((normalized + 1) * (levels - 1) / 2), scale


def index_score(q_codes: np.ndarray, k_codes: np.ndarray, k_scale: int) -> int:
    """``k_scale * sum (2q - 15)(2k - 15)``: the dot product of the two
    dequantised vectors up to the query's own scale, common to every block."""
    q = 2 * np.asarray(q_codes, dtype=np.int64) - 15
    k = 2 * np.asarray(k_codes, dtype=np.int64) - 15
    return int(k_scale) * int((q * k).sum())


def eligible_blocks(pos: int, window: int, block: int) -> int:
    """Blocks whose last position is at least ``window`` before ``pos``."""
    span = pos - window + 1
    return span // block if span >= 0 else 0


def topk_stream(candidates, k: int) -> list[tuple[int, int]]:
    """Streaming insertion into a list sorted by score: an earlier candidate
    keeps its place against an equal later one.  Returns ``[(id, score)]``
    in rank order."""
    return sorted(candidates, key=lambda c: -c[1])[:k]


def index_codes_pack(codes: np.ndarray, nbytes: int) -> bytes:
    out = bytearray(nbytes)
    for e, c in enumerate(codes):
        out[e // 2] |= (int(c) & 0xF) << (4 * (e % 2))
    return bytes(out)


def index_codes_unpack(data: bytes, n: int) -> np.ndarray:
    return np.array([(data[e // 2] >> (4 * (e % 2))) & 0xF for e in range(n)], dtype=np.int64)


# ---------------------------------------------------------------------------
# A memory image
# ---------------------------------------------------------------------------


class MemoryImage:
    """Bytes of the die memory, written and read by records, dumped as beats."""

    def __init__(self, nbytes: int):
        self.data = bytearray(nbytes)

    def write(self, addr: int, payload: bytes) -> None:
        self.data[addr:addr + len(payload)] = payload

    def read(self, addr: int, n: int) -> bytes:
        return bytes(self.data[addr:addr + n])

    def to_hex(self, path: Path) -> None:
        beats = len(self.data) // BEAT
        write_hex(path, [int.from_bytes(self.data[b * BEAT:(b + 1) * BEAT], "little") for b in range(beats)], DW)

    @classmethod
    def from_hex(cls, path: Path, nbytes: int) -> "MemoryImage":
        image = cls(nbytes)
        for b, line in enumerate(path.read_text(encoding="utf-8").split()):
            image.data[b * BEAT:(b + 1) * BEAT] = int(line, 16).to_bytes(BEAT, "little")
        return image


# ---------------------------------------------------------------------------
# A context's global-layer stores
# ---------------------------------------------------------------------------


class GlobalContextMemory:
    """The window, block store and index of one context, in the integer
    formats the memory holds, kept in a ``MemoryImage`` through the map."""

    def __init__(self, mm: MemoryMap, top_blocks: int, image: MemoryImage | None = None, ctx: int = 0):
        self.mm, self.top, self.ctx = mm, top_blocks, ctx
        self.image = image if image is not None else MemoryImage(mm.context_bytes * (ctx + 1))
        self.sum_k = np.zeros((mm.kv_heads, mm.head_dim), dtype=np.int64)
        self.sum_v = np.zeros((mm.kv_heads, mm.head_dim), dtype=np.int64)
        self.sum_idx = np.zeros(mm.index_dim, dtype=np.int64)
        self.next_pos = 0

    def _write_record(self, addr: int, k: np.ndarray, v: np.ndarray) -> None:
        half = self.mm.kv_half_bytes
        self.image.write(addr, kv_pack(kv_quant(k, self.mm.kv_bits), self.mm.kv_bits, half))
        self.image.write(addr + half, kv_pack(kv_quant(v, self.mm.kv_bits), self.mm.kv_bits, half))

    def _read_record(self, addr: int) -> tuple[np.ndarray, np.ndarray]:
        mm = self.mm
        k = kv_unpack(self.image.read(addr, mm.kv_half_bytes), mm.kv_bits, mm.head_dim)
        v = kv_unpack(self.image.read(addr + mm.kv_half_bytes, mm.kv_half_bytes), mm.kv_bits, mm.head_dim)
        return k, v

    def append(self, pos: int, k: np.ndarray, v: np.ndarray, idx_k: np.ndarray) -> dict:
        """This token's int8 keys and values ``[kv_heads, hd]`` and its raw
        int8 ``index_k`` into the stores; a completed block is finalised."""
        mm = self.mm
        if pos != self.next_pos:
            raise ValueError("positions must be appended in order")
        self.next_pos = pos + 1
        k = np.asarray(k, dtype=np.int64)
        v = np.asarray(v, dtype=np.int64)
        for n in range(mm.kv_heads):
            self._write_record(mm.window_record_addr(self.ctx, pos, n), k[n], v[n])
        self.sum_k += k
        self.sum_v += v
        self.sum_idx += np.asarray(idx_k, dtype=np.int64)
        out: dict = {"block": None}
        if (pos + 1) % mm.block == 0:
            b = pos // mm.block
            k_bar = block_mean_int(self.sum_k, mm.block)
            v_bar = block_mean_int(self.sum_v, mm.block)
            for n in range(mm.kv_heads):
                self._write_record(mm.block_record_addr(self.ctx, b, n), k_bar[n], v_bar[n])
            u = index_unit(self.sum_idx, mm.block)
            codes, scale = index_codes(u)
            record = index_codes_pack(codes, mm.index_code_beats * BEAT) + bytes([scale]) + bytes(BEAT - 1)
            self.image.write(mm.index_record_addr(self.ctx, b), record)
            out.update(block=b, k_bar=k_bar, v_bar=v_bar, unit=u, codes=codes, scale=scale)
            self.sum_k[:] = 0
            self.sum_v[:] = 0
            self.sum_idx[:] = 0
        return out

    def scan(self, pos: int, q_unit: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
        """The eligible blocks scored against the token's int8 unit query; the top-K in rank order."""
        mm = self.mm
        q_codes, _ = index_codes(q_unit)
        candidates = []
        for b in range(eligible_blocks(pos, mm.local_window, mm.block)):
            record = self.image.read(mm.index_record_addr(self.ctx, b), mm.index_record_bytes)
            k_codes = index_codes_unpack(record, mm.index_dim)
            candidates.append((b, index_score(q_codes, k_codes, record[mm.index_code_beats * BEAT])))
        return topk_stream(candidates, self.top), q_codes

    def rows(self, pos: int, selected: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Key and value rows ``[kv_heads, N, hd]`` for the attention core: the
        window in position order, then the selected blocks in rank order;
        also the record addresses in that order for KV head 0."""
        mm = self.mm
        positions = list(range(max(0, pos - mm.local_window + 1), pos + 1))
        addrs = [[mm.window_record_addr(self.ctx, p, n) for n in range(mm.kv_heads)] for p in positions]
        addrs += [[mm.block_record_addr(self.ctx, b, n) for n in range(mm.kv_heads)] for b, _ in selected]
        k_rows = np.zeros((mm.kv_heads, len(addrs), mm.head_dim), dtype=np.int64)
        v_rows = np.zeros_like(k_rows)
        head0 = []
        for i, per_head in enumerate(addrs):
            for n, addr in enumerate(per_head):
                k_rows[n, i], v_rows[n, i] = self._read_record(addr)
                if n == 0:
                    head0.append(addr)
        return k_rows, v_rows, head0

    def retrieve(self, pos: int, q_unit: np.ndarray) -> dict:
        selected, q_codes = self.scan(pos, q_unit)
        k_rows, v_rows, addrs = self.rows(pos, selected)
        return {"selected": selected, "q_codes": q_codes, "k_rows": k_rows, "v_rows": v_rows, "addrs": addrs}


class GlobalContextMemoryFloat:
    """The same stores in float with the reference model's exact arithmetic."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.keys: list[np.ndarray] = []                 # per position [kv_heads, hd]
        self.values: list[np.ndarray] = []
        self.idx: list[np.ndarray] = []
        self.k_blocks: list[np.ndarray] = []
        self.v_blocks: list[np.ndarray] = []
        self.index_keys: list[np.ndarray] = []           # dequantised, as the reference scores them

    def append(self, pos: int, k: np.ndarray, v: np.ndarray, idx_k: np.ndarray) -> None:
        cfg = self.cfg
        self.keys.append(np.asarray(k, dtype=np.float64))
        self.values.append(np.asarray(v, dtype=np.float64))
        self.idx.append(np.asarray(idx_k, dtype=np.float64))
        bs = cfg.retrieval_block_size
        if (pos + 1) % bs == 0:
            lo = pos + 1 - bs
            self.k_blocks.append(np.mean(self.keys[lo:pos + 1], axis=0))
            self.v_blocks.append(np.mean(self.values[lo:pos + 1], axis=0))
            m = np.mean(self.idx[lo:pos + 1], axis=0)
            m = m / max(np.linalg.norm(m), 1e-12)
            codes, scale = index_codes_float(m, 2 ** cfg.index_bits)
            self.index_keys.append((codes * 2 / (2 ** cfg.index_bits - 1) - 1) * scale)

    def retrieve(self, pos: int, idx_q: np.ndarray) -> dict:
        cfg = self.cfg
        w, bs = cfg.local_window, cfg.retrieval_block_size
        q = np.asarray(idx_q, dtype=np.float64)
        q = q / max(np.linalg.norm(q), 1e-12)
        codes, scale = index_codes_float(q, 2 ** cfg.index_bits)
        q = (codes * 2 / (2 ** cfg.index_bits - 1) - 1) * scale
        n = eligible_blocks(pos, w, bs)
        scores = [(b, float(q @ self.index_keys[b])) for b in range(n)]
        selected = sorted(scores, key=lambda c: -c[1])[:cfg.top_blocks]
        positions = list(range(max(0, pos - w + 1), pos + 1))
        k_rows = np.stack([self.keys[p] for p in positions] + [self.k_blocks[b] for b, _ in selected], axis=1)
        v_rows = np.stack([self.values[p] for p in positions] + [self.v_blocks[b] for b, _ in selected], axis=1)
        return {"selected": selected, "k_rows": k_rows, "v_rows": v_rows}


# ---------------------------------------------------------------------------
# Vectors for the RTL testbenches
# ---------------------------------------------------------------------------


def _params(directory: Path, **params) -> dict:
    (directory / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    return params


def emit_topk_vectors(directory: Path, rng: np.random.Generator, k: int, n: int) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    scores = rng.integers(-(1 << 20), 1 << 20, n)
    scores[n // 3] = scores[n // 4]                        # a tie: the earlier one ranks first
    candidates = [(i, int(s)) for i, s in enumerate(scores)]
    ranked = topk_stream(candidates, k)
    write_hex(directory / "scores.hex", scores, 24)
    write_hex(directory / "expected_id.hex", [i for i, _ in ranked], 16)
    write_hex(directory / "expected_score.hex", [s for _, s in ranked], 24)
    return _params(directory, K=k, N=n, EXPECTED=len(ranked))


def emit_index_scan_vectors(directory: Path, rng: np.random.Generator, mm: MemoryMap, blocks: int, k: int) -> dict:
    """An index region of ``blocks`` records and a query; expected top-K."""
    directory.mkdir(parents=True, exist_ok=True)
    image = MemoryImage(mm.index_record_addr(0, blocks))
    candidates = []
    q_unit = index_unit(rng.integers(-128, 128, mm.index_dim))
    q_codes, _ = index_codes(q_unit)
    for b in range(blocks):
        codes, scale = index_codes(index_unit(rng.integers(-128, 128, mm.index_dim)))
        record = index_codes_pack(codes, mm.index_code_beats * BEAT) + bytes([scale]) + bytes(BEAT - 1)
        image.write(mm.index_record_addr(0, b), record)
        candidates.append((b, index_score(q_codes, codes, scale)))
    ranked = topk_stream(candidates, k)
    image.to_hex(directory / "mem.hex")
    write_hex(directory / "q_codes.hex", [int(sum(int(c) << (4 * e) for e, c in enumerate(q_codes)))], 4 * mm.index_dim)
    write_hex(directory / "expected_id.hex", [i for i, _ in ranked], 16)
    write_hex(directory / "expected_score.hex", [s for _, s in ranked], 32)
    return _params(directory, IDIM=mm.index_dim, BLOCKS=blocks, K=k, BASE=mm.index_record_addr(0, 0),
                   REC_BEATS=mm.index_code_beats + 1, WORDS=len(image.data) // BEAT, EXPECTED=len(ranked))


def emit_kv_append_vectors(directory: Path, rng: np.random.Generator, mm: MemoryMap, tokens: int, top: int) -> dict:
    """``tokens`` appends from position 0; the expected memory image after them."""
    directory.mkdir(parents=True, exist_ok=True)
    write_luts(directory)
    store = GlobalContextMemory(mm, top)
    ks = rng.integers(-128, 128, (tokens, mm.kv_heads, mm.head_dim))
    vs = rng.integers(-128, 128, (tokens, mm.kv_heads, mm.head_dim))
    idx = rng.integers(-128, 128, (tokens, mm.index_dim))
    for pos in range(tokens):
        store.append(pos, ks[pos], vs[pos], idx[pos])
    pack8 = lambda row: int(sum((int(e) & 0xFF) << (8 * j) for j, e in enumerate(row)))
    write_hex(directory / "k.hex", [pack8(ks[t].reshape(-1)) for t in range(tokens)], 8 * mm.kv_heads * mm.head_dim)
    write_hex(directory / "v.hex", [pack8(vs[t].reshape(-1)) for t in range(tokens)], 8 * mm.kv_heads * mm.head_dim)
    write_hex(directory / "idx.hex", [pack8(idx[t]) for t in range(tokens)], 8 * mm.index_dim)
    store.image.to_hex(directory / "expected_mem.hex")
    regions = mm.regions()
    return _params(directory, TOKENS=tokens, HD=mm.head_dim, NKV=mm.kv_heads, IDIM=mm.index_dim, BS=mm.block,
                   KV_BITS=mm.kv_bits, W=mm.local_window, WINDOW_BASE=regions["window0"][0],
                   BLOCK_BASE=regions["blocks0"][0], INDEX_BASE=regions["index0"][0],
                   WORDS=len(store.image.data) // BEAT)


def emit_record_reader_vectors(directory: Path, rng: np.random.Generator, mm: MemoryMap, tokens: int, top: int,
                               g: int, lanes: int) -> dict:
    """A filled store, one token's retrieval, and the attention over its rows."""
    from fabric.layer import attention_int
    directory.mkdir(parents=True, exist_ok=True)
    write_luts(directory)
    store = GlobalContextMemory(mm, top)
    for pos in range(tokens):
        store.append(pos, rng.integers(-128, 128, (mm.kv_heads, mm.head_dim)),
                     rng.integers(-128, 128, (mm.kv_heads, mm.head_dim)), rng.integers(-128, 128, mm.index_dim))
    pos = tokens - 1
    got = store.retrieve(pos, index_unit(rng.integers(-128, 128, mm.index_dim)))
    q = rng.integers(-128, 128, (g, mm.head_dim))
    gate = rng.integers(-128, 128, (g, mm.head_dim))
    consts = dict(mult_s=int(rng.integers(1 << 12, 1 << 16)), sh_s=18, mult_gate=int(rng.integers(1, 1 << 16)), sh_gate=12,
                  mult_o=int(rng.integers(1, 1 << 16)), sh_o=24)
    out = attention_int(q, gate, got["k_rows"][0], got["v_rows"][0], **consts)
    pack8 = lambda row: int(sum((int(e) & 0xFF) << (8 * j) for j, e in enumerate(row)))
    store.image.to_hex(directory / "mem.hex")
    write_hex(directory / "addrs.hex", got["addrs"], 32)
    write_hex(directory / "q.hex", [pack8(row) for row in q], 8 * mm.head_dim)
    write_hex(directory / "gate.hex", [pack8(row) for row in gate], 8 * mm.head_dim)
    write_hex(directory / "expected_out.hex", [pack8(row) for row in out], 8 * mm.head_dim)
    return _params(directory, HD=mm.head_dim, KV_BITS=mm.kv_bits, G=g, L=lanes, N=len(got["addrs"]),
                   REC_BEATS=mm.kv_record_bytes // BEAT, WORDS=len(store.image.data) // BEAT,
                   **{k.upper(): v for k, v in consts.items()})


def emit_row_dma_vectors(directory: Path, rng: np.random.Generator, k: int, v: int) -> dict:
    """One head's state in memory, one token of the delta rule through the
    DMA and the engine, and the memory image expected afterwards."""
    from fabric.layer import delta_state_int, ysh_for
    directory.mkdir(parents=True, exist_ok=True)
    row_bytes = v * 2
    base = 4096
    image = MemoryImage(base + k * row_bytes + BEAT)
    s = rng.integers(-32768, 32768, (k, v))
    for i in range(k):
        image.write(base + i * row_bytes, b"".join((int(e) & 0xFFFF).to_bytes(2, "little") for e in s[i]))
    image.to_hex(directory / "mem.hex")
    kq = rng.standard_normal((2, k))
    kq = np.rint(kq / np.linalg.norm(kq, axis=1, keepdims=True) * 127).astype(np.int64)
    vv = rng.integers(-128, 128, v)
    decay, beta = int(rng.integers(0, 1 << 16)), int(rng.integers(0, 1 << 16))
    s_new, y = delta_state_int(s, kq[0], vv, kq[1], decay, beta)
    for i in range(k):
        image.write(base + i * row_bytes, b"".join((int(e) & 0xFFFF).to_bytes(2, "little") for e in s_new[i]))
    image.to_hex(directory / "expected_mem.hex")
    write_hex(directory / "k.hex", kq[0], 8)
    write_hex(directory / "q.hex", kq[1], 8)
    write_hex(directory / "v.hex", vv, 8)
    write_hex(directory / "expected_y.hex", y, 16)
    return _params(directory, K=k, V=v, DECAY=decay, BETA=beta, YSH=ysh_for(k), BASE=base,
                   WORDS=len(image.data) // BEAT)
