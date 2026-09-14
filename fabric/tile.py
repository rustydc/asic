"""Fixed-weight fabric tile: bit-exact reference model, coefficient compiler, and mapping.

A tile is a via-programmed ROM of ``rows x cols`` 4-bit coefficients feeding
``cols`` multiply-accumulate columns.  Each cycle it reads ``rows_per_cycle``
ROM rows, forms the coefficient multiples of the matching activations, routes
the selected multiple into every column accumulator, and after ``rows /
rows_per_cycle`` cycles emits ``cols`` partial sums.  Matrices wider than a tile
use more tiles across columns; matrices deeper than a tile use more tiles down
the rows and a partial-sum reduction.

This module is the golden model for ``fabric/rtl/fabric_tile.sv`` and the
source of the via pattern that personalizes a base die.  Everything in
``DensityModel`` is a placeholder until the MPW tile is measured.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Tile specification and integer arithmetic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileSpec:
    rows: int = 4096            # input dimensions per tile (ROM wordlines)
    cols: int = 64              # outputs per tile (MAC columns)
    weight_bits: int = 4        # symmetric signed coefficients, -(2^(b-1)-1) .. 2^(b-1)-1
    act_bits: int = 8           # signed activations
    rows_per_cycle: int = 2     # ROM banks read in parallel
    acc_bits: int = 24          # accumulator width
    scale_bits: int = 16        # per-column requantization multiplier width (unsigned)
    shift_bits: int = 5         # per-column requantization shift width

    def __post_init__(self) -> None:
        if self.rows % self.rows_per_cycle:
            raise ValueError("rows must be a multiple of rows_per_cycle")
        if not 2 <= self.weight_bits <= 8 or self.act_bits not in (4, 8):
            raise ValueError("unsupported weight or activation width")
        worst = self.rows * self.act_max * self.weight_max
        if worst >= 2 ** (self.acc_bits - 1):
            raise ValueError("accumulator too narrow for the tile depth")

    @property
    def weight_max(self) -> int:
        return 2 ** (self.weight_bits - 1) - 1

    @property
    def act_max(self) -> int:
        return 2 ** (self.act_bits - 1) - 1

    @property
    def cycles_per_pass(self) -> int:
        return self.rows // self.rows_per_cycle

    @property
    def coefficients(self) -> int:
        return self.rows * self.cols

    @property
    def rom_bits(self) -> int:
        return self.coefficients * self.weight_bits


def saturate(value: np.ndarray, bits: int) -> np.ndarray:
    limit = 2 ** (bits - 1)
    return np.clip(value, -limit, limit - 1)


def requantize(acc: np.ndarray, mult: np.ndarray, shift: np.ndarray, spec: TileSpec) -> np.ndarray:
    """Hardware requantization: ``sat8((acc * mult + 2^(shift-1)) >> shift)`` with arithmetic shift."""
    acc = acc.astype(np.int64)
    product = acc * mult.astype(np.int64)
    rounding = np.where(shift > 0, np.int64(1) << np.maximum(shift.astype(np.int64) - 1, 0), 0)
    shifted = (product + rounding) >> shift.astype(np.int64)
    return saturate(shifted, spec.act_bits).astype(np.int8)


# ---------------------------------------------------------------------------
# Quantization and compilation
# ---------------------------------------------------------------------------


@dataclass
class QuantizedMatrix:
    """Integer weights ``[in_features, out_features]`` plus per-output requantization."""

    weights: np.ndarray          # int8 array holding weight_bits-wide values
    mult: np.ndarray             # uint16 per output column
    shift: np.ndarray            # uint8 per output column
    scale: np.ndarray            # float per output column (for reference only)

    @property
    def in_features(self) -> int:
        return self.weights.shape[0]

    @property
    def out_features(self) -> int:
        return self.weights.shape[1]


def quantize_matrix(weight: np.ndarray, spec: TileSpec, *, act_scale: float = 1.0,
                    out_scale: float = 1.0) -> QuantizedMatrix:
    """Symmetric per-output-channel quantization of ``weight[in, out]``.

    ``act_scale`` is the real value of one activation LSB and ``out_scale`` the
    real value of one output LSB; the per-column requantizer maps the integer
    accumulator back to output units.  QAT should supply all three; this
    absmax rule is the training-free fallback.
    """
    weight = np.asarray(weight, dtype=np.float64)
    absmax = np.abs(weight).max(axis=0)
    scale = np.where(absmax > 0, absmax / spec.weight_max, 1.0)
    q = np.rint(weight / scale)
    q = saturate(q, spec.weight_bits).astype(np.int8)
    q = np.maximum(q, -spec.weight_max)  # symmetric alphabet, never -(2^(b-1))
    real_per_acc_lsb = scale * act_scale / out_scale
    mult, shift = fixed_point_scale(real_per_acc_lsb, spec)
    return QuantizedMatrix(q, mult, shift, scale)


def fixed_point_scale(value: np.ndarray, spec: TileSpec) -> tuple[np.ndarray, np.ndarray]:
    """Represent ``value`` as ``mult / 2^shift`` with ``mult`` in scale_bits and shift in shift_bits."""
    value = np.asarray(value, dtype=np.float64)
    max_shift = 2 ** spec.shift_bits - 1
    max_mult = 2 ** spec.scale_bits - 1
    shift = np.clip(np.floor(np.log2(max_mult / np.maximum(value, 1e-30))), 0, max_shift).astype(np.uint8)
    mult = np.clip(np.rint(value * (2.0 ** shift)), 0, max_mult).astype(np.uint16)
    return mult, shift


@dataclass
class Tile:
    """One personalized tile: its via map and requantization constants."""

    row_block: int
    col_block: int
    via_map: np.ndarray          # int8 [rows, cols] coefficients (zero-padded)
    mult: np.ndarray             # uint16 [cols]
    shift: np.ndarray            # uint8 [cols]
    valid_rows: int
    valid_cols: int

    def rom_words(self, spec: TileSpec) -> np.ndarray:
        """ROM contents as unsigned ``weight_bits``-wide words, ``[rows, cols]``."""
        mask = (1 << spec.weight_bits) - 1
        return (self.via_map.astype(np.int32) & mask).astype(np.uint16)

    def via_count(self, spec: TileSpec) -> int:
        """Number of programmed vias: one per set ROM bit."""
        words = self.rom_words(spec).astype(np.uint32)
        return int(sum(int(((words >> bit) & 1).sum()) for bit in range(spec.weight_bits)))


@dataclass
class CompiledMatrix:
    in_features: int
    out_features: int
    spec: TileSpec
    tiles: list[Tile] = field(default_factory=list)

    @property
    def row_blocks(self) -> int:
        return math.ceil(self.in_features / self.spec.rows)

    @property
    def col_blocks(self) -> int:
        return math.ceil(self.out_features / self.spec.cols)

    @property
    def utilization(self) -> float:
        return self.in_features * self.out_features / (len(self.tiles) * self.spec.coefficients)


def compile_matrix(q: QuantizedMatrix, spec: TileSpec) -> CompiledMatrix:
    """Cut a quantized matrix into zero-padded tiles."""
    compiled = CompiledMatrix(q.in_features, q.out_features, spec)
    for rb in range(compiled.row_blocks):
        r0, r1 = rb * spec.rows, min((rb + 1) * spec.rows, q.in_features)
        for cb in range(compiled.col_blocks):
            c0, c1 = cb * spec.cols, min((cb + 1) * spec.cols, q.out_features)
            via = np.zeros((spec.rows, spec.cols), dtype=np.int8)
            via[: r1 - r0, : c1 - c0] = q.weights[r0:r1, c0:c1]
            mult = np.zeros(spec.cols, dtype=np.uint16)
            shift = np.zeros(spec.cols, dtype=np.uint8)
            mult[: c1 - c0] = q.mult[c0:c1]
            shift[: c1 - c0] = q.shift[c0:c1]
            compiled.tiles.append(Tile(rb, cb, via, mult, shift, r1 - r0, c1 - c0))
    return compiled


def decompile(compiled: CompiledMatrix) -> np.ndarray:
    """Reassemble the integer weight matrix from tiles (round-trip check)."""
    spec = compiled.spec
    out = np.zeros((compiled.in_features, compiled.out_features), dtype=np.int8)
    for tile in compiled.tiles:
        r0, c0 = tile.row_block * spec.rows, tile.col_block * spec.cols
        out[r0:r0 + tile.valid_rows, c0:c0 + tile.valid_cols] = tile.via_map[: tile.valid_rows, : tile.valid_cols]
    return out


# ---------------------------------------------------------------------------
# Bit-exact tile execution
# ---------------------------------------------------------------------------


def coefficient_multiples(x: np.ndarray, spec: TileSpec) -> np.ndarray:
    """The shared multiples ``0..weight_max`` of each activation, as the hardware forms them."""
    x = x.astype(np.int64)
    return np.stack([k * x for k in range(spec.weight_max + 1)], axis=-1)


def tile_forward(tile: Tile, x: np.ndarray, spec: TileSpec, psum_in: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Run one tile over an activation vector.

    Returns ``(psum, y)``: the raw accumulators after the pass and the
    requantized outputs.  The accumulation order is the hardware's: banks of
    ``rows_per_cycle`` rows, in row order, so overflow behaviour matches.
    """
    x = np.asarray(x, dtype=np.int64)
    if x.shape != (spec.rows,):
        raise ValueError(f"expected {spec.rows} activations, got {x.shape}")
    acc = np.zeros(spec.cols, dtype=np.int64) if psum_in is None else psum_in.astype(np.int64).copy()
    multiples = coefficient_multiples(x, spec)                      # [rows, weight_max + 1]
    w = tile.via_map.astype(np.int64)                               # [rows, cols]
    magnitude = np.abs(w)
    sign = np.sign(w)
    for cycle in range(spec.cycles_per_pass):
        rows = range(cycle * spec.rows_per_cycle, (cycle + 1) * spec.rows_per_cycle)
        for r in rows:
            selected = multiples[r][magnitude[r]]                  # via-selected multiple per column
            acc += sign[r] * selected
        acc = saturate(acc, spec.acc_bits)                          # accumulator wraps are a design error; saturate
    return acc, requantize(acc, tile.mult, tile.shift, spec)


def matrix_forward(compiled: CompiledMatrix, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full matrix through its tiles: parallel column blocks, reduced row blocks."""
    spec = compiled.spec
    x = np.asarray(x, dtype=np.int64)
    if x.shape != (compiled.in_features,):
        raise ValueError("activation length must equal in_features")
    padded = np.zeros(compiled.row_blocks * spec.rows, dtype=np.int64)
    padded[: compiled.in_features] = x
    psum = np.zeros((compiled.col_blocks, spec.cols), dtype=np.int64)
    for tile in compiled.tiles:
        chunk = padded[tile.row_block * spec.rows:(tile.row_block + 1) * spec.rows]
        partial, _ = tile_forward(tile, chunk, spec)
        psum[tile.col_block] += partial
    psum = saturate(psum, spec.acc_bits)
    y = np.zeros((compiled.col_blocks, spec.cols), dtype=np.int8)
    for cb in range(compiled.col_blocks):
        tile = next(t for t in compiled.tiles if t.col_block == cb)
        y[cb] = requantize(psum[cb], tile.mult, tile.shift, spec)
    return psum.reshape(-1)[: compiled.out_features], y.reshape(-1)[: compiled.out_features]


def reference_matmul(q: QuantizedMatrix, x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.int64) @ q.weights.astype(np.int64)


# ---------------------------------------------------------------------------
# Via pattern export
# ---------------------------------------------------------------------------


def via_coordinates(tile: Tile, spec: TileSpec, *, bit_pitch_nm: int = 100, row_pitch_nm: int = 200,
                    origin_nm: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Programmed via centres ``[n, 2]`` in nanometres for one tile's ROM array.

    Layout convention: rows run along y (wordlines), and each column occupies
    ``weight_bits`` adjacent bit-cell pitches along x.  The GDS writer applies
    the foundry's actual bit-cell geometry; this fixes the addressing.
    """
    words = tile.rom_words(spec)
    rows, cols = np.nonzero(words)
    coords = []
    for r, c in zip(rows, cols):
        word = int(words[r, c])
        for bit in range(spec.weight_bits):
            if (word >> bit) & 1:
                x = origin_nm[0] + (c * spec.weight_bits + bit) * bit_pitch_nm + bit_pitch_nm // 2
                y = origin_nm[1] + r * row_pitch_nm + row_pitch_nm // 2
                coords.append((x, y))
    return np.array(coords, dtype=np.int64).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Model mapping and density model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixShape:
    name: str
    in_features: int
    out_features: int
    pass_group: int             # matrices in the same group run in parallel on one activation


def layer_matrices(cfg, layer_idx: int) -> list[MatrixShape]:
    """Fabric matrices of one decoder layer, grouped into sequential passes."""
    d = cfg.hidden_size
    inter = cfg.layer_intermediate_size(layer_idx)
    if cfg.is_global_layer(layer_idx):
        nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        mats = [MatrixShape("q_proj", d, nh * hd * 2, 0), MatrixShape("k_proj", d, nkv * hd, 0),
                MatrixShape("v_proj", d, nkv * hd, 0), MatrixShape("index_q", d, cfg.index_dim, 0),
                MatrixShape("index_k", d, cfg.index_dim, 0), MatrixShape("o_proj", nh * hd, d, 1)]
    else:
        kd, vd, nv = cfg.linear_key_dim, cfg.linear_value_dim, cfg.linear_num_value_heads
        mats = [MatrixShape("in_proj_qkv", d, 2 * kd + vd, 0), MatrixShape("in_proj_z", d, vd, 0),
                MatrixShape("in_proj_b", d, nv, 0), MatrixShape("in_proj_a", d, nv, 0),
                MatrixShape("out_proj", vd, d, 1)]
    mats += [MatrixShape("gate_proj", d, inter, 2), MatrixShape("up_proj", d, inter, 2),
             MatrixShape("down_proj", inter, d, 3)]
    return mats


def head_matrices(cfg, num_head_asics: int = 2) -> list[MatrixShape]:
    rows = math.ceil(cfg.vocab_size / num_head_asics)
    return [MatrixShape("lm_head_slice", cfg.hidden_size, rows, 0)]


def tiles_for(shape: MatrixShape, spec: TileSpec) -> int:
    return math.ceil(shape.in_features / spec.rows) * math.ceil(shape.out_features / spec.cols)


@dataclass(frozen=True)
class DensityModel:
    """Area, clock, and energy placeholders (28 nm class). Replace with MPW measurements."""

    node: str = "28nm-class placeholder"
    rom_um2_per_bit: float = 0.03           # via-programmed ROM bit cell incl. array overhead
    mac_um2_per_column_per_bank: float = 200.0   # 8:1 multiple select + add/sub, per rows_per_cycle unit
    acc_um2_per_column: float = 150.0       # accumulator, requantizer share, output register
    tile_overhead_um2: float = 2500.0       # multiples generator, ROM periphery, control per tile
    clock_mhz: float = 800.0
    rom_fj_per_bit: float = 3.0
    mac_fj_per_coefficient: float = 60.0

    def tile_area_um2(self, spec: TileSpec) -> float:
        rom = spec.rom_bits * self.rom_um2_per_bit
        mac = spec.cols * (spec.rows_per_cycle * self.mac_um2_per_column_per_bank + self.acc_um2_per_column)
        return rom + mac + self.tile_overhead_um2

    def energy_per_coefficient_fj(self, spec: TileSpec) -> float:
        return spec.weight_bits * self.rom_fj_per_bit + self.mac_fj_per_coefficient


@dataclass
class DieReport:
    label: str
    tiles: int
    coefficients: int
    utilization: float
    area_mm2: float
    stage_cycles: int
    stage_us: float
    energy_uj_per_token: float
    per_matrix: dict[str, int]

    def as_dict(self) -> dict:
        return asdict(self)


def report_die(label: str, matrices: Sequence[Sequence[MatrixShape]], spec: TileSpec, density: DensityModel
               ) -> DieReport:
    """Tile budget for a die made of the given layers (each a list of matrices)."""
    tiles = 0
    coefficients = 0
    per_matrix: dict[str, int] = {}
    stage_cycles = 0
    for layer in matrices:
        groups = sorted({m.pass_group for m in layer})
        stage_cycles += len(groups) * spec.cycles_per_pass
        for m in layer:
            n = tiles_for(m, spec)
            tiles += n
            coefficients += m.in_features * m.out_features
            per_matrix[m.name] = per_matrix.get(m.name, 0) + n
    area = tiles * density.tile_area_um2(spec) / 1e6
    energy = coefficients * density.energy_per_coefficient_fj(spec) / 1e9
    return DieReport(label, tiles, coefficients, coefficients / (tiles * spec.coefficients), area,
                     stage_cycles, stage_cycles / density.clock_mhz, energy, per_matrix)


def tile_spec_for(cfg, *, rows: int | None = None, **overrides) -> TileSpec:
    """Tile geometry for a model preset: depth equals the hidden width unless overridden.

    The base die's tile depth is fixed at manufacture, so the 4B build (hidden
    2560) is a shallower tile than the 9B (hidden 4096); both keep 64 columns.
    """
    return TileSpec(rows=cfg.hidden_size if rows is None else rows, **overrides)


def report_appliance(cfg, spec: TileSpec, density: DensityModel, num_head_asics: int = 2) -> dict[str, DieReport]:
    shard = [layer_matrices(cfg, i) for i in range(cfg.recurrent_every)]
    head = [head_matrices(cfg, num_head_asics)]
    return {"layer_die": report_die("layer die (R,R,R,G)", shard, spec, density),
            "head_die": report_die("head die (LM head slice)", head, spec, density)}


# ---------------------------------------------------------------------------
# Test-vector emission for the RTL testbench
# ---------------------------------------------------------------------------


def write_hex(path: Path, values: Iterable[int], width_bits: int) -> None:
    digits = math.ceil(width_bits / 4)
    mask = (1 << width_bits) - 1
    path.write_text("".join(f"{int(v) & mask:0{digits}x}\n" for v in values), encoding="utf-8")


def emit_vectors(directory: Path, tile: Tile, x: np.ndarray, spec: TileSpec, psum_in: np.ndarray | None = None) -> dict:
    """Write ROM contents, stimulus, and expected results for ``tb_fabric_tile.sv``."""
    directory.mkdir(parents=True, exist_ok=True)
    words = tile.rom_words(spec)
    # ROM: one line per row, columns packed little-endian (column 0 in the lowest nibble).
    row_words = []
    for r in range(spec.rows):
        word = 0
        for c in range(spec.cols):
            word |= int(words[r, c]) << (c * spec.weight_bits)
        row_words.append(word)
    write_hex(directory / "rom.hex", row_words, spec.cols * spec.weight_bits)
    write_hex(directory / "x.hex", x, spec.act_bits)
    psum0 = np.zeros(spec.cols, dtype=np.int64) if psum_in is None else psum_in
    write_hex(directory / "psum_in.hex", psum0, spec.acc_bits)
    write_hex(directory / "mult.hex", tile.mult, spec.scale_bits)
    write_hex(directory / "shift.hex", tile.shift, spec.shift_bits)
    psum, y = tile_forward(tile, x, spec, psum0)
    write_hex(directory / "expected_psum.hex", psum, spec.acc_bits)
    write_hex(directory / "expected_q.hex", y, spec.act_bits)
    params = {"rows": spec.rows, "cols": spec.cols, "weight_bits": spec.weight_bits, "act_bits": spec.act_bits,
              "rows_per_cycle": spec.rows_per_cycle, "acc_bits": spec.acc_bits, "scale_bits": spec.scale_bits,
              "shift_bits": spec.shift_bits}
    (directory / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", default="qwen3_5_9b")
    parser.add_argument("--rows", type=int, default=None, help="tile depth; defaults to the hidden width")
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--rows-per-cycle", type=int, default=2)
    parser.add_argument("--clock-mhz", type=float, default=800.0)
    args = parser.parse_args()
    from fixed_llm_poc import ASICLMConfig  # local import: torch is optional for the tile model

    cfg = ASICLMConfig.from_preset(args.geometry)
    spec = tile_spec_for(cfg, rows=args.rows, cols=args.cols, rows_per_cycle=args.rows_per_cycle)
    density = DensityModel(clock_mhz=args.clock_mhz)
    for report in report_appliance(cfg, spec, density).values():
        print(f"{report.label}: {report.tiles} tiles, {report.coefficients / 1e6:.0f}M coefficients, "
              f"utilization {report.utilization:.1%}, area {report.area_mm2:.0f} mm2, "
              f"stage {report.stage_cycles} cycles = {report.stage_us:.1f} us, "
              f"{report.energy_uj_per_token:.0f} uJ/token")


if __name__ == "__main__":
    main()
