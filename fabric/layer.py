"""The rest of the layer: fixed-point vector units between the fabric passes.

The fabric tile (``fabric/tile.py``) evaluates ``y = Wx``.  Everything else in
a decoder layer is element-wise or per-head arithmetic on vectors of at most a
few thousand elements: the RMS norms, the causal convolution and its SiLU, the
per-head gates, the Gated DeltaNet state update, the gated norm, SwiGLU, the
residual adds, the rotary embedding and the attention core of a global layer.
This module is the bit-exact reference for those units as the RTL in
``fabric/rtl/fabric_vector.sv``, ``fabric_norm.sv``, ``fabric_recurrent.sv``,
``fabric_ffn.sv`` and ``fabric_attention.sv`` implements them, the compiler
that turns a layer's float weights and a calibration into their constants,
and the float reference the integer layer is checked against.

Number formats
--------------
* residual stream ``h``: int16 at one scale per layer (``s_h``);
* fabric activations: int8 (``TileSpec.act_bits``), one scale per matrix;
* ``F16``: int16 with ``FF`` = 10 fraction bits (Q5.10), the input of every
  nonlinearity and the output of SiLU;
* ``U16``: unsigned Q0.16 for sigmoid, exp and the softmax weights, 1.0
  clipped to 65535;
* normalised elements ``n = x / sqrt(sum x^2) * 2^NF`` in int16;
* recurrent state ``S``: int16 at ``s_v / 256`` (see ``delta_state_int`` for
  why int8 cannot hold a slow decay);
* every unit ends in the tile's requantizer, ``sat((v * mult + 2^(sh-1)) >> sh)``.

Nonlinearities are tables with linear interpolation (``Lut``); the inverse
square root and the reciprocal are a table seed and one Newton step.  All
of it is integer arithmetic on Python ints or int64 arrays, so an RTL run
must match it bit for bit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fabric.tile import (QuantizedMatrix, TileSpec, fixed_point_scale, quantize_matrix, reference_matmul,
                         requantize, write_hex)

FF = 10                       # fraction bits of the F16 format
UB = 16                       # bits of the unit-interval format
ONE_U = (1 << UB) - 1         # 1.0 in U16
NF = 14                       # fraction bits of a normalised element
SHB = 6                       # shift field width of the vector requantizers
MB = 16                       # multiplier width of the vector requantizers


# ---------------------------------------------------------------------------
# Integer helpers
# ---------------------------------------------------------------------------


def sat(v, bits: int):
    limit = 1 << (bits - 1)
    if isinstance(v, np.ndarray):
        return np.clip(v, -limit, limit - 1)
    return max(-limit, min(limit - 1, int(v)))


def rnd_shr(v, sh: int):
    """Arithmetic right shift with round-half-up, the tile's rounding."""
    if isinstance(v, np.ndarray):
        return (v + ((1 << (sh - 1)) if sh > 0 else 0)) >> sh
    return (int(v) + ((1 << (sh - 1)) if sh > 0 else 0)) >> sh


def requant(v, mult: int, shift: int, bits: int):
    """``sat_bits((v * mult + 2^(shift-1)) >> shift)`` for ints or int64 arrays."""
    if isinstance(v, np.ndarray):
        return sat(rnd_shr(v.astype(np.int64) * np.int64(mult), int(shift)), bits)
    return sat(rnd_shr(int(v) * int(mult), int(shift)), bits)


def bit_length(v: int) -> int:
    return int(v).bit_length()


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Lut:
    """A table read with an index of ``ibits`` bits and interpolated over
    ``fbits`` fraction bits (``2^ibits + 1`` entries), or read directly when
    ``fbits`` is None."""

    name: str
    ibits: int
    fbits: int | None
    width: int
    signed: bool
    table: np.ndarray

    def interp(self, u):
        u = np.asarray(u, dtype=np.int64)
        idx = u >> self.fbits
        frac = u & ((1 << self.fbits) - 1)
        t0 = self.table[idx]
        t1 = self.table[idx + 1]
        return t0 + (((t1 - t0) * frac + (1 << (self.fbits - 1))) >> self.fbits)


def _build_luts() -> dict[str, Lut]:
    def interp_table(fn, lo, step, ibits, scale, width, signed):
        n = (1 << ibits) + 1
        values = np.rint(np.array([fn(lo + i * step) for i in range(n)]) * scale).astype(np.int64)
        limit = (1 << (width - 1)) - 1 if signed else (1 << width) - 1
        return np.clip(values, -limit - 1 if signed else 0, limit)

    sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x))
    softplus = lambda x: math.log1p(math.exp(x)) if x < 30 else x
    luts = {
        # sigmoid over [-8, 8): F16 input offset by 8, 256 steps of 1/16
        "sigmoid": Lut("sigmoid", 8, 6, UB, False, interp_table(sigmoid, -8.0, 1 / 16, 8, 1 << UB, UB, False)),
        # exp(-t) over [0, 32): 1024 steps of 1/32
        "exp": Lut("exp", 10, 5, UB, False, interp_table(lambda t: math.exp(-t), 0.0, 1 / 32, 10, 1 << UB, UB, False)),
        # softplus over [-16, 16): 2048 steps of 1/64, F16 output
        "softplus": Lut("softplus", 11, 4, 16, False, interp_table(softplus, -16.0, 1 / 64, 11, 1 << FF, 16, False)),
        # sin of a turn in Q0.16: 1024 steps, Q1.15 output
        "sin": Lut("sin", 10, 6, 16, True, interp_table(lambda x: math.sin(2 * math.pi * x), 0.0, 1 / 1024, 10, 32767, 16, True)),
    }
    # rsqrt seed: M = (256 + i + 0.5) / 1024 in [0.25, 1), Q1.15 result (17 bits)
    rs = np.array([round(32768 / math.sqrt((256 + i + 0.5) / 1024)) for i in range(768)], dtype=np.int64)
    luts["rsqrt"] = Lut("rsqrt", 10, None, 17, False, rs)
    # reciprocal seed: M = (512 + i + 0.5) / 1024 in [0.5, 1), Q1.15 result (17 bits)
    rc = np.array([round(32768 * 1024 / (512 + i + 0.5)) for i in range(512)], dtype=np.int64)
    luts["recip"] = Lut("recip", 9, None, 17, False, rc)
    return luts


LUTS = _build_luts()


def write_luts(directory: Path) -> None:
    """The tables as hex images for ``$readmemh``."""
    directory.mkdir(parents=True, exist_ok=True)
    for lut in LUTS.values():
        write_hex(directory / f"lut_{lut.name}.hex", lut.table, lut.width)


def sigmoid_fixed(t):
    """F16 -> U16."""
    t = np.asarray(t, dtype=np.int64)
    u = np.clip(t + (8 << FF), 0, (16 << FF) - 1)
    return LUTS["sigmoid"].interp(u)


def silu_fixed(t):
    """F16 -> F16: ``t * sigmoid(t)``."""
    t = np.asarray(t, dtype=np.int64)
    return (t * sigmoid_fixed(t) + (1 << (UB - 1))) >> UB


def exp_neg_fixed(t):
    """Unsigned F16-scaled ``t >= 0`` -> U16 ``exp(-t)``; zero from 32 up."""
    t = np.asarray(t, dtype=np.int64)
    inside = t < (32 << FF)
    return np.where(inside, LUTS["exp"].interp(np.minimum(t, (32 << FF) - 1)), 0)


def softplus_fixed(t):
    """F16 -> unsigned F16; the identity from 16 up."""
    t = np.asarray(t, dtype=np.int64)
    u = np.clip(t + (16 << FF), 0, (32 << FF) - 1)
    return np.where(t >= (16 << FF), t, LUTS["softplus"].interp(u))


def sin_cos_fixed(turn):
    """Q0.16 turn -> (sin, cos) in Q1.15."""
    turn = np.asarray(turn, dtype=np.int64) & 0xFFFF
    return LUTS["sin"].interp(turn), LUTS["sin"].interp((turn + 0x4000) & 0xFFFF)


def rsqrt_fixed(ss: int, sw: int) -> tuple[int, int]:
    """``1/sqrt(ss)`` for ``1 <= ss < 2^sw`` (``sw`` even) as ``(R, a)`` with
    ``1/sqrt(ss) = R * 2^(a/2 - 15 - sw/2)``, R in Q1.15 (17 bits)."""
    ss = int(ss)
    if not 1 <= ss < (1 << sw) or sw % 2:
        raise ValueError("rsqrt operand out of range")
    lz = sw - bit_length(ss)
    a = lz & ~1
    m = (ss << a) >> (sw - 16)                     # [2^14, 2^16)
    r0 = int(LUTS["rsqrt"].table[(m >> 6) - 256])
    t = (m * r0 * r0) >> 16                        # M r0^2 in Q2.30
    u = (3 << 30) - t
    r1 = (r0 * u) >> 31                            # r0 (3 - M r0^2) / 2
    return r1, a


def recip_fixed(l: int, lw: int) -> tuple[int, int]:
    """``1/l`` for ``1 <= l < 2^lw`` as ``(R, lz)`` with ``1/l = R * 2^(lz - 15 - lw)``."""
    l = int(l)
    if not 1 <= l < (1 << lw):
        raise ValueError("reciprocal operand out of range")
    lz = lw - bit_length(l)
    m = ((l << lz) >> (lw - 16)) if lw >= 16 else ((l << lz) << (16 - lw))    # [2^15, 2^16)
    r0 = int(LUTS["recip"].table[(m >> 6) - 512])
    t = (m * r0) >> 16                             # M r0 in Q1.15
    u = (2 << 15) - t
    r1 = (r0 * u) >> 15                            # r0 (2 - M r0)
    return r1, lz


# ---------------------------------------------------------------------------
# The units
# ---------------------------------------------------------------------------


def sw_for(xw: int, d: int) -> int:
    """Width of the sum of ``d`` squares of ``xw``-bit values, rounded up to even."""
    bits = 2 * xw - 1 + math.ceil(math.log2(d))
    return bits + (bits & 1)


def rmsnorm_int(x: np.ndarray, gain, mult: int, shift: int, *, xw: int, ow: int, eps_int: int = 0) -> np.ndarray:
    """``sat_ow((n * gain * mult) >> shift)`` with ``n = x / sqrt(sum x^2 + eps) * 2^NF``.

    RMSNorm's ``sqrt(D)`` and the L2 norm's absence of it are both in
    ``mult``; a per-element weight that cannot be folded into the following
    matrix rides in ``gain`` (int16), as does the gate of the gated norm.
    """
    x = np.asarray(x, dtype=np.int64)
    d = x.shape[0]
    sw = sw_for(xw, d)
    ss = max(int((x * x).sum()) + int(eps_int), 1)
    r, a = rsqrt_fixed(ss, sw)
    sh = 1 + sw // 2 - a // 2
    n = sat(rnd_shr(x * r, sh), 16)
    g = np.asarray(gain, dtype=np.int64) if np.ndim(gain) else np.full(d, int(gain), dtype=np.int64)
    return requant(n * g, mult, shift, ow)


def conv_silu_int(hist: np.ndarray, x_new: np.ndarray, w: np.ndarray, mult_in, sh_in, mult_out, sh_out
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Depthwise causal convolution of ``kernel`` taps and SiLU, per channel.

    ``hist[c, j]`` holds the ``kernel - 1`` previous inputs oldest first and
    ``x_new[c]`` the current one; the taps ``w[c, j]`` are int8.  Returns
    the int8 outputs and the shifted history.  The constants are per channel.
    """
    hist = np.asarray(hist, dtype=np.int64)
    x_new = np.asarray(x_new, dtype=np.int64)
    w = np.asarray(w, dtype=np.int64)
    window = np.concatenate([hist, x_new[:, None]], axis=1)
    acc = (window * w).sum(axis=1)
    t = sat(_requant_vec(acc, mult_in, sh_in), 16)
    s = silu_fixed(t)
    y = sat(_requant_vec(s, mult_out, sh_out), 8)
    return y, window[:, 1:]


def _requant_vec(v: np.ndarray, mult, sh) -> np.ndarray:
    """Per-element ``(v * mult + 2^(sh-1)) >> sh`` with array ``mult`` and ``sh``."""
    v = np.asarray(v, dtype=np.int64)
    mult = np.broadcast_to(np.asarray(mult, dtype=np.int64), v.shape)
    sh = np.broadcast_to(np.asarray(sh, dtype=np.int64), v.shape)
    rounding = np.where(sh > 0, np.int64(1) << np.maximum(sh - 1, 0), 0)
    return (v * mult + rounding) >> sh


def head_gates_int(a_acc, b_acc, mult_a, sh_a, mult_b, sh_b, a_coef, dt_bias) -> tuple[np.ndarray, np.ndarray]:
    """Per head: ``beta = sigmoid(b)`` and ``decay = exp(-A softplus(a + dt_bias))``.

    ``a_acc`` and ``b_acc`` are the fabric's raw accumulators for the one-column
    ``in_proj_a`` and ``in_proj_b`` outputs; ``a_coef`` is ``exp(A_log)`` in
    Q6.10 and ``dt_bias`` in F16.  Both results are U16.
    """
    ta = sat(_requant_vec(a_acc, mult_a, sh_a), 16)
    ta = sat(ta + np.asarray(dt_bias, dtype=np.int64), 16)
    sp = softplus_fixed(ta)
    t = (np.asarray(a_coef, dtype=np.int64) * sp + (1 << (FF - 1))) >> FF
    decay = exp_neg_fixed(t)
    tb = sat(_requant_vec(b_acc, mult_b, sh_b), 16)
    beta = sigmoid_fixed(tb)
    return decay, beta


def ysh_for(k: int) -> int:
    """Shift that brings ``q . S`` (unit int8 ``q``, int16 ``S``) into int16."""
    return math.ceil(7 + math.log2(k) / 2)


def delta_state_int(s: np.ndarray, k: np.ndarray, v: np.ndarray, q: np.ndarray, decay: int, beta: int,
                    ysh: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One token of the delta rule on one head.

    ``s`` is int16 ``[K, V]`` at scale ``s_v / 256``; ``k`` and ``q`` are int8
    unit vectors (scale ``2^-7``); ``v`` int8 at ``s_v``; ``decay`` and
    ``beta`` U16.  Returns the new state and ``y = q . S'`` shifted into int16.

    The state is int16 rather than the int8 the simulator's traffic figure
    assumed because round-to-nearest cannot apply a slow decay to a narrow
    value: ``S * d`` rounds back to ``S`` whenever ``|S| < 1 / (1 - d)``, which
    at ``d = 0.999`` is every int8 value and the bottom three percent of int16.
    """
    s = np.asarray(s, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    q = np.asarray(q, dtype=np.int64)
    kk, vv = s.shape
    ysh = ysh_for(kk) if ysh is None else ysh
    sd = rnd_shr(s * int(decay), UB)                                   # int16, |sd| <= |s|
    pred = rnd_shr((k[:, None] * sd).sum(axis=0), 15)                  # at the scale of v
    diff = sat(v - pred, 16)
    delta = rnd_shr(int(beta) * k[:, None] * diff[None, :], 15)
    s_new = sat(sd + delta, 16)
    y = sat(rnd_shr((q[:, None] * s_new).sum(axis=0), ysh), 16)
    return s_new, y


HALF_U = 1 << (UB - 1)        # the scale below which the int8 state is rescaled
E_MIN, E_MAX = -4, 6          # the exponent of the int8 state's LSB, s_v * 2^-e
PEAK_GROW = 47                # a head whose peak since its last rescale is at most this doubles its resolution
SAT_SHIFT = 6                 # a head with more than K*V / 2^SAT_SHIFT saturated elements halves its resolution at once


def state_recip(g: int) -> int:
    """``floor(2^31 / g)`` for a scale in ``[2^15, 2^16)``: ``1 / g`` in Q1.15, 32768 to 65536."""
    return (1 << 31) // int(g)


def state_rescale(g: int, decay: int, e: int, peak: int, nsat: int, elements: int) -> tuple[int, bool, int]:
    """The head's new scale and whether this token rescales the stored state:
    ``(g1, rescale, de)``.  The scale takes the decay; the state is rescaled
    by ``g1 * 2^de`` (and the scale returns to one) when the scale has
    fallen below one half or more than ``elements / 2^SAT_SHIFT`` of the
    state saturated last token.  ``de`` is +1 when the largest value written
    since the last rescale was at most ``PEAK_GROW`` (the byte never more
    than a third used, so the resolution doubles), -1 to back off from
    saturation, else 0."""
    g1 = rnd_shr(int(g) * int(decay), UB)
    saturated = nsat > (elements >> SAT_SHIFT)
    if not (g1 < HALF_U or saturated):
        return g1, False, 0
    if saturated and e > E_MIN:
        de = -1
    elif not saturated and peak <= PEAK_GROW and e < E_MAX:
        de = 1
    else:
        de = 0
    return g1, True, de


def delta_state_int8(t: np.ndarray, k: np.ndarray, v: np.ndarray, q: np.ndarray, decay: int, beta: int,
                     g: int, e: int, peak: int, nsat: int, ysh: int | None = None
                     ) -> tuple[np.ndarray, int, int, int, int, np.ndarray]:
    """One token of the delta rule on one head whose state is int8 with a scale.

    The state is ``S = g * T * 2^-e``: ``T`` int8 ``[K, V]``, ``g`` a U16 scale
    and ``e`` a small exponent per head (``e = 0`` puts the LSB at ``s_v``,
    the top byte of the int16 state's range), with ``peak`` the largest
    ``|T|`` written since the last rescale and ``nsat`` the number of
    saturated elements written last token.  The decay multiplies the scale,
    not the stored values, so it costs no rounding; when the scale falls
    below one half (or enough of the state saturated) the head is rescaled
    once, the only rounding the state sees, and the exponent moves to keep
    the byte in use.  The rank-one update is added at ``1 / g``, and
    ``y = q . S'`` is shifted into int16 exactly as ``delta_state_int``
    does.  Returns the new state, scale, exponent, peak, saturated count
    and ``y``.
    """
    t = np.asarray(t, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    q = np.asarray(q, dtype=np.int64)
    kk, vv = t.shape
    ysh = ysh_for(kk) if ysh is None else ysh
    g1, rescale, de = state_rescale(g, decay, e, peak, nsat, kk * vv)
    if rescale:
        t = sat(rnd_shr(t * g1, UB - de), 8)                            # times g1 * 2^de
        g1, e = ONE_U, e + de
    r = state_recip(g1)
    pred = rnd_shr((k[:, None] * t).sum(axis=0) * g1, 7 + UB + e)      # at the scale of v
    diff = sat(v - pred, 16)
    c = rnd_shr(int(beta) * diff * r, 24 - e)                           # beta * diff / g in LSBs, k's 2^-7 still to come
    t_new = sat(t + rnd_shr(k[:, None] * c[None, :], 14), 8)
    y = sat(rnd_shr((q[:, None] * t_new).sum(axis=0) * g1, ysh + 8 + e), 16)
    peak_new = max(0 if rescale else int(peak), int(np.abs(t_new).max()))    # the largest |T| since the last rescale
    return t_new, int(g1), int(e), peak_new, int((np.abs(t_new) >= 127).sum()), y


def swiglu_int(g: np.ndarray, u: np.ndarray, mult_g: int, sh_g: int, mult_o: int, sh_o: int) -> np.ndarray:
    """``silu(g) * u`` on int8 fabric outputs, back to int8."""
    tg = sat(requant(np.asarray(g, dtype=np.int64), mult_g, sh_g, 16), 16)
    p = silu_fixed(tg) * np.asarray(u, dtype=np.int64)
    return requant(p, mult_o, sh_o, 8)


def residual_int(h: np.ndarray, y: np.ndarray, mult: int, shift: int) -> np.ndarray:
    """``h + y`` with ``y`` rescaled from its int8 scale to the residual's."""
    h = np.asarray(h, dtype=np.int64)
    return sat(h + rnd_shr(np.asarray(y, dtype=np.int64) * int(mult), int(shift)), 16)


def rotary_table_int(pos: int, inv_freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(sin, cos)`` in Q1.15 for one position: ``pos * inv_freq`` in Q0.32
    turns, the fraction kept, its top 16 bits looked up."""
    angle = (int(pos) * np.asarray(inv_freq, dtype=np.int64)) & 0xFFFFFFFF
    return sin_cos_fixed(angle >> 16)


def rotary_int(x: np.ndarray, sin: np.ndarray, cos: np.ndarray, r: int, mult: int, shift: int) -> np.ndarray:
    """Rotate pairs ``(i, i + r/2)`` of the first ``r`` elements of an int16
    head vector, then requantize every element to int8."""
    x = np.asarray(x, dtype=np.int64).copy()
    half = r // 2
    x1, x2 = x[:half], x[half:r]
    sin = np.asarray(sin, dtype=np.int64)
    cos = np.asarray(cos, dtype=np.int64)
    y1 = sat(rnd_shr(x1 * cos - x2 * sin, 15), 16)
    y2 = sat(rnd_shr(x2 * cos + x1 * sin, 15), 16)
    x[:half], x[half:r] = y1, y2
    return requant(x, mult, shift, 8)


def attention_int(q: np.ndarray, gate: np.ndarray, k_rows: np.ndarray, v_rows: np.ndarray, *, mult_s: int, sh_s: int,
                  mult_gate: int, sh_gate: int, mult_o: int, sh_o: int, lw: int = 28) -> np.ndarray:
    """Online-softmax attention of ``G`` int8 queries over one stream of int8
    key and value rows, gated and requantized to int8.

    Scores are ``(q . k) * mult_s >> sh_s`` in F16 units without saturation;
    the running maximum ``m``, the weight ``p = exp(m - s)`` and the rescale
    factor of the accumulators are U16; ``o`` accumulates ``p * v`` and ``l``
    accumulates ``p``.  At the end ``w = o / l * 2^8`` (int16) and the output
    is ``sat8(w * sigmoid(gate) * mult_o >> sh_o)``.
    """
    q = np.asarray(q, dtype=np.int64)
    gate = np.asarray(gate, dtype=np.int64)
    k_rows = np.asarray(k_rows, dtype=np.int64)
    v_rows = np.asarray(v_rows, dtype=np.int64)
    g_heads, hd = q.shape
    out = np.zeros_like(q)
    for g in range(g_heads):
        m = None
        l = 0
        o = np.zeros(hd, dtype=np.int64)
        for k_row, v_row in zip(k_rows, v_rows):
            s = rnd_shr(int((q[g] * k_row).sum()) * mult_s, sh_s)
            if m is None or s > m:
                f = ONE_U if m is None else int(exp_neg_fixed(min(s - m, (32 << FF))))
                m = s
                p = ONE_U
            else:
                f = ONE_U
                p = int(exp_neg_fixed(min(m - s, (32 << FF))))
            l = rnd_shr(l * f, UB) + p
            o = rnd_shr(o * f, UB) + p * v_row
        if not 1 <= l < (1 << lw):
            raise ValueError("softmax denominator out of range")
        r, lz = recip_fixed(l, lw)
        w = sat(rnd_shr(o * r, 7 + lw - lz), 16)
        sig = sigmoid_fixed(requant(gate[g], mult_gate, sh_gate, 16))
        out[g] = requant(w * sig, mult_o, sh_o, 8)
    return out


# ---------------------------------------------------------------------------
# Float reference of one token through a layer (numpy, float64)
# ---------------------------------------------------------------------------


def _rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * (1.0 + w)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def recurrent_layer_float(w: dict, cfg, x: np.ndarray, s: np.ndarray, hist: np.ndarray) -> dict:
    """One token through a Gated DeltaNet layer, all intermediates returned.

    ``w`` maps Hugging Face parameter names (without the layer prefix) to
    float arrays in the PyTorch layout; ``s`` is ``[v_heads, K, V]`` and
    ``hist`` ``[conv_dim, kernel - 1]``.
    """
    eps = cfg.rms_eps
    nk, nv, hk, hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    kd, vd = nk * hk, nv * hv
    r: dict = {}
    r["h"] = _rmsnorm(x, w["input_layernorm.weight"], eps)
    r["qkv"] = r["h"] @ w["linear_attn.in_proj_qkv.weight"].T
    r["z"] = r["h"] @ w["linear_attn.in_proj_z.weight"].T
    r["b"] = r["h"] @ w["linear_attn.in_proj_b.weight"].T
    r["a"] = r["h"] @ w["linear_attn.in_proj_a.weight"].T
    window = np.concatenate([hist, r["qkv"][:, None]], axis=1)
    r["conv"] = (window * w["linear_attn.conv1d.weight"][:, 0, :]).sum(axis=1)
    r["hist_next"] = window[:, 1:]
    mixed = _silu(r["conv"])
    q, k, v = mixed[:kd], mixed[kd:2 * kd], mixed[2 * kd:]
    qh = q.reshape(nk, hk)
    kh = k.reshape(nk, hk)
    qh = qh / np.linalg.norm(qh, axis=-1, keepdims=True)
    kh = kh / np.linalg.norm(kh, axis=-1, keepdims=True)
    repeat = nv // nk
    r["q_unit"], r["k_unit"] = np.repeat(qh, repeat, axis=0), np.repeat(kh, repeat, axis=0)
    r["v"] = v.reshape(nv, hv)
    qs = r["q_unit"] * hk ** -0.5
    r["beta"] = 1.0 / (1.0 + np.exp(-r["b"]))
    r["decay"] = np.exp(-np.exp(w["linear_attn.A_log"]) * _softplus(r["a"] + w["linear_attn.dt_bias"]))
    s = s * r["decay"][:, None, None]
    pred = np.einsum("hk,hkv->hv", r["k_unit"], s)
    s = s + r["beta"][:, None, None] * r["k_unit"][:, :, None] * (r["v"] - pred)[:, None, :]
    r["s_next"] = s
    r["y"] = np.einsum("hk,hkv->hv", qs, s)
    zh = r["z"].reshape(nv, hv)
    yn = r["y"] / np.sqrt((r["y"] ** 2).mean(axis=-1, keepdims=True) + eps)
    r["y_norm"] = (w["linear_attn.norm.weight"] * yn * _silu(zh)).reshape(vd)
    r["mixer"] = r["y_norm"] @ w["linear_attn.out_proj.weight"].T
    r["x1"] = x + r["mixer"]
    _ffn(w, cfg, r)
    return r


def _ffn(w: dict, cfg, r: dict) -> None:
    r["h2"] = _rmsnorm(r["x1"], w["post_attention_layernorm.weight"], cfg.rms_eps)
    r["gate"] = r["h2"] @ w["mlp.gate_proj.weight"].T
    r["up"] = r["h2"] @ w["mlp.up_proj.weight"].T
    r["act"] = _silu(r["gate"]) * r["up"]
    r["ffn"] = r["act"] @ w["mlp.down_proj.weight"].T
    r["x2"] = r["x1"] + r["ffn"]


def rope_float(x: np.ndarray, pos: int, cfg) -> np.ndarray:
    """Rotate the first ``rotary_dim`` elements of every head of ``x [heads, hd]``."""
    rd = cfg.rotary_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (np.arange(0, rd, 2, dtype=np.float64) / rd))
    ang = pos * inv_freq
    cos, sin = np.cos(ang), np.sin(ang)
    x = x.copy()
    x1, x2 = x[:, :rd // 2].copy(), x[:, rd // 2:rd].copy()
    x[:, :rd // 2] = x1 * cos - x2 * sin
    x[:, rd // 2:rd] = x2 * cos + x1 * sin
    return x


def global_layer_float(w: dict, cfg, x: np.ndarray, pos: int, k_rows: np.ndarray, v_rows: np.ndarray) -> dict:
    """One token through a gated-attention layer whose key and value rows
    (``[kv_heads, N, hd]``, this token's own included) the memory system has
    already produced: exactly the arithmetic the datapath performs."""
    eps = cfg.rms_eps
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    group = nh // nkv
    r: dict = {}
    r["h"] = _rmsnorm(x, w["input_layernorm.weight"], eps)
    qg = (r["h"] @ w["self_attn.q_proj.weight"].T).reshape(nh, 2 * hd)
    r["q_raw"], r["gate"] = qg[:, :hd], qg[:, hd:]
    r["k_raw"] = (r["h"] @ w["self_attn.k_proj.weight"].T).reshape(nkv, hd)
    r["v"] = (r["h"] @ w["self_attn.v_proj.weight"].T).reshape(nkv, hd)
    r["index_q"] = r["h"] @ w["self_attn.index_q.weight"].T
    r["index_k"] = r["h"] @ w["self_attn.index_k.weight"].T
    r["q"] = rope_float(_rmsnorm(r["q_raw"], w["self_attn.q_norm.weight"], eps), pos, cfg)
    r["k"] = rope_float(_rmsnorm(r["k_raw"], w["self_attn.k_norm.weight"], eps), pos, cfg)
    out = np.zeros((nh, hd))
    for n in range(nkv):
        for g in range(group):
            h = n * group + g
            scores = k_rows[n] @ r["q"][h] * hd ** -0.5
            p = np.exp(scores - scores.max())
            p /= p.sum()
            out[h] = p @ v_rows[n]
    r["att"] = (out * (1.0 / (1.0 + np.exp(-r["gate"])))).reshape(nh * hd)
    r["mixer"] = r["att"] @ w["self_attn.o_proj.weight"].T
    r["x1"] = x + r["mixer"]
    _ffn(w, cfg, r)
    return r


# ---------------------------------------------------------------------------
# Compilation: float weights and a calibration -> constants
# ---------------------------------------------------------------------------


def calibrate(intermediates: list[dict], keys: list[str]) -> dict[str, float]:
    """Absolute maxima over a set of float runs, the static calibration."""
    return {key: max(float(np.abs(r[key]).max()) for r in intermediates) for key in keys}


def _scale(absmax: float, bits: int) -> float:
    return max(absmax, 1e-12) / ((1 << (bits - 1)) - 1)


def _fp(value: float) -> tuple[int, int]:
    """``value`` as ``mult / 2^shift`` in the vector requantizer's widths."""
    mult, shift = fixed_point_scale(np.array([value]), TileSpec(scale_bits=MB, shift_bits=SHB))
    return int(mult[0]), int(shift[0])


@dataclass
class Norm:
    gain: np.ndarray | int | None
    mult: int
    shift: int
    xw: int
    ow: int
    eps_int: int = 0                 # rms_eps in the input's integer units, times D


@dataclass
class FfnConsts:
    norm: Norm
    gate_proj: QuantizedMatrix
    up_proj: QuantizedMatrix
    down_proj: QuantizedMatrix
    mult_g: int
    sh_g: int
    mult_o: int
    sh_o: int
    res_mult: int
    res_shift: int


@dataclass
class RecurrentConsts:
    s_h: float                       # residual LSB
    norm: Norm
    in_proj_qkv: QuantizedMatrix
    in_proj_z: QuantizedMatrix
    in_proj_b: QuantizedMatrix
    in_proj_a: QuantizedMatrix
    conv_w: np.ndarray               # int8 [C, kernel]
    conv_mult_in: np.ndarray
    conv_sh_in: np.ndarray
    conv_mult_out: np.ndarray
    conv_sh_out: np.ndarray
    unit_norm: Norm                  # q and k to int8 unit vectors
    gate_mult_a: np.ndarray
    gate_sh_a: np.ndarray
    gate_mult_b: np.ndarray
    gate_sh_b: np.ndarray
    a_coef: np.ndarray               # Q6.10
    dt_bias: np.ndarray              # F16
    s_v: float
    z_mult: int
    z_shift: int
    gated_norm: Norm
    out_proj: QuantizedMatrix
    res_mult: int
    res_shift: int
    ffn: FfnConsts


@dataclass
class GlobalConsts:
    s_h: float
    norm: Norm
    q_proj: QuantizedMatrix
    k_proj: QuantizedMatrix
    v_proj: QuantizedMatrix
    q_norm: Norm
    k_norm: Norm
    inv_freq: np.ndarray             # Q0.32 turns per position
    rot_mult_q: int
    rot_sh_q: int
    rot_mult_k: int
    rot_sh_k: int
    s_q: float
    s_k: float
    s_v: float
    mult_s: int
    sh_s: int
    mult_gate: int
    sh_gate: int
    mult_o: int
    sh_o: int
    o_proj: QuantizedMatrix
    res_mult: int
    res_shift: int
    ffn: FfnConsts
    index_q: QuantizedMatrix | None = None
    index_k: QuantizedMatrix | None = None
    unit_norm: Norm | None = None       # the index query to an int8 unit vector


def _matrix(w: np.ndarray, spec: TileSpec, act_scale: float, out_scale: float, row_gain: np.ndarray | None = None
            ) -> QuantizedMatrix:
    """A PyTorch ``[out, in]`` weight, an optional per-input gain folded into it, as tiles' ``[in, out]``."""
    m = w.T.astype(np.float64)
    if row_gain is not None:
        m = m * row_gain[:, None]
    return quantize_matrix(m, spec, act_scale=act_scale, out_scale=out_scale)


def _eps_int(eps: float, d: int, s_in: float) -> int:
    """``mean(x^2) + eps`` as ``sum(x_int^2) + eps_int``: ``eps_int = d * eps / s_in^2``."""
    return int(round(d * eps / (s_in * s_in)))


def _norm_consts(absmax_out: float, d: int, xw: int, ow: int, gain=1, gain_bits: int = 0, extra: float = 1.0,
                 eps_int: int = 0) -> Norm:
    """Constants for ``rmsnorm_int``: the output LSB is ``absmax_out / max``;
    one normalised LSB is ``sqrt(d) * extra / 2^NF`` of real value, and the
    gain carries ``gain_bits`` fraction bits."""
    s_out = _scale(absmax_out, ow)
    mult, shift = _fp(math.sqrt(d) * extra / (1 << (NF + gain_bits)) / s_out)
    return Norm(gain, mult, shift, xw, ow, eps_int)


def _compile_ffn(w: dict, cfg, spec: TileSpec, cal: dict, s_h: float) -> FfnConsts:
    d = cfg.hidden_size
    norm = _norm_consts(cal["h2"], d, 16, 8, eps_int=_eps_int(cfg.rms_eps, d, s_h))
    s_n = _scale(cal["h2"], 8)
    s_g, s_u, s_act, s_ffn = (_scale(cal[k], 8) for k in ("gate", "up", "act", "ffn"))
    gain = 1.0 + w["post_attention_layernorm.weight"]
    mult_g, sh_g = _fp(s_g * (1 << FF))
    mult_o, sh_o = _fp(s_u / (1 << FF) / s_act)
    res_mult, res_shift = _fp(s_ffn / s_h)
    return FfnConsts(norm, _matrix(w["mlp.gate_proj.weight"], spec, s_n, s_g, gain),
                     _matrix(w["mlp.up_proj.weight"], spec, s_n, s_u, gain),
                     _matrix(w["mlp.down_proj.weight"], spec, s_act, s_ffn),
                     mult_g, sh_g, mult_o, sh_o, res_mult, res_shift)


RECURRENT_CAL_KEYS = ["x2", "h", "qkv", "z", "conv", "v", "y_norm", "mixer", "h2", "gate", "up", "act", "ffn"]
GLOBAL_CAL_KEYS = ["x2", "h", "q_raw", "k_raw", "v", "q", "k", "att", "mixer", "h2", "gate", "up", "act", "ffn",
                   "index_q", "index_k"]


def compile_recurrent_layer(w: dict, cfg, spec: TileSpec, cal: dict[str, float]) -> RecurrentConsts:
    d = cfg.hidden_size
    nk, nv, hk, hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    kd = nk * hk
    s_h = _scale(cal["x2"], 16)
    norm = _norm_consts(cal["h"], d, 16, 8, eps_int=_eps_int(cfg.rms_eps, d, s_h))
    s_n = _scale(cal["h"], 8)
    gain = 1.0 + w["input_layernorm.weight"]
    s_qkv = _scale(cal["qkv"], 8)
    s_z = _scale(cal["z"], 8)
    qkv = _matrix(w["linear_attn.in_proj_qkv.weight"], spec, s_n, s_qkv, gain)
    z = _matrix(w["linear_attn.in_proj_z.weight"], spec, s_n, s_z, gain)
    # The one-column heads use the raw accumulator; the out scale only sets the unused requantizer.
    b = _matrix(w["linear_attn.in_proj_b.weight"], spec, s_n, 1.0, gain)
    a = _matrix(w["linear_attn.in_proj_a.weight"], spec, s_n, 1.0, gain)
    conv = w["linear_attn.conv1d.weight"][:, 0, :]
    s_w = _scale(float(np.abs(conv).max()), 8)
    conv_w = sat(np.rint(conv / s_w), 8).astype(np.int8)
    c = conv.shape[0]
    mult_in, sh_in = _fp(s_qkv * s_w * (1 << FF))
    s_qk = _scale(max(cal["conv"], 1e-6), 8)      # q and k are normalised next; only the range matters
    s_v = _scale(cal["v"], 8)
    mult_qk, sh_qk = _fp(1.0 / (1 << FF) / s_qk)
    mult_v, sh_v = _fp(1.0 / (1 << FF) / s_v)
    conv_mult_out = np.array([mult_qk] * (2 * kd) + [mult_v] * (c - 2 * kd), dtype=np.int64)
    conv_sh_out = np.array([sh_qk] * (2 * kd) + [sh_v] * (c - 2 * kd), dtype=np.int64)
    unit_norm = Norm(1, 1, NF - 7, 8, 8)           # n / 2^7: a unit vector at 2^-7
    # Head gates from the raw accumulators: real = acc * s_n * column scale.
    gate_mult_a, gate_sh_a = zip(*[_fp(s_n * float(a.scale[h]) * (1 << FF)) for h in range(nv)])
    gate_mult_b, gate_sh_b = zip(*[_fp(s_n * float(b.scale[h]) * (1 << FF)) for h in range(nv)])
    a_coef = np.clip(np.rint(np.exp(w["linear_attn.A_log"]) * (1 << FF)), 0, 65535).astype(np.int64)
    dt_bias = sat(np.rint(w["linear_attn.dt_bias"] * (1 << FF)), 16).astype(np.int64)
    z_mult, z_shift = _fp(s_z * (1 << FF))
    s_o = _scale(cal["y_norm"], 8)
    s_y = 2.0 ** (ysh_for(hk) - 15) * s_v
    gated = _norm_consts(cal["y_norm"], hv, 16, 8, gain=None, gain_bits=FF, eps_int=_eps_int(cfg.rms_eps, hv, s_y))
    out_proj = _matrix(w["linear_attn.out_proj.weight"], spec, s_o, _scale(cal["mixer"], 8),
                       np.tile(w["linear_attn.norm.weight"], nv))
    res_mult, res_shift = _fp(_scale(cal["mixer"], 8) / s_h)
    return RecurrentConsts(s_h, norm, qkv, z, b, a, conv_w, np.full(c, mult_in), np.full(c, sh_in), conv_mult_out,
                           conv_sh_out, unit_norm, np.array(gate_mult_a), np.array(gate_sh_a), np.array(gate_mult_b),
                           np.array(gate_sh_b), a_coef, dt_bias, s_v, z_mult, z_shift, gated, out_proj, res_mult,
                           res_shift, _compile_ffn(w, cfg, spec, cal, s_h))


def compile_global_layer(w: dict, cfg, spec: TileSpec, cal: dict[str, float]) -> GlobalConsts:
    d, hd = cfg.hidden_size, cfg.head_dim
    rd = cfg.rotary_dim
    s_h = _scale(cal["x2"], 16)
    norm = _norm_consts(cal["h"], d, 16, 8, eps_int=_eps_int(cfg.rms_eps, d, s_h))
    s_n = _scale(cal["h"], 8)
    gain = 1.0 + w["input_layernorm.weight"]
    s_qraw = _scale(max(cal["q_raw"], cal["gate"] if "gate" in cal else 0.0), 8)
    s_kraw, s_v = _scale(cal["k_raw"], 8), _scale(cal["v"], 8)
    q_proj = _matrix(w["self_attn.q_proj.weight"], spec, s_n, s_qraw, gain)
    k_proj = _matrix(w["self_attn.k_proj.weight"], spec, s_n, s_kraw, gain)
    v_proj = _matrix(w["self_attn.v_proj.weight"], spec, s_n, s_v, gain)
    # Head norms keep their weight as a gain in Q3.13; the 16-bit output holds n * gain / 2.
    q_gain = sat(np.rint((1.0 + w["self_attn.q_norm.weight"]) * (1 << 13)), 16).astype(np.int64)
    k_gain = sat(np.rint((1.0 + w["self_attn.k_norm.weight"]) * (1 << 13)), 16).astype(np.int64)
    q_norm = Norm(q_gain, 1, 14, 8, 16, _eps_int(cfg.rms_eps, hd, s_qraw))
    k_norm = Norm(k_gain, 1, 14, 8, 16, _eps_int(cfg.rms_eps, hd, s_kraw))
    inv_freq = 1.0 / (cfg.rope_theta ** (np.arange(0, rd, 2, dtype=np.float64) / rd))
    inv_freq_q32 = np.rint(inv_freq / (2 * math.pi) * (1 << 32)).astype(np.int64) & 0xFFFFFFFF
    # One 16-bit normed LSB is sqrt(hd) / 2^NF * 2 of real value (the gain's Q3.13 over the shift of 14).
    s_q, s_k = _scale(cal["q"], 8), _scale(cal["k"], 8)
    per_lsb = math.sqrt(hd) / (1 << NF) * 2
    rot_mult_q, rot_sh_q = _fp(per_lsb / s_q)
    rot_mult_k, rot_sh_k = _fp(per_lsb / s_k)
    mult_s, sh_s = _fp(s_q * s_k * hd ** -0.5 * (1 << FF))
    mult_gate, sh_gate = _fp(s_qraw * (1 << FF))
    s_att = _scale(cal["att"], 8)
    mult_o, sh_o = _fp(s_v / 256 / (1 << UB) / s_att)      # w is v / 2^8, the gate U16
    o_proj = _matrix(w["self_attn.o_proj.weight"], spec, s_att, _scale(cal["mixer"], 8))
    res_mult, res_shift = _fp(_scale(cal["mixer"], 8) / s_h)
    index_q = _matrix(w["self_attn.index_q.weight"], spec, s_n, _scale(cal["index_q"], 8), gain)
    index_k = _matrix(w["self_attn.index_k.weight"], spec, s_n, _scale(cal["index_k"], 8), gain)
    return GlobalConsts(s_h, norm, q_proj, k_proj, v_proj, q_norm, k_norm, inv_freq_q32, rot_mult_q, rot_sh_q,
                        rot_mult_k, rot_sh_k, s_q, s_k, s_v, mult_s, sh_s, mult_gate, sh_gate, mult_o, sh_o, o_proj,
                        res_mult, res_shift, _compile_ffn(w, cfg, spec, cal, s_h), index_q, index_k,
                        Norm(1, 1, NF - 7, 8, 8))


def dequantized_weights(w: dict, consts, cfg) -> dict:
    """The float weights the integer layer actually carries: every fabric
    matrix at its int4 values times its column scales, the folded norm
    weights removed from the norms, the conv taps at int8.  A float run on
    these isolates the datapath's own arithmetic from weight quantisation."""
    out = dict(w)

    def deq(q: QuantizedMatrix) -> np.ndarray:
        return (q.weights.astype(np.float64) * q.scale[None, :]).T

    if isinstance(consts, RecurrentConsts):
        out["input_layernorm.weight"] = np.zeros_like(w["input_layernorm.weight"])
        out["linear_attn.in_proj_qkv.weight"] = deq(consts.in_proj_qkv)
        out["linear_attn.in_proj_z.weight"] = deq(consts.in_proj_z)
        out["linear_attn.in_proj_b.weight"] = deq(consts.in_proj_b)
        out["linear_attn.in_proj_a.weight"] = deq(consts.in_proj_a)
        s_w = float(np.abs(w["linear_attn.conv1d.weight"]).max()) / 127
        out["linear_attn.conv1d.weight"] = (consts.conv_w.astype(np.float64) * s_w)[:, None, :]
        out["linear_attn.norm.weight"] = np.ones_like(w["linear_attn.norm.weight"])
        out["linear_attn.out_proj.weight"] = deq(consts.out_proj)
    else:
        out["input_layernorm.weight"] = np.zeros_like(w["input_layernorm.weight"])
        out["self_attn.q_proj.weight"] = deq(consts.q_proj)
        out["self_attn.k_proj.weight"] = deq(consts.k_proj)
        out["self_attn.v_proj.weight"] = deq(consts.v_proj)
        out["self_attn.o_proj.weight"] = deq(consts.o_proj)
        out["self_attn.index_q.weight"] = deq(consts.index_q)
        out["self_attn.index_k.weight"] = deq(consts.index_k)
    out["post_attention_layernorm.weight"] = np.zeros_like(w["post_attention_layernorm.weight"])
    out["mlp.gate_proj.weight"] = deq(consts.ffn.gate_proj)
    out["mlp.up_proj.weight"] = deq(consts.ffn.up_proj)
    out["mlp.down_proj.weight"] = deq(consts.ffn.down_proj)
    return out


# ---------------------------------------------------------------------------
# Integer layer forward
# ---------------------------------------------------------------------------


def _fabric(q: QuantizedMatrix, x: np.ndarray, spec: TileSpec) -> tuple[np.ndarray, np.ndarray]:
    """A fabric pass: raw accumulators and requantized int8 (``tile_forward`` proves the tiles equal this)."""
    acc = reference_matmul(q, x)
    return acc, requantize(acc, q.mult, q.shift, spec)


def _norm(x: np.ndarray, n: Norm, gain=None) -> np.ndarray:
    return rmsnorm_int(x, n.gain if gain is None else gain, n.mult, n.shift, xw=n.xw, ow=n.ow, eps_int=n.eps_int)


def _ffn_int(c: FfnConsts, spec: TileSpec, x1: np.ndarray) -> dict:
    r = {"h2": _norm(x1, c.norm)}
    _, r["gate"] = _fabric(c.gate_proj, r["h2"], spec)
    _, r["up"] = _fabric(c.up_proj, r["h2"], spec)
    r["act"] = swiglu_int(r["gate"], r["up"], c.mult_g, c.sh_g, c.mult_o, c.sh_o)
    _, r["ffn"] = _fabric(c.down_proj, r["act"], spec)
    r["x2"] = residual_int(x1, r["ffn"], c.res_mult, c.res_shift)
    return r


def recurrent_layer_int(c: RecurrentConsts, cfg, spec: TileSpec, x: np.ndarray, s: np.ndarray, hist: np.ndarray,
                        scale: np.ndarray | None = None) -> dict:
    """One token through the integer recurrent layer: ``x`` int16, ``s``
    int16 ``[v_heads, K, V]``, ``hist`` int8 ``[conv_dim, kernel - 1]``.
    With ``scale`` (an int array ``[v_heads, 4]`` of ``g``, ``e``, ``peak``
    and ``nsat`` per head) the state is the int8 ``T`` of
    ``delta_state_int8`` and the result carries ``scale_next``."""
    nk, nv, hk, hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    kd = nk * hk
    r: dict = {"h": _norm(x, c.norm)}
    _, r["qkv"] = _fabric(c.in_proj_qkv, r["h"], spec)
    _, r["z"] = _fabric(c.in_proj_z, r["h"], spec)
    b_acc, _ = _fabric(c.in_proj_b, r["h"], spec)
    a_acc, _ = _fabric(c.in_proj_a, r["h"], spec)
    r["conv"], r["hist_next"] = conv_silu_int(hist, r["qkv"], c.conv_w, c.conv_mult_in, c.conv_sh_in,
                                              c.conv_mult_out, c.conv_sh_out)
    q = np.stack([_norm(r["conv"][i * hk:(i + 1) * hk], c.unit_norm) for i in range(nk)])
    k = np.stack([_norm(r["conv"][kd + i * hk:kd + (i + 1) * hk], c.unit_norm) for i in range(nk)])
    repeat = nv // nk
    r["q_unit"], r["k_unit"] = np.repeat(q, repeat, axis=0), np.repeat(k, repeat, axis=0)
    r["v"] = r["conv"][2 * kd:].reshape(nv, hv)
    r["decay"], r["beta"] = head_gates_int(a_acc[:nv], b_acc[:nv], c.gate_mult_a, c.gate_sh_a, c.gate_mult_b,
                                           c.gate_sh_b, c.a_coef, c.dt_bias)
    s_next = np.zeros_like(s)
    y = np.zeros((nv, hv), dtype=np.int64)
    scale_next = np.zeros((nv, 4), dtype=np.int64)
    for h in range(nv):
        if scale is None:
            s_next[h], y[h] = delta_state_int(s[h], r["k_unit"][h], r["v"][h], r["q_unit"][h], int(r["decay"][h]),
                                              int(r["beta"][h]))
        else:
            s_next[h], *scale_next[h], y[h] = delta_state_int8(s[h], r["k_unit"][h], r["v"][h], r["q_unit"][h],
                                                               int(r["decay"][h]), int(r["beta"][h]), *map(int, scale[h]))
    r["s_next"], r["y"] = s_next, y
    if scale is not None:
        r["scale_next"] = scale_next
    z = r["z"].reshape(nv, hv)
    y_norm = np.zeros((nv, hv), dtype=np.int64)
    for h in range(nv):
        gate = silu_fixed(requant(z[h], c.z_mult, c.z_shift, 16))
        y_norm[h] = _norm(y[h], c.gated_norm, gain=gate)
    r["y_norm"] = y_norm.reshape(nv * hv)
    _, r["mixer"] = _fabric(c.out_proj, r["y_norm"], spec)
    r["x1"] = residual_int(x, r["mixer"], c.res_mult, c.res_shift)
    r.update(_ffn_int(c.ffn, spec, r["x1"]))
    return r


def global_layer_int(c: GlobalConsts, cfg, spec: TileSpec, x: np.ndarray, pos: int, k_rows: np.ndarray,
                     v_rows: np.ndarray) -> dict:
    """One token through the integer global layer over int8 key and value
    rows ``[kv_heads, N, hd]`` (the current token's own, produced here, are
    appended by the caller between calls)."""
    nh, nkv, hd, rd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rotary_dim
    group = nh // nkv
    r: dict = {"h": _norm(x, c.norm)}
    _, qg = _fabric(c.q_proj, r["h"], spec)
    qg = qg.reshape(nh, 2 * hd)
    r["q_raw"], r["gate"] = qg[:, :hd], qg[:, hd:]
    _, k_raw = _fabric(c.k_proj, r["h"], spec)
    _, v = _fabric(c.v_proj, r["h"], spec)
    r["k_raw"], r["v"] = k_raw.reshape(nkv, hd), v.reshape(nkv, hd)
    _, r["index_q"] = _fabric(c.index_q, r["h"], spec)
    _, r["index_k"] = _fabric(c.index_k, r["h"], spec)
    r["index_q_unit"] = _norm(r["index_q"], c.unit_norm)
    sin, cos = rotary_table_int(pos, c.inv_freq)
    r["q"] = np.stack([rotary_int(_norm(r["q_raw"][h], c.q_norm), sin, cos, rd, c.rot_mult_q, c.rot_sh_q) for h in range(nh)])
    r["k"] = np.stack([rotary_int(_norm(r["k_raw"][n], c.k_norm), sin, cos, rd, c.rot_mult_k, c.rot_sh_k) for n in range(nkv)])
    att = np.zeros((nh, hd), dtype=np.int64)
    for n in range(nkv):
        heads = slice(n * group, (n + 1) * group)
        att[heads] = attention_int(r["q"][heads], r["gate"][heads], k_rows[n], v_rows[n], mult_s=c.mult_s, sh_s=c.sh_s,
                                   mult_gate=c.mult_gate, sh_gate=c.sh_gate, mult_o=c.mult_o, sh_o=c.sh_o)
    r["att"] = att.reshape(nh * hd)
    _, r["mixer"] = _fabric(c.o_proj, r["att"], spec)
    r["x1"] = residual_int(x, r["mixer"], c.res_mult, c.res_shift)
    r.update(_ffn_int(c.ffn, spec, r["x1"]))
    return r


# ---------------------------------------------------------------------------
# Vectors for the RTL testbenches
# ---------------------------------------------------------------------------


def _params(directory: Path, **params) -> dict:
    (directory / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    return params


def emit_unit_vectors(directory: Path, rng: np.random.Generator, n: int = 512) -> dict:
    """Stimulus and expected results for the scalar nonlinearities."""
    directory.mkdir(parents=True, exist_ok=True)
    write_luts(directory)
    t = np.concatenate([rng.integers(-32768, 32768, n - 8), [-32768, -8193, -8192, -1, 0, 1, 8191, 32767]])
    write_hex(directory / "t.hex", t, 16)
    write_hex(directory / "exp_sigmoid.hex", sigmoid_fixed(t), UB)
    write_hex(directory / "exp_silu.hex", silu_fixed(t), 16)
    write_hex(directory / "exp_softplus.hex", softplus_fixed(t), 16)
    tu = np.concatenate([rng.integers(0, 1 << 22, n - 4), [0, (32 << FF) - 1, 32 << FF, (1 << 22) - 1]])
    write_hex(directory / "tu.hex", tu, 22)
    write_hex(directory / "exp_exp.hex", exp_neg_fixed(tu), UB)
    sw, lw = 44, 28
    ss = [int(rng.integers(1, 1 << int(rng.integers(1, sw + 1)))) for _ in range(n - 3)] + [1, (1 << sw) - 1, 1 << (sw - 2)]
    ls = [int(rng.integers(1, 1 << int(rng.integers(1, lw + 1)))) for _ in range(n - 3)] + [1, (1 << lw) - 1, 1 << (lw - 1)]
    write_hex(directory / "ss.hex", ss, sw)
    write_hex(directory / "l.hex", ls, lw)
    rs = [rsqrt_fixed(v, sw) for v in ss]
    rc = [recip_fixed(v, lw) for v in ls]
    write_hex(directory / "exp_rsqrt.hex", [r for r, _ in rs], 17)
    write_hex(directory / "exp_rsqrt_a.hex", [a for _, a in rs], 6)
    write_hex(directory / "exp_recip.hex", [r for r, _ in rc], 17)
    write_hex(directory / "exp_recip_lz.hex", [lz for _, lz in rc], 5)
    return _params(directory, N=n, SW=sw, LW=lw)


def emit_rmsnorm_vectors(directory: Path, x: np.ndarray, gain: np.ndarray, mult: int, shift: int, *, xw: int, ow: int,
                         lanes: int, eps_int: int = 0) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    write_luts(directory)
    d = len(x)
    write_hex(directory / "x.hex", x, xw)
    write_hex(directory / "gain.hex", gain, 16)
    write_hex(directory / "expected_y.hex", rmsnorm_int(x, gain, mult, shift, xw=xw, ow=ow, eps_int=eps_int), ow)
    return _params(directory, D=d, XW=xw, OW=ow, L=lanes, SW=sw_for(xw, d), MULT=mult, SHIFT=shift, EPS=eps_int)


def emit_conv_vectors(directory: Path, rng: np.random.Generator, channels: int, kernel: int, lanes: int) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    write_luts(directory)
    hist = rng.integers(-128, 128, (channels, kernel - 1))
    x_new = rng.integers(-128, 128, channels)
    w = rng.integers(-128, 128, (channels, kernel))
    mult_in = rng.integers(1, 1 << MB, channels)
    sh_in = rng.integers(8, 20, channels)
    mult_out = rng.integers(1, 1 << MB, channels)
    sh_out = rng.integers(14, 24, channels)
    y, hist_next = conv_silu_int(hist, x_new, w, mult_in, sh_in, mult_out, sh_out)
    write_hex(directory / "hist.hex", [int(sum(int(v & 0xFF) << (8 * j) for j, v in enumerate(row))) for row in hist], 8 * (kernel - 1))
    write_hex(directory / "x.hex", x_new, 8)
    write_hex(directory / "w.hex", [int(sum(int(v & 0xFF) << (8 * j) for j, v in enumerate(row))) for row in w], 8 * kernel)
    write_hex(directory / "mult_in.hex", mult_in, MB)
    write_hex(directory / "sh_in.hex", sh_in, SHB)
    write_hex(directory / "mult_out.hex", mult_out, MB)
    write_hex(directory / "sh_out.hex", sh_out, SHB)
    write_hex(directory / "expected_y.hex", y, 8)
    write_hex(directory / "expected_hist.hex", [int(sum(int(v & 0xFF) << (8 * j) for j, v in enumerate(row))) for row in hist_next], 8 * (kernel - 1))
    return _params(directory, C=channels, K=kernel, L=lanes)


def emit_head_gate_vectors(directory: Path, rng: np.random.Generator, heads: int, acc_bits: int = 24) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    write_luts(directory)
    a_acc = rng.integers(-(1 << (acc_bits - 1)), 1 << (acc_bits - 1), heads)
    b_acc = rng.integers(-(1 << (acc_bits - 1)), 1 << (acc_bits - 1), heads)
    mult_a, sh_a = rng.integers(1, 1 << MB, heads), rng.integers(16, 30, heads)
    mult_b, sh_b = rng.integers(1, 1 << MB, heads), rng.integers(16, 30, heads)
    a_coef = rng.integers(0, 1 << 16, heads)
    dt_bias = rng.integers(-4096, 4096, heads)
    decay, beta = head_gates_int(a_acc, b_acc, mult_a, sh_a, mult_b, sh_b, a_coef, dt_bias)
    for name, values, width in (("a_acc", a_acc, acc_bits), ("b_acc", b_acc, acc_bits), ("mult_a", mult_a, MB),
                                ("sh_a", sh_a, SHB), ("mult_b", mult_b, MB), ("sh_b", sh_b, SHB), ("a_coef", a_coef, 16),
                                ("dt_bias", dt_bias, 16), ("expected_decay", decay, UB), ("expected_beta", beta, UB)):
        write_hex(directory / f"{name}.hex", values, width)
    return _params(directory, H=heads, ACC=acc_bits)


def emit_delta_vectors(directory: Path, rng: np.random.Generator, k: int, v: int) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    s = rng.integers(-32768, 32768, (k, v))
    kq = rng.standard_normal((2, k))
    kq = np.rint(kq / np.linalg.norm(kq, axis=1, keepdims=True) * 127).astype(np.int64)
    vv = rng.integers(-128, 128, v)
    decay, beta = int(rng.integers(0, 1 << UB)), int(rng.integers(0, 1 << UB))
    s_new, y = delta_state_int(s, kq[0], vv, kq[1], decay, beta)
    pack = lambda row: int(sum((int(e) & 0xFFFF) << (16 * j) for j, e in enumerate(row)))
    write_hex(directory / "s.hex", [pack(row) for row in s], 16 * v)
    write_hex(directory / "k.hex", kq[0], 8)
    write_hex(directory / "q.hex", kq[1], 8)
    write_hex(directory / "v.hex", vv, 8)
    write_hex(directory / "expected_s.hex", [pack(row) for row in s_new], 16 * v)
    write_hex(directory / "expected_y.hex", y, 16)
    return _params(directory, K=k, V=v, DECAY=decay, BETA=beta, YSH=ysh_for(k))


def emit_delta8_vectors(directory: Path, rng: np.random.Generator, k: int, v: int, case: str,
                        vl: int | None = None) -> dict:
    """Vectors for the int8 state engine.  ``case``: ``plain`` (no rescale this
    token), ``renorm`` (the scale crosses one half, exponent kept), ``grow``
    (rescale with room to double the resolution), ``shrink`` (the state
    saturated last token, resolution halved)."""
    directory.mkdir(parents=True, exist_ok=True)
    limit = min(40, PEAK_GROW) if case == "grow" else 126        # no saturated element unless the case wants them
    t = rng.integers(-limit, limit + 1, (k, v))
    if case == "shrink":                                          # more than 1 / 2^SAT_SHIFT of the head saturated
        t.ravel()[:(k * v >> SAT_SHIFT) + 2] = 127
    kq = rng.standard_normal((2, k))
    kq = np.rint(kq / np.linalg.norm(kq, axis=1, keepdims=True) * 127).astype(np.int64)
    vv = rng.integers(-128, 128, v)
    g = int(rng.integers(HALF_U, ONE_U + 1))
    if case == "renorm":                                        # the scale lands in [0.4, 0.5): the peak keeps the exponent
        decay = int(rng.integers(-(-(int(0.4 * (1 << UB)) << UB) // g), (HALF_U << UB) // g))
    elif case == "grow":
        decay = int(rng.integers(0, (HALF_U << UB) // g))
    else:
        decay = int(rng.integers(-(-(HALF_U << UB) // g), 1 << UB))
    beta = int(rng.integers(0, 1 << UB))
    e, peak, nsat = int(rng.integers(E_MIN + 1, E_MAX)), int(np.abs(t).max()), int((np.abs(t) >= 127).sum())
    t_new, g_new, e_new, peak_new, nsat_new, y = delta_state_int8(t, kq[0], vv, kq[1], decay, beta, g, e, peak, nsat)
    assert (g_new == ONE_U) == (case != "plain") and e_new - e == {"grow": 1, "shrink": -1}.get(case, 0), case
    pack = lambda row: int(sum((int(x) & 0xFF) << (8 * j) for j, x in enumerate(row)))
    write_hex(directory / "s.hex", [pack(row) for row in t], 8 * v)
    write_hex(directory / "k.hex", kq[0], 8)
    write_hex(directory / "q.hex", kq[1], 8)
    write_hex(directory / "v.hex", vv, 8)
    write_hex(directory / "expected_s.hex", [pack(row) for row in t_new], 8 * v)
    write_hex(directory / "expected_y.hex", y, 16)
    return _params(directory, K=k, V=v, VL=vl or v, DECAY=decay, BETA=beta, G=g, E=e & 0xFF, PEAK=peak, NSAT=nsat, EXPECTED_G=g_new,
                   EXPECTED_E=e_new & 0xFF, EXPECTED_PEAK=peak_new, EXPECTED_NSAT=nsat_new, YSH=ysh_for(k),
                   PEAK_GROW=PEAK_GROW, SAT_SHIFT=SAT_SHIFT)


def emit_ffn_vectors(directory: Path, rng: np.random.Generator, n: int, lanes: int) -> dict:
    """SwiGLU and the residual add on one stream."""
    directory.mkdir(parents=True, exist_ok=True)
    write_luts(directory)
    g, u = rng.integers(-128, 128, n), rng.integers(-128, 128, n)
    h = rng.integers(-32768, 32768, n)
    mult_g, sh_g = int(rng.integers(1, 1 << MB)), 12
    mult_o, sh_o = int(rng.integers(1, 1 << MB)), 20
    res_mult, res_shift = int(rng.integers(1, 1 << MB)), 9
    act = swiglu_int(g, u, mult_g, sh_g, mult_o, sh_o)
    h2 = residual_int(h, act, res_mult, res_shift)
    for name, values, width in (("g", g, 8), ("u", u, 8), ("h", h, 16), ("expected_act", act, 8), ("expected_h", h2, 16)):
        write_hex(directory / f"{name}.hex", values, width)
    return _params(directory, N=n, L=lanes, MULT_G=mult_g, SH_G=sh_g, MULT_O=mult_o, SH_O=sh_o, RES_MULT=res_mult,
                   RES_SHIFT=res_shift)


def emit_rotary_vectors(directory: Path, rng: np.random.Generator, hd: int, r: int, lanes: int, pos: int) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    write_luts(directory)
    inv_freq = rng.integers(0, 1 << 32, r // 2)
    x = rng.integers(-32768, 32768, hd)
    mult, shift = int(rng.integers(1, 1 << MB)), 22
    sin, cos = rotary_table_int(pos, inv_freq)
    y = rotary_int(x, sin, cos, r, mult, shift)
    write_hex(directory / "inv_freq.hex", inv_freq, 32)
    write_hex(directory / "x.hex", x, 16)
    write_hex(directory / "expected_sin.hex", sin, 16)
    write_hex(directory / "expected_cos.hex", cos, 16)
    write_hex(directory / "expected_y.hex", y, 8)
    return _params(directory, HD=hd, R=r, L=lanes, POS=pos, MULT=mult, SHIFT=shift)


def emit_attention_vectors(directory: Path, rng: np.random.Generator, hd: int, g: int, rows: int, lanes: int) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    write_luts(directory)
    q = rng.integers(-128, 128, (g, hd))
    gate = rng.integers(-128, 128, (g, hd))
    k_rows = rng.integers(-128, 128, (rows, hd))
    v_rows = rng.integers(-128, 128, (rows, hd))
    consts = dict(mult_s=int(rng.integers(1 << 12, 1 << MB)), sh_s=16, mult_gate=int(rng.integers(1, 1 << MB)), sh_gate=12,
                  mult_o=int(rng.integers(1, 1 << MB)), sh_o=24)
    out = attention_int(q, gate, k_rows, v_rows, **consts)
    pack8 = lambda row: int(sum((int(e) & 0xFF) << (8 * j) for j, e in enumerate(row)))
    write_hex(directory / "q.hex", [pack8(row) for row in q], 8 * hd)
    write_hex(directory / "gate.hex", [pack8(row) for row in gate], 8 * hd)
    write_hex(directory / "k.hex", [pack8(row) for row in k_rows], 8 * hd)
    write_hex(directory / "v.hex", [pack8(row) for row in v_rows], 8 * hd)
    write_hex(directory / "expected_out.hex", [pack8(row) for row in out], 8 * hd)
    return _params(directory, HD=hd, G=g, N=rows, L=lanes, **{k.upper(): v for k, v in consts.items()})
