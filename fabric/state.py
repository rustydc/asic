"""The recurrent state's width, by simulation.

The delta rule keeps a ``K x V`` state per head that every token decays
and adds a rank-one update to, and the memory moves all of it in and out
each token: at the 9B geometry 2 MB per token per recurrent layer as
int16, which is what makes the layer memory-bound on sixteen PSRAMs.  An
int8 state halves that, but round-to-nearest cannot apply a slow decay to
a narrow value (``S * d`` rounds back to ``S`` below ``1 / (1 - d)``).

This module runs the candidate schemes against the float rule on
synthetic heads and reports the cosine of the read-out ``y = q . S``:

* int16 round-to-nearest, as first built (``delta_state_int``);
* int8 round-to-nearest at the top byte of the int16 range;
* int8 stochastic rounding, which is unbiased but noisy;
* int8 with a per-head scale that absorbs the decay and an exponent
  that keeps the byte in use (``delta_state_int8``): the scheme the design
  takes.

``python -m fabric.state`` prints the table.
"""
from __future__ import annotations

import numpy as np

from fabric.layer import ONE_U, delta_state_int, delta_state_int8

DECAY_MODES = {"slow": "d = 0.999", "fast": "d = 0.98", "mixed": "rate log-uniform, d = 0.905 to 0.999"}


def unit8(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.standard_normal(n)
    x /= np.linalg.norm(x)
    return np.clip(np.rint(x * 127), -127, 127).astype(np.int64)


def make_tokens(rng: np.random.Generator, n: int, k: int, v: int, mode: str) -> list[tuple]:
    """``(k, v, q, decay, beta)`` per token: unit int8 keys and queries, int8 values, U16 gates."""
    toks = []
    for _ in range(n):
        if mode == "slow":
            d = 0.999
        elif mode == "fast":
            d = 0.98
        else:
            d = float(np.exp(-np.exp(rng.uniform(np.log(0.001), np.log(0.1)))))
        toks.append((unit8(rng, k), np.clip(np.rint(rng.standard_normal(v) * 40), -127, 127).astype(np.int64),
                     unit8(rng, k), int(d * ONE_U), int(rng.uniform(0.3, 1.0) * ONE_U)))
    return toks


def truth(toks: list[tuple], k: int, v: int) -> list[np.ndarray]:
    """The float delta rule on the same integer inputs; y in the units of the int16 read-out."""
    s = np.zeros((k, v))
    ys = []
    for kk, vv, qq, d, b in toks:
        kf, qf = kk / 127.0, qq / 127.0
        s = s * (d / ONE_U)
        s = s + (b / ONE_U) * np.outer(kf, vv - kf @ s)
        ys.append(qf @ s)
    return ys


def _sround(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = np.floor(x)
    return (f + (rng.random(x.shape) < (x - f))).astype(np.int64)


def int8_plain(toks: list[tuple], k: int, v: int, stochastic: bool, rng: np.random.Generator) -> list[np.ndarray]:
    """An int8 state at scale s_v, the decay rounded into the state every token."""
    s = np.zeros((k, v), dtype=np.int64)
    ys = []
    for kk, vv, qq, d, b in toks:
        kf, qf = kk / 127.0, qq / 127.0
        sd = s * (d / ONE_U)
        sd = np.clip(_sround(sd, rng) if stochastic else np.rint(sd), -127, 127).astype(np.int64)
        inc = (b / ONE_U) * np.outer(kf, vv - kf @ sd)
        s = np.clip(sd + (_sround(inc, rng) if stochastic else np.rint(inc).astype(np.int64)), -127, 127).astype(np.int64)
        ys.append(qf @ s)
    return ys


def int16_built(toks: list[tuple], k: int, v: int) -> list[np.ndarray]:
    s = np.zeros((k, v), dtype=np.int64)
    ys = []
    for kk, vv, qq, d, b in toks:
        s, y = delta_state_int(s, kk, vv, qq, d, b)
        ys.append(y)
    return ys


def int8_scaled(toks: list[tuple], k: int, v: int) -> list[np.ndarray]:
    t = np.zeros((k, v), dtype=np.int64)
    g, e, peak, nsat = ONE_U, 0, 0, 0
    ys = []
    for kk, vv, qq, d, b in toks:
        t, g, e, peak, nsat, y = delta_state_int8(t, kk, vv, qq, d, b, g, e, peak, nsat)
        ys.append(y)
    return ys


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


SCHEMES = ("int16 round-to-nearest (built first)", "int8 round-to-nearest", "int8 stochastic rounding",
           "int8 with a per-head scale (taken)")


def study(n: int = 3000, k: int = 128, v: int = 128, seed: int = 1) -> dict[str, dict[str, tuple[float, float]]]:
    """``mode -> scheme -> (mean, min)`` cosine of y against the float rule over the last two thirds of n tokens."""
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for mode in DECAY_MODES:
        toks = make_tokens(rng, n, k, v, mode)
        yt = truth(toks, k, v)
        runs = {SCHEMES[0]: int16_built(toks, k, v),
                SCHEMES[1]: int8_plain(toks, k, v, False, np.random.default_rng(seed)),
                SCHEMES[2]: int8_plain(toks, k, v, True, np.random.default_rng(seed)),
                SCHEMES[3]: int8_scaled(toks, k, v)}
        out[mode] = {}
        for name, ys in runs.items():
            c = [cosine(a, b) for a, b in zip(ys[n // 3:], yt[n // 3:])]
            out[mode][name] = (float(np.mean(c)), float(np.min(c)))
    return out


def report_markdown(result: dict[str, dict[str, tuple[float, float]]], n: int, k: int, v: int) -> str:
    lines = [f"Cosine of the read-out y against the float delta rule, {k} x {v} heads, the last {n - n // 3} of {n} tokens (mean / min).", "",
             "| Scheme | " + " | ".join(f"{m} ({DECAY_MODES[m]})" for m in DECAY_MODES) + " |",
             "| --- |" + " ---: |" * len(DECAY_MODES)]
    for scheme in SCHEMES:
        cells = [f"{result[m][scheme][0]:.4f} / {result[m][scheme][1]:.4f}" for m in DECAY_MODES]
        lines.append(f"| {scheme} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    n, k, v = 3000, 128, 128
    print(report_markdown(study(n, k, v), n, k, v))


if __name__ == "__main__":
    main()
