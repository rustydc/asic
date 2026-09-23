import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fabric import layer as L
from fabric.tile import TileSpec

RTL = Path(__file__).parents[1] / "rtl"
SOURCES = [RTL / name for name in ("fabric_sram.sv", "fabric_vector.sv", "fabric_norm.sv", "fabric_recurrent.sv", "fabric_ffn.sv",
                                   "fabric_attention.sv")]


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class FixedPointTest(unittest.TestCase):
    def test_tables_are_close_to_the_functions(self) -> None:
        t = np.arange(-8192, 8192)
        self.assertLess(np.abs(L.sigmoid_fixed(t) / 65536 - 1 / (1 + np.exp(-t / 1024))).max(), 1e-4)
        self.assertLess(np.abs(L.silu_fixed(t) / 1024 - (t / 1024) / (1 + np.exp(-t / 1024))).max(), 2e-3)
        tu = np.arange(0, 32 << 10)
        self.assertLess(np.abs(L.exp_neg_fixed(tu) / 65536 - np.exp(-tu / 1024)).max(), 2e-4)
        self.assertEqual(int(L.exp_neg_fixed(np.array([32 << 10]))[0]), 0)
        ts = np.arange(-16384, 16384)
        self.assertLess(np.abs(L.softplus_fixed(ts) / 1024 - np.log1p(np.exp(ts / 1024))).max(), 1e-3)
        self.assertEqual(int(L.softplus_fixed(np.array([20000]))[0]), 20000)
        turn = np.arange(65536)
        s, c = L.sin_cos_fixed(turn)
        self.assertLess(np.abs(s / 32767 - np.sin(2 * np.pi * turn / 65536)).max(), 5e-5)
        self.assertLess(np.abs(c / 32767 - np.cos(2 * np.pi * turn / 65536)).max(), 5e-5)

    def test_rsqrt_and_reciprocal_are_sixteen_bit_accurate(self) -> None:
        rng = np.random.default_rng(0)
        worst_r = worst_l = 0.0
        for _ in range(5000):
            ss = int(rng.integers(1, 1 << int(rng.integers(1, 45))))
            r, a = L.rsqrt_fixed(ss, 44)
            worst_r = max(worst_r, abs(r * 2.0 ** (a / 2 - 15 - 22) * math.sqrt(ss) - 1))
            l = int(rng.integers(1, 1 << int(rng.integers(1, 29))))
            r, lz = L.recip_fixed(l, 28)
            worst_l = max(worst_l, abs(r * 2.0 ** (lz - 15 - 28) * l - 1))
        self.assertLess(worst_r, 1e-4)
        self.assertLess(worst_l, 1e-4)
        for ss in (1, (1 << 44) - 1, 1 << 42):
            L.rsqrt_fixed(ss, 44)
        with self.assertRaises(ValueError):
            L.rsqrt_fixed(0, 44)

    def test_rmsnorm_int_tracks_the_float_norm(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.integers(-20000, 20000, 4096)
        # unit RMS in 1/16 steps: mult / 2^shift = sqrt(D) / 2^14 * 16
        mult, shift = L._fp(math.sqrt(4096) / (1 << L.NF) * 16)
        y = L.rmsnorm_int(x, 1, mult, shift, xw=16, ow=8)
        ref = x / np.sqrt((x.astype(np.float64) ** 2).mean()) * 16
        self.assertLess(np.abs(y - ref).max(), 1.0)
        unit = L.rmsnorm_int(rng.integers(-128, 128, 128), 1, 1, L.NF - 7, xw=8, ow=8)
        self.assertAlmostEqual(float(np.sqrt(((unit / 128.0) ** 2).sum())), 1.0, delta=0.02)

    def test_delta_state_needs_sixteen_bits_for_a_slow_decay(self) -> None:
        # At d = 0.999 an int8 state cannot decay at all; int16 loses only its bottom 3 percent.
        d = int(round(0.999 * 65535))
        k = np.zeros(128, dtype=np.int64)
        q = np.zeros(128, dtype=np.int64)
        v = np.zeros(128, dtype=np.int64)
        s8 = np.full((128, 128), 100, dtype=np.int64)          # an int8-sized value
        s16 = np.full((128, 128), 20000, dtype=np.int64)
        s8_new, _ = L.delta_state_int(s8, k, v, q, d, 0)
        s16_new, _ = L.delta_state_int(s16, k, v, q, d, 0)
        self.assertTrue((s8_new == s8).all())
        self.assertTrue((s16_new == 19980).all())
        # The int8 state with a scale does decay: the scale carries it and the state is rescaled once when it halves.
        t, g, e, peak, nsat = s8.copy(), L.ONE_U, 0, 100, 0
        rescales = 0
        for step in range(700):
            t, g, e, peak, nsat, _ = L.delta_state_int8(t, k, v, q, d, 0, g, e, peak, nsat)
            rescales += g == L.ONE_U
        self.assertEqual(rescales, 1)
        self.assertTrue((t == 50).all())                       # 100 * 0.999^n at the crossing, rounded once
        self.assertEqual(e, 0)                                 # 50 leaves no room to double
        value = t[0, 0] * g / L.ONE_U * 2.0 ** -e
        self.assertLess(value, 100 * 0.999 ** 680)
        self.assertGreater(value, 100 * 0.999 ** 720)
        # A small state gains resolution at its rescale; one with enough saturated elements loses it at once.
        t, g, e, peak, nsat, _ = L.delta_state_int8(np.full((128, 128), 20), k, v, q, int(0.4 * 65535), 0, L.ONE_U, 0, 20, 0)
        self.assertEqual((e, t[0, 0]), (1, 16))                # 20 * 0.4 * 2
        t, g, e, peak, nsat, _ = L.delta_state_int8(np.full((128, 128), 127), k, v, q, 65535, 0, L.ONE_U, 0, 127, 16384)
        self.assertEqual((e, t[0, 0], g, nsat), (-1, 63, L.ONE_U, 0))   # 127 * 65535/65536 / 2, nothing saturated any more
        t, g, e, peak, nsat, _ = L.delta_state_int8(np.full((128, 128), 127), k, v, q, 65535, 0, L.ONE_U, 0, 127, 200)
        self.assertEqual((e, nsat), (0, 16384))                # 200 of 16384 saturated is within the 1/64 tolerated

    def test_attention_int_matches_a_float_softmax(self) -> None:
        rng = np.random.default_rng(2)
        hd, rows = 64, 40
        q = rng.integers(-100, 100, (2, hd))
        k = rng.integers(-100, 100, (rows, hd))
        v = rng.integers(-128, 128, (rows, hd))
        gate = rng.integers(-128, 128, (2, hd))
        s_q = s_k = 0.03
        mult_s, sh_s = L._fp(s_q * s_k * hd ** -0.5 * 1024)
        mult_gate, sh_gate = L._fp(0.05 * 1024)
        mult_o, sh_o = L._fp(1 / 256 / 65536)                  # output in v units
        out = L.attention_int(q, gate, k, v, mult_s=mult_s, sh_s=sh_s, mult_gate=mult_gate, sh_gate=sh_gate,
                              mult_o=mult_o, sh_o=sh_o)
        scores = (q * s_q) @ (k * s_k).T * hd ** -0.5
        p = np.exp(scores - scores.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        ref = (p @ v) / (1 + np.exp(-gate * 0.05))
        self.assertGreater(cosine(out, ref), 0.999)
        self.assertLess(np.abs(out - ref).max(), 2.0)


class LayerTest(unittest.TestCase):
    """The integer layers against the float reference on the tiny geometry."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
            from fixed_llm_poc import ASICDecoderLayer, tiny_config
        except ImportError:  # pragma: no cover
            raise unittest.SkipTest("PyTorch not installed")
        cls.torch = torch
        cls.cfg = tiny_config()
        torch.manual_seed(0)
        cls.layers = {}
        for idx in (0, 3):
            layer = ASICDecoderLayer(cls.cfg, idx)
            with torch.no_grad():
                for name, p in layer.named_parameters():
                    if name.endswith("norm.weight"):
                        p.add_(0.2 * torch.randn_like(p))
            cls.layers[idx] = layer
        cls.spec = TileSpec(rows=cls.cfg.hidden_size, cols=16)

    @staticmethod
    def weights(layer) -> dict:
        return {k: v.detach().double().numpy() for k, v in layer.state_dict().items()}

    def test_float_reference_equals_the_torch_layer(self) -> None:
        torch, cfg = self.torch, self.cfg
        rng = np.random.default_rng(0)
        layer = self.layers[0]
        w = self.weights(layer)
        xs = rng.standard_normal((6, cfg.hidden_size))
        with torch.no_grad():
            out, state = layer(torch.tensor(xs, dtype=torch.float32).unsqueeze(0))
        nv, hk, hv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
        s = np.zeros((nv, hk, hv))
        hist = np.zeros((layer.linear_attn.conv_dim, cfg.linear_conv_kernel - 1))
        for t, x in enumerate(xs):
            r = L.recurrent_layer_float(w, cfg, x, s, hist)
            s, hist = r["s_next"], r["hist_next"]
            np.testing.assert_allclose(r["x2"], out[0, t].numpy(), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(s, state.matrix[0].numpy(), rtol=1e-3, atol=1e-4)
        # The global layer over its own rows equals torch when every token is in the window.
        layer = self.layers[3]
        w = self.weights(layer)
        with torch.no_grad():
            out, _ = layer(torch.tensor(xs, dtype=torch.float32).unsqueeze(0))
        nkv, hd = cfg.num_key_value_heads, cfg.head_dim
        ks, vs = [], []
        for t, x in enumerate(xs):
            own = L.global_layer_float(w, cfg, x, t, np.zeros((nkv, 1, hd)), np.zeros((nkv, 1, hd)))
            ks.append(own["k"])
            vs.append(own["v"])
            r = L.global_layer_float(w, cfg, x, t, np.stack(ks, axis=1), np.stack(vs, axis=1))
            np.testing.assert_allclose(r["x2"], out[0, t].numpy(), rtol=1e-3, atol=1e-3)

    def test_recurrent_layer_int_tracks_the_float_layer(self) -> None:
        cfg = self.cfg
        rng = np.random.default_rng(3)
        layer = self.layers[0]
        w = self.weights(layer)
        nv, hk, hv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
        conv_dim = layer.linear_attn.conv_dim
        xs = [rng.standard_normal(cfg.hidden_size) * 2.0 for _ in range(10)]

        def float_run(weights):
            s, hist, runs = np.zeros((nv, hk, hv)), np.zeros((conv_dim, cfg.linear_conv_kernel - 1)), []
            for x in xs:
                r = L.recurrent_layer_float(weights, cfg, x, s, hist)
                s, hist = r["s_next"], r["hist_next"]
                runs.append(r)
            return runs

        exact = float_run(w)
        cal = L.calibrate(exact, L.RECURRENT_CAL_KEYS)
        cal["x2"] = max(cal["x2"], max(float(np.abs(x).max()) for x in xs))
        consts = L.compile_recurrent_layer(w, cfg, self.spec, cal)
        self.assertEqual(consts.conv_w.shape, (conv_dim, cfg.linear_conv_kernel))
        self.assertEqual(consts.in_proj_qkv.in_features, cfg.hidden_size)
        # The reference that carries the int4 weights isolates the datapath's own arithmetic.
        quantized = float_run(L.dequantized_weights(w, consts, cfg))
        s_i = np.zeros((nv, hk, hv), dtype=np.int64)
        hist_i = np.zeros((conv_dim, cfg.linear_conv_kernel - 1), dtype=np.int64)
        for x, rq, rf in zip(xs, quantized, exact):
            ri = L.recurrent_layer_int(consts, cfg, self.spec, np.rint(x / consts.s_h).astype(np.int64), s_i, hist_i)
            s_i, hist_i = ri["s_next"], ri["hist_next"]
            self.assertLess(np.abs(rq["decay"] - ri["decay"] / 65535).max(), 0.02)
            self.assertLess(np.abs(rq["beta"] - ri["beta"] / 65535).max(), 0.05)
            self.assertGreater(cosine(rq["y"], ri["y"]), 0.95)
            self.assertGreater(cosine(rq["mixer"], ri["mixer"]), 0.85)
            self.assertGreater(cosine(rq["ffn"], ri["ffn"]), 0.95)
            self.assertGreater(cosine(rq["x2"], ri["x2"] * consts.s_h), 0.998)
            # Against the unquantized weights the residual stream still holds; the mixer is the
            # int4 weights' business (a random-init head whose output is a cancellation flips).
            self.assertGreater(cosine(rf["x2"], ri["x2"] * consts.s_h), 0.99)
        self.assertLess(np.abs(s_i).max(), 32768)

    def test_recurrent_layer_with_the_int8_state_tracks_the_int16_layer(self) -> None:
        cfg = self.cfg
        rng = np.random.default_rng(5)
        w = self.weights(self.layers[0])
        nv, hk, hv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
        conv_dim = self.layers[0].linear_attn.conv_dim
        xs = [rng.standard_normal(cfg.hidden_size) * 2.0 for _ in range(12)]
        s, hist, runs = np.zeros((nv, hk, hv)), np.zeros((conv_dim, cfg.linear_conv_kernel - 1)), []
        for x in xs:
            r = L.recurrent_layer_float(w, cfg, x, s, hist)
            s, hist = r["s_next"], r["hist_next"]
            runs.append(r)
        cal = L.calibrate(runs, L.RECURRENT_CAL_KEYS)
        cal["x2"] = max(cal["x2"], max(float(np.abs(x).max()) for x in xs))
        consts = L.compile_recurrent_layer(w, cfg, self.spec, cal)
        s16 = np.zeros((nv, hk, hv), dtype=np.int64)
        t8 = np.zeros((nv, hk, hv), dtype=np.int64)
        sc8 = np.tile([L.ONE_U, 0, 0, 0], (nv, 1)).astype(np.int64)
        h16 = h8 = np.zeros((conv_dim, cfg.linear_conv_kernel - 1), dtype=np.int64)
        state_cos, y_cos = [], []
        for x in xs:
            xi = np.rint(x / consts.s_h).astype(np.int64)
            r16 = L.recurrent_layer_int(consts, cfg, self.spec, xi, s16, h16)
            r8 = L.recurrent_layer_int(consts, cfg, self.spec, xi, t8, h8, scale=sc8)
            s16, h16 = r16["s_next"], r16["hist_next"]
            t8, sc8, h8 = r8["s_next"], r8["scale_next"], r8["hist_next"]
            # S = g * T * 2^-e against the int16 state at s_v / 256.
            eff = t8 * (sc8[:, 0] / L.ONE_U * 2.0 ** -sc8[:, 1])[:, None, None]
            state_cos.append(cosine(eff.ravel(), s16.ravel() / 256))
            y_cos.append(cosine(r8["y"], r16["y"]))
            self.assertGreater(cosine(r8["x2"], r16["x2"]), 0.998)
        # This random-init model's states are a few s_v and its decays near zero, so every
        # head rescales every token and the exponent climbs to use the byte.
        self.assertGreater(np.mean(state_cos[2:]), 0.99)
        self.assertGreater(np.mean(y_cos[2:]), 0.98)
        self.assertGreater(sc8[:, 1].max(), 0)

    def test_global_layer_int_tracks_the_float_layer(self) -> None:
        cfg = self.cfg
        rng = np.random.default_rng(4)
        layer = self.layers[3]
        w = self.weights(layer)
        nkv, hd = cfg.num_key_value_heads, cfg.head_dim
        xs = [rng.standard_normal(cfg.hidden_size) * 2.0 for _ in range(10)]

        def float_run(weights):
            kf, vf, runs = [], [], []
            for pos, x in enumerate(xs):
                own = L.global_layer_float(weights, cfg, x, pos, np.zeros((nkv, 1, hd)), np.zeros((nkv, 1, hd)))
                kf.append(own["k"])
                vf.append(own["v"])
                runs.append(L.global_layer_float(weights, cfg, x, pos, np.stack(kf, axis=1), np.stack(vf, axis=1)))
            return runs

        exact = float_run(w)
        cal = L.calibrate(exact, L.GLOBAL_CAL_KEYS)
        cal["x2"] = max(cal["x2"], max(float(np.abs(x).max()) for x in xs))
        consts = L.compile_global_layer(w, cfg, self.spec, cal)
        quantized = float_run(L.dequantized_weights(w, consts, cfg))
        ki, vi = [], []
        for pos, (x, rq, rf) in enumerate(zip(xs, quantized, exact)):
            xi = np.rint(x / consts.s_h).astype(np.int64)
            zero = np.zeros((nkv, 1, hd), dtype=np.int64)
            own = L.global_layer_int(consts, cfg, self.spec, xi, pos, zero, zero)
            ki.append(own["k"])
            vi.append(own["v"])
            ri = L.global_layer_int(consts, cfg, self.spec, xi, pos, np.stack(ki, axis=1), np.stack(vi, axis=1))
            self.assertGreater(cosine(rq["q"], ri["q"]), 0.99)
            self.assertGreater(cosine(rq["k"], ri["k"]), 0.99)
            self.assertGreater(cosine(rq["att"], ri["att"]), 0.97)
            self.assertGreater(cosine(rq["mixer"], ri["mixer"]), 0.95)
            self.assertGreater(cosine(rq["x2"], ri["x2"] * consts.s_h), 0.998)
            self.assertGreater(cosine(rf["x2"], ri["x2"] * consts.s_h), 0.99)


@unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog not installed")
class VectorRtlTest(unittest.TestCase):
    """Every vector unit against its Python model, bit for bit, in Icarus Verilog."""

    def run_rtl(self, top: str, emit) -> str:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            params = emit(work)
            args = [f"-P{top}.{name}={value}" for name, value in params.items()]
            subprocess.run(["iverilog", "-g2012", "-I", str(RTL), "-s", top, "-o", "sim.vvp", *args,
                            *map(str, SOURCES), str(RTL / f"{top}.sv")],
                           cwd=work, check=True, capture_output=True, text=True)
            result = subprocess.run(["vvp", "sim.vvp"], cwd=work, check=True, capture_output=True, text=True)
        return result.stdout

    def check(self, top: str, emit) -> None:
        out = self.run_rtl(top, emit)
        self.assertIn("PASS", out, out)

    def test_scalar_units(self) -> None:
        rng = np.random.default_rng(10)
        self.check("tb_vector_units", lambda d: L.emit_unit_vectors(d, rng))

    def test_rmsnorm_in_its_three_uses(self) -> None:
        rng = np.random.default_rng(11)
        # The residual norm: int16 in, int8 out, a gain, an epsilon.
        self.check("tb_rmsnorm", lambda d: L.emit_rmsnorm_vectors(
            d, rng.integers(-32768, 32768, 128), rng.integers(-32768, 32768, 128), 40000, 24, xw=16, ow=8, lanes=4, eps_int=1000))
        # The L2 normaliser: int8 in, an int8 unit vector out.
        self.check("tb_rmsnorm", lambda d: L.emit_rmsnorm_vectors(
            d, rng.integers(-128, 128, 64), np.ones(64, dtype=np.int64), 1, 7, xw=8, ow=8, lanes=8))
        # The head norm: int8 in, a Q3.13 gain, int16 out.
        self.check("tb_rmsnorm", lambda d: L.emit_rmsnorm_vectors(
            d, rng.integers(-128, 128, 32), rng.integers(4000, 12000, 32), 1, 14, xw=8, ow=16, lanes=4))

    def test_conv_silu(self) -> None:
        rng = np.random.default_rng(12)
        self.check("tb_conv_silu", lambda d: L.emit_conv_vectors(d, rng, 64, 4, 8))

    def test_head_gates(self) -> None:
        rng = np.random.default_rng(13)
        self.check("tb_head_gates", lambda d: L.emit_head_gate_vectors(d, rng, 32))

    def test_delta_state(self) -> None:
        rng = np.random.default_rng(14)
        self.check("tb_delta_state", lambda d: L.emit_delta_vectors(d, rng, 16, 16))
        self.check("tb_delta_state", lambda d: L.emit_delta_vectors(d, rng, 128, 128))

    def test_delta_state8(self) -> None:
        rng = np.random.default_rng(18)
        for k, v, case in ((16, 16, "plain"), (16, 16, "renorm"), (16, 16, "grow"), (16, 16, "shrink"),
                           (128, 128, "renorm"), (128, 128, "grow")):
            self.check("tb_delta_state8", lambda d: L.emit_delta8_vectors(d, rng, k, v, case))
        # The same head sliced: sixteen lanes of arithmetic over eight slices,
        # which is the rate the rows arrive at over the buffer's port.
        for k, v, vl, case in ((128, 128, 16, "renorm"), (128, 128, 16, "grow"),
                               (128, 128, 32, "shrink"), (16, 16, 4, "plain")):
            self.check("tb_delta_state8", lambda d: L.emit_delta8_vectors(d, rng, k, v, case, vl))

    def test_swiglu_and_residual(self) -> None:
        rng = np.random.default_rng(15)
        self.check("tb_ffn", lambda d: L.emit_ffn_vectors(d, rng, 256, 8))

    def test_rotary(self) -> None:
        rng = np.random.default_rng(16)
        self.check("tb_rotary", lambda d: L.emit_rotary_vectors(d, rng, 64, 16, 8, 12345))
        self.check("tb_rotary", lambda d: L.emit_rotary_vectors(d, rng, 256, 64, 16, 131071))

    def test_attention(self) -> None:
        rng = np.random.default_rng(17)
        self.check("tb_attention", lambda d: L.emit_attention_vectors(d, rng, 32, 2, 20, 8))
        self.check("tb_attention", lambda d: L.emit_attention_vectors(d, rng, 256, 4, 24, 64))


if __name__ == "__main__":
    unittest.main()
