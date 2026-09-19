import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fabric import layer as L
from fabric import memory as M
from fabric.tile import TileSpec

RTL = Path(__file__).parents[1] / "rtl"
SOURCES = [RTL / name for name in ("fabric_vector.sv", "fabric_norm.sv", "fabric_recurrent.sv", "fabric_ffn.sv",
                                   "fabric_attention.sv", "fabric_memory.sv")]

SMALL = dict(kv_heads=2, head_dim=64, local_window=16, block=4, context_tokens=64, index_dim=32, v_heads=2, k_dim=16,
             v_dim=16, conv_dim=64, recurrent_layers=1)


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class MapTest(unittest.TestCase):
    def test_nine_b_map_sizes_and_capacity(self) -> None:
        mm = M.MemoryMap()                                   # the 9B geometry, int8 state with its scales, int8 KV
        self.assertEqual(mm.state_bytes, 32 * 16 + 32 * 128 * 128)
        self.assertEqual(M.MemoryMap(state_bits=16).state_bytes, 32 * 128 * 128 * 2)
        self.assertEqual(mm.kv_record_bytes, 512)
        self.assertEqual(mm.window_bytes, 512 * 4 * 512)
        self.assertEqual(mm.blocks, 8192)
        self.assertEqual(mm.index_record_bytes, 80)
        regions = mm.regions()
        self.assertEqual(regions["state0"][0], 0)
        self.assertEqual(regions["hist0"][0], -(-mm.state_bytes // M.ALIGN) * M.ALIGN)
        self.assertTrue(all(off % M.ALIGN == 0 for off, _ in regions.values()))
        # int4 KV halves the block store; the int8 state is half the int16 one plus a beat per head.
        four = M.MemoryMap(kv_bits=4)
        self.assertEqual(four.block_store_bytes, mm.block_store_bytes // 2)
        self.assertEqual(mm.state_bytes, M.MemoryMap(state_bits=16).state_bytes // 2 + 32 * M.BEAT)
        self.assertGreater(mm.contexts_that_fit(4 << 30), four.contexts_that_fit(1 << 30))
        self.assertGreaterEqual(four.contexts_that_fit(1 << 30), 32)
        self.assertIn("| **total** |", mm.report_markdown({"PSRAM": 1 << 30}))

    def test_addresses_follow_the_records(self) -> None:
        mm = M.MemoryMap(**SMALL)
        self.assertEqual(mm.state_scale_addr(0, 0, 1), M.BEAT)
        self.assertEqual(mm.state_row_addr(0, 0, 1, 2), mm.state_scale_bytes + (16 + 2) * mm.state_row_bytes)
        self.assertEqual(M.MemoryMap(**SMALL, state_bits=16).state_row_addr(0, 0, 1, 2), (16 + 2) * 32)
        self.assertEqual(mm.window_record_addr(0, 17, 1), mm.regions()["window0"][0] + (1 * 2 + 1) * mm.kv_record_bytes)
        self.assertEqual(mm.block_record_addr(1, 3, 0), mm.context_bytes + mm.regions()["blocks0"][0] + 3 * 2 * mm.kv_record_bytes)
        self.assertEqual(mm.index_record_addr(0, 5), mm.regions()["index0"][0] + 5 * mm.index_record_bytes)
        traffic = mm.bytes_per_token(63, 2)
        self.assertEqual(traffic["window_read"], 16 * 2 * mm.kv_record_bytes)
        self.assertEqual(traffic["index_scan"], M.eligible_blocks(63, 16, 4) * mm.index_record_bytes)
        self.assertEqual(traffic["block_append"], 2 * mm.kv_record_bytes + mm.index_record_bytes)


class FunctionTest(unittest.TestCase):
    def test_kv_pack_round_trips(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.integers(-128, 128, 64)
        for bits in (8, 4):
            stored = M.kv_quant(x, bits)
            data = M.kv_pack(stored, bits, M.MemoryMap(head_dim=64, kv_bits=bits).kv_half_bytes)
            np.testing.assert_array_equal(M.kv_unpack(data, bits, 64), stored)
        self.assertTrue((M.kv_quant(x, 4) % 16 == 0).all())
        self.assertLessEqual(np.abs(M.kv_quant(x, 4) - x).max(), 15)     # 8 of rounding, more only at 127

    def test_index_codes_follow_the_reference_quantiser(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(20):
            u = M.index_unit(rng.integers(-128, 128, 128))
            codes, scale = M.index_codes(u)
            ref, _ = M.index_codes_float(u.astype(np.float64))
            self.assertEqual(scale, int(np.abs(u).max()))
            self.assertLessEqual(np.abs(codes - ref).max(), 1)        # the reciprocal's rounding
            self.assertLess((codes != ref).mean(), 0.2)                # ties round differently
            data = M.index_codes_pack(codes, 64)
            np.testing.assert_array_equal(M.index_codes_unpack(data, 128), codes)
        q = M.index_codes(M.index_unit(rng.integers(-128, 128, 128)))[0]
        self.assertEqual(M.index_score(q, q, 100), 100 * int(((2 * q - 15) ** 2).sum()))

    def test_eligibility_and_topk(self) -> None:
        self.assertEqual(M.eligible_blocks(10, 16, 4), 0)
        self.assertEqual(M.eligible_blocks(15, 16, 4), 0)
        self.assertEqual(M.eligible_blocks(19, 16, 4), 1)       # block 0 ends at 3 = 19 - 16
        self.assertEqual(M.eligible_blocks(131071, 512, 16), (131072 - 512) // 16)
        ranked = M.topk_stream([(0, 5), (1, 9), (2, 9), (3, 1), (4, 7)], 3)
        self.assertEqual(ranked, [(1, 9), (2, 9), (4, 7)])
        self.assertEqual(M.topk_stream([], 3), [])


class StoreTest(unittest.TestCase):
    """The float twin against PyTorch, and the integer store against the twin."""

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
        cls.layer = ASICDecoderLayer(cls.cfg, 3)
        with torch.no_grad():
            for name, p in cls.layer.named_parameters():
                if name.endswith("norm.weight"):
                    p.add_(0.2 * torch.randn_like(p))
        cls.w = {k: v.detach().double().numpy() for k, v in cls.layer.state_dict().items()}
        cls.spec = TileSpec(rows=cls.cfg.hidden_size, cols=16)

    def test_float_store_reproduces_the_torch_mixer(self) -> None:
        torch, cfg, w = self.torch, self.cfg, self.w
        rng = np.random.default_rng(0)
        xs = rng.standard_normal((48, cfg.hidden_size))
        with torch.no_grad():
            out, _ = self.layer(torch.tensor(xs, dtype=torch.float32).unsqueeze(0))
        nkv, hd = cfg.num_key_value_heads, cfg.head_dim
        mem = M.GlobalContextMemoryFloat(cfg)
        selected = []
        for t, x in enumerate(xs):
            own = L.global_layer_float(w, cfg, x, t, np.zeros((nkv, 1, hd)), np.zeros((nkv, 1, hd)))
            mem.append(t, own["k"], own["v"], own["index_k"])
            got = mem.retrieve(t, own["index_q"])
            r = L.global_layer_float(w, cfg, x, t, got["k_rows"], got["v_rows"])
            np.testing.assert_allclose(r["x2"], out[0, t].numpy(), rtol=1e-4, atol=1e-4)
            selected.append(got["selected"])
        self.assertEqual(len(selected[15]), 0)                  # nothing eligible inside the window
        self.assertEqual(len(selected[47]), cfg.top_blocks)

    def test_integer_store_tracks_the_float_store(self) -> None:
        cfg, w = self.cfg, self.w
        rng = np.random.default_rng(5)
        nkv, hd = cfg.num_key_value_heads, cfg.head_dim
        xs = [rng.standard_normal(cfg.hidden_size) * 2.0 for _ in range(40)]
        # Calibrate and compile from a float run with the float store.
        mem_f = M.GlobalContextMemoryFloat(cfg)
        runs = []
        for t, x in enumerate(xs):
            own = L.global_layer_float(w, cfg, x, t, np.zeros((nkv, 1, hd)), np.zeros((nkv, 1, hd)))
            mem_f.append(t, own["k"], own["v"], own["index_k"])
            got = mem_f.retrieve(t, own["index_q"])
            runs.append(L.global_layer_float(w, cfg, x, t, got["k_rows"], got["v_rows"]))
        cal = L.calibrate(runs, L.GLOBAL_CAL_KEYS)
        cal["x2"] = max(cal["x2"], max(float(np.abs(x).max()) for x in xs))
        consts = L.compile_global_layer(w, cfg, self.spec, cal)
        wq = L.dequantized_weights(w, consts, cfg)
        for kv_bits in (8, 4):
            mm = M.MemoryMap.from_config(cfg, context_tokens=64, kv_bits=kv_bits)
            mem_i = M.GlobalContextMemory(mm, cfg.top_blocks)
            mem_q = M.GlobalContextMemoryFloat(cfg)
            overlaps, att_cos, out_cos = [], [], []
            for t, x in enumerate(xs):
                xi = np.rint(x / consts.s_h).astype(np.int64)
                zero = np.zeros((nkv, 1, hd), dtype=np.int64)
                own = L.global_layer_int(consts, cfg, self.spec, xi, t, zero, zero)
                mem_i.append(t, own["k"], own["v"], own["index_k"])
                got = mem_i.retrieve(t, own["index_q_unit"])
                ri = L.global_layer_int(consts, cfg, self.spec, xi, t, got["k_rows"], got["v_rows"])
                own_q = L.global_layer_float(wq, cfg, x, t, np.zeros((nkv, 1, hd)), np.zeros((nkv, 1, hd)))
                mem_q.append(t, own_q["k"], own_q["v"], own_q["index_k"])
                got_q = mem_q.retrieve(t, own_q["index_q"])
                rq = L.global_layer_float(wq, cfg, x, t, got_q["k_rows"], got_q["v_rows"])
                if got_q["selected"]:
                    chosen = {b for b, _ in got["selected"]}
                    overlaps.append(len(chosen & {b for b, _ in got_q["selected"]}) / len(got_q["selected"]))
                att_cos.append(cosine(rq["att"], ri["att"]))
                out_cos.append(cosine(rq["x2"], ri["x2"] * consts.s_h))
            self.assertEqual(len(got["k_rows"][0]), min(40, cfg.local_window) + cfg.top_blocks)
            self.assertGreater(float(np.mean(overlaps)), 0.7, f"kv_bits={kv_bits}")
            self.assertGreater(float(np.mean(att_cos)), 0.95, f"kv_bits={kv_bits}")
            self.assertGreater(min(out_cos), 0.995, f"kv_bits={kv_bits}")


@unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog not installed")
class MemoryRtlTest(unittest.TestCase):
    def check(self, top: str, emit) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            params = emit(work)
            args = [f"-P{top}.{name}={value}" for name, value in params.items()]
            subprocess.run(["iverilog", "-g2012", "-I", str(RTL), "-s", top, "-o", "sim.vvp", *args,
                            *map(str, SOURCES), str(RTL / f"{top}.sv")],
                           cwd=work, check=True, capture_output=True, text=True)
            result = subprocess.run(["vvp", "sim.vvp"], cwd=work, check=True, capture_output=True, text=True)
        self.assertIn("PASS", result.stdout, result.stdout)

    def test_topk(self) -> None:
        rng = np.random.default_rng(20)
        self.check("tb_topk", lambda d: M.emit_topk_vectors(d, rng, 8, 64))

    def test_index_scan(self) -> None:
        rng = np.random.default_rng(21)
        self.check("tb_index_scan", lambda d: M.emit_index_scan_vectors(d, rng, M.MemoryMap(**SMALL), 40, 8))
        self.check("tb_index_scan", lambda d: M.emit_index_scan_vectors(d, rng, M.MemoryMap(context_tokens=2048), 100, 32))

    def test_kv_append(self) -> None:
        rng = np.random.default_rng(22)
        self.check("tb_kv_append", lambda d: M.emit_kv_append_vectors(d, rng, M.MemoryMap(**SMALL), 20, 2))
        self.check("tb_kv_append", lambda d: M.emit_kv_append_vectors(d, rng, M.MemoryMap(**SMALL, kv_bits=4), 20, 2))

    def test_record_reader_into_attention(self) -> None:
        rng = np.random.default_rng(23)
        self.check("tb_record_reader", lambda d: M.emit_record_reader_vectors(d, rng, M.MemoryMap(**SMALL), 30, 2, 2, 16))
        self.check("tb_record_reader", lambda d: M.emit_record_reader_vectors(d, rng, M.MemoryMap(**SMALL, kv_bits=4), 30, 2, 2, 16))

    def test_row_dma_round_trip(self) -> None:
        rng = np.random.default_rng(24)
        self.check("tb_row_dma", lambda d: M.emit_row_dma_vectors(d, rng, 16, 16))
        self.check("tb_row_dma", lambda d: M.emit_row_dma_vectors(d, rng, 128, 128))


if __name__ == "__main__":
    unittest.main()
