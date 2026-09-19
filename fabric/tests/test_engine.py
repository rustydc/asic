import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from fabric import engine as E
from fabric import layer as L
from fabric import sequencer as S
from fabric.memory import GlobalContextMemory, MemoryMap
from fabric.tile import TileSpec

RTL = Path(__file__).parents[1] / "rtl"
SOURCES = [RTL / name for name in ("fabric_vector.sv", "fabric_norm.sv", "fabric_recurrent.sv", "fabric_ffn.sv", "fabric_attention.sv",
                                   "fabric_memory.sv", "fabric_tile.sv", "fabric_sequencer.sv", "fabric_engine.sv",
                                   "fabric_phy.sv", "fabric_cdc.sv", "fabric_hpi.sv", "tb_layer_engine.sv")]


def run_engine(case: unittest.TestCase, cfg, c, spec, mm, steps: list[S.Step], inputs: dict, memory=None, ndev: int = 0,
               model_tiles: bool = False, log=None) -> int:
    """Emit, simulate and check one program; returns the engine's cycle count.  With ``ndev`` the memory is the HPI path,
    with ``model_tiles`` the tiles' behavioural columns (full-size runs)."""
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        t0 = time.time()
        run = E.EngineRun(work, cfg, c, spec, mm, steps, inputs, memory, ndev, model_tiles)
        args = [f"-Ptb_layer_engine.{name}={value}" for name, value in run.params.items()]
        t1 = time.time()
        subprocess.run(["iverilog", "-g2012", "-I", str(RTL), "-s", "tb_layer_engine", "-o", "sim.vvp", *args, *map(str, SOURCES)],
                       cwd=work, check=True, capture_output=True, text=True)
        t2 = time.time()
        out = subprocess.run(["vvp", "sim.vvp"], cwd=work, check=True, capture_output=True, text=True).stdout
        if log is not None:
            log(f"emit {t1 - t0:.0f} s, compile {t2 - t1:.0f} s, simulate {time.time() - t2:.0f} s: {out.strip().splitlines()[-2]}")
        case.assertIn("PASS", out, out)
        case.assertEqual(run.check(work), [])
        issue = [tuple(int(v) for v in line.split()) for line in (work / "issue.txt").read_text().splitlines()]
    case.assertEqual([row[1] for row in issue], [i % 256 for i in range(len(steps))])     # program order, tags in step order
    passed = next(line for line in out.splitlines() if line.startswith("PASS"))
    return int(passed.split(" in ")[1].split()[0])


class LayoutTest(unittest.TestCase):
    def test_layout_places_every_operand_once_and_aligned(self) -> None:
        from fixed_llm_poc import tiny_config
        cfg = tiny_config()
        spec, mm = TileSpec(rows=cfg.hidden_size, cols=16), MemoryMap.from_config(cfg)
        prog = S.recurrent_program(cfg, None, spec, mm)
        lay = E.Layout(prog, S.recurrent_layout(cfg, spec, mm)["sizes"])
        self.assertTrue(all(a % 16 == 0 for a in lay.vb.values()))
        spans = sorted((a, a + lay.size(n)) for n, a in lay.vb.items())
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            self.assertLessEqual(a1, b0)                                     # no two buffers overlap
        self.assertEqual(len([n for n in lay.vb if n.startswith("s_slot")]), 4)    # the four heads' slots
        words = S.encode(prog, lay)
        conv = next(i for i, s in enumerate(prog) if s.name == "conv")
        self.assertEqual((words[conv] >> 64) & ((1 << 30) - 1), lay.vb["P1"])            # src
        self.assertEqual((words[conv] >> 154) & ((1 << 30) - 1), lay.vb["hist_next"])     # a3
        # A stream keeps the slots shared and gives each token its own copies of the rest.
        two = S.stream(prog, 2)
        lay2 = E.Layout(two, S.recurrent_layout(cfg, spec, mm)["sizes"])
        self.assertIn("x@0", lay2.vb)
        self.assertIn("x@1", lay2.vb)
        self.assertEqual(len([n for n in lay2.vb if n.startswith("s_slot")]), 4)
        self.assertEqual(len(lay2.mem), 2 * len(lay.mem))


@unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog not installed")
class EngineRtlTest(unittest.TestCase):
    """The layer engine reproduces the integer recurrent layer bit for bit
    on the tiny geometry: the residual out, the int8 state and its scales,
    the conv history."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
            from fixed_llm_poc import ASICDecoderLayer, tiny_config
        except ImportError:  # pragma: no cover
            raise unittest.SkipTest("PyTorch not installed")
        cls.cfg = cfg = tiny_config()
        torch.manual_seed(0)
        layer = ASICDecoderLayer(cfg, 0)
        w = {k: v.detach().double().numpy() for k, v in layer.state_dict().items()}
        cls.spec = TileSpec(rows=cfg.hidden_size, cols=16)
        cls.mm = MemoryMap.from_config(cfg)
        rng = np.random.default_rng(3)
        nv, hk, hv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
        cls.conv_dim = layer.linear_attn.conv_dim
        cls.xs = [rng.standard_normal(cfg.hidden_size) * 2.0 for _ in range(6)]
        s, hist, runs = np.zeros((nv, hk, hv)), np.zeros((cls.conv_dim, cfg.linear_conv_kernel - 1)), []
        for x in cls.xs:
            r = L.recurrent_layer_float(w, cfg, x, s, hist)
            s, hist = r["s_next"], r["hist_next"]
            runs.append(r)
        cal = L.calibrate(runs, L.RECURRENT_CAL_KEYS)
        cal["x2"] = max(cal["x2"], max(float(np.abs(x).max()) for x in cls.xs))
        cls.c = L.compile_recurrent_layer(w, cfg, cls.spec, cal)
        cls.prog = S.recurrent_program(cfg, cls.c, cls.spec, cls.mm)

    def context_after(self, tokens: int) -> dict:
        """The int8 state, its scales and the history after ``tokens`` tokens of the sequence."""
        cfg, nv, hk, hv = self.cfg, self.cfg.linear_num_value_heads, self.cfg.linear_key_head_dim, self.cfg.linear_value_head_dim
        s = np.zeros((nv, hk, hv), dtype=np.int64)
        sc = np.tile([L.ONE_U, 0, 0, 0], (nv, 1)).astype(np.int64)
        hist = np.zeros((self.conv_dim, cfg.linear_conv_kernel - 1), dtype=np.int64)
        for x in self.xs[:tokens]:
            r = L.recurrent_layer_int(self.c, cfg, self.spec, np.rint(x / self.c.s_h).astype(np.int64), s, hist, scale=sc)
            s, sc, hist = r["s_next"], r["scale_next"], r["hist_next"]
        return {"x": np.rint(self.xs[tokens] / self.c.s_h).astype(np.int64), "s_mem": s, "scale_mem": sc, "hist_mem": hist}

    def run_engine(self, steps: list[S.Step], inputs: dict, ndev: int = 0) -> int:
        return run_engine(self, self.cfg, self.c, self.spec, self.mm, steps, inputs, ndev=ndev)

    def test_one_token_from_a_running_context(self) -> None:
        inputs = self.context_after(2)
        self.assertGreater(np.abs(inputs["s_mem"]).max(), 0)
        cycles = self.run_engine(self.prog, inputs)
        self.assertGreater(cycles, S.schedule(self.prog).cycles // 2)

    def test_one_token_over_the_hpi_devices(self) -> None:
        # The same token with the bridge, the stripe unit and four PSRAM models behind the memory port.
        cycles = self.run_engine(self.prog, self.context_after(2), ndev=4)
        self.assertGreater(cycles, S.schedule(self.prog).cycles // 2)

    def test_a_chunk_of_three_tokens(self) -> None:
        # Prefill: three tokens of one context through the multi-token tiles, each head's state moved once.
        inputs = self.context_after(1)
        inputs["x"] = np.stack([np.rint(x / self.c.s_h).astype(np.int64) for x in self.xs[1:4]])
        chunk = S.recurrent_program(self.cfg, self.c, self.spec, self.mm, chunk=3)
        self.assertEqual(len([s for s in chunk if s.unit == "mem"]), len([s for s in self.prog if s.unit == "mem"]))
        cycles = self.run_engine(chunk, inputs)
        single = self.run_engine(self.prog, self.context_after(1))
        self.assertLess(cycles, 2.5 * single)                    # three tokens for well under three tokens' time

    def test_a_stream_of_two_contexts(self) -> None:
        two = S.stream(self.prog, 2)
        inputs = {}
        for token, tokens in ((0, 3), (1, 1)):
            inputs.update({f"{k}@{token}": v for k, v in self.context_after(tokens).items()})
        self.run_engine(two, inputs)


@unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog not installed")
class GlobalEngineRtlTest(unittest.TestCase):
    """The global layer on the engine: the append, the index scan and
    top-K, the record reader and the attention cores over a filled
    context, bit for bit against the integer layer and the memory model."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
            from fixed_llm_poc import ASICDecoderLayer, tiny_config
        except ImportError:  # pragma: no cover
            raise unittest.SkipTest("PyTorch not installed")
        cls.cfg = cfg = tiny_config()
        torch.manual_seed(0)
        layer = ASICDecoderLayer(cfg, 3)
        w = {k: v.detach().double().numpy() for k, v in layer.state_dict().items()}
        cls.spec = TileSpec(rows=cfg.hidden_size, cols=16)
        cls.mm = MemoryMap.from_config(cfg, context_tokens=256, recurrent_layers=0)     # a small image: the global stores only
        rng = np.random.default_rng(4)
        nkv, hd = cfg.num_key_value_heads, cfg.head_dim
        cls.xs = [rng.standard_normal(cfg.hidden_size) * 2.0 for _ in range(40)]
        kf, vf, runs = [], [], []
        for pos, x in enumerate(cls.xs):
            own = L.global_layer_float(w, cfg, x, pos, np.zeros((nkv, 1, hd)), np.zeros((nkv, 1, hd)))
            kf.append(own["k"])
            vf.append(own["v"])
            runs.append(L.global_layer_float(w, cfg, x, pos, np.stack(kf, axis=1), np.stack(vf, axis=1)))
        cal = L.calibrate(runs, L.GLOBAL_CAL_KEYS)
        cal["x2"] = max(cal["x2"], max(float(np.abs(x).max()) for x in cls.xs))
        cls.c = L.compile_global_layer(w, cfg, cls.spec, cal)

    def context_at(self, pos: int) -> tuple[dict, tuple[bytes, bytes]]:
        """The context after positions before ``pos``, the token's inputs
        (its rows as the memory serves them after its own append) and the
        image before and after."""
        cfg, mm = self.cfg, self.mm
        nkv, hd = cfg.num_key_value_heads, cfg.head_dim
        store = GlobalContextMemory(mm, cfg.top_blocks)
        zero = np.zeros((nkv, 1, hd), dtype=np.int64)
        own = None
        for p in range(pos + 1):
            xi = np.rint(self.xs[p] / self.c.s_h).astype(np.int64)
            own = L.global_layer_int(self.c, cfg, self.spec, xi, p, zero, zero)
            if p == pos:
                before = bytes(store.image.data)
            store.append(p, own["k"], own["v"], own["index_k"])
        got = store.retrieve(pos, own["index_q_unit"])
        inputs = {"x": np.rint(self.xs[pos] / self.c.s_h).astype(np.int64), "k_rows": got["k_rows"], "v_rows": got["v_rows"]}
        return inputs, (before, bytes(store.image.data))

    def chunk_at(self, pos: int, tokens: int) -> tuple[dict, tuple[bytes, bytes]]:
        """A chunk from ``pos``: each token's rows as the memory serves them after its own append."""
        cfg, mm = self.cfg, self.mm
        nkv, hd = cfg.num_key_value_heads, cfg.head_dim
        store = GlobalContextMemory(mm, cfg.top_blocks)
        zero = np.zeros((nkv, 1, hd), dtype=np.int64)
        xs, k_rows, v_rows = [], [], []
        for p in range(pos + tokens):
            xi = np.rint(self.xs[p] / self.c.s_h).astype(np.int64)
            own = L.global_layer_int(self.c, cfg, self.spec, xi, p, zero, zero)
            if p == pos:
                before = bytes(store.image.data)
            store.append(p, own["k"], own["v"], own["index_k"])
            if p >= pos:
                got = store.retrieve(p, own["index_q_unit"])
                xs.append(xi)
                k_rows.append(got["k_rows"])
                v_rows.append(got["v_rows"])
        return {"x": np.stack(xs), "k_rows": k_rows, "v_rows": v_rows}, (before, bytes(store.image.data))

    def test_a_chunk_of_three_tokens(self) -> None:
        # Prefill from position 30: the middle token closes a block; each token sees the earlier ones' records.
        inputs, images = self.chunk_at(30, 3)
        prog = S.global_program(self.cfg, self.c, self.spec, self.mm, 30, chunk=3)
        run_engine(self, self.cfg, self.c, self.spec, self.mm, prog, inputs, {"m_ctx": images})

    def test_one_token_at_a_block_end(self) -> None:
        pos = 31                                     # four blocks eligible, two chosen; this token closes a block
        inputs, images = self.context_at(pos)
        self.assertEqual(inputs["k_rows"].shape[1], self.mm.local_window + self.cfg.top_blocks)
        self.assertNotEqual(images[0], images[1])
        prog = S.global_program(self.cfg, self.c, self.spec, self.mm, pos)
        run_engine(self, self.cfg, self.c, self.spec, self.mm, prog, inputs, {"m_ctx": images})

    def test_one_token_over_the_hpi_devices(self) -> None:
        pos = 31
        inputs, images = self.context_at(pos)
        prog = S.global_program(self.cfg, self.c, self.spec, self.mm, pos)
        run_engine(self, self.cfg, self.c, self.spec, self.mm, prog, inputs, {"m_ctx": images}, ndev=4)

    def test_a_stream_of_two_contexts(self) -> None:
        programs, inputs, memory = [], {}, {}
        for token, pos in ((0, 31), (1, 20)):
            own, images = self.context_at(pos)
            inputs.update({f"{k}@{token}": v for k, v in own.items()})
            memory[f"m_ctx@{token}"] = images
            programs.append(S.retarget(S.global_program(self.cfg, self.c, self.spec, self.mm, pos), token))
        run_engine(self, self.cfg, self.c, self.spec, self.mm, S.interleave(programs), inputs, memory)


@unittest.skipUnless(os.environ.get("FABRIC_FULL_SIZE") and shutil.which("iverilog"), "set FABRIC_FULL_SIZE=1 for the full-size run")
class FullSizeEngineTest(unittest.TestCase):
    """One token of the 9B recurrent layer through the engine at the design's
    own geometry: 4096 wide, 834 tiles of 4096 x 64, 32 heads of 128 x 128
    state, with the tiles' behavioural columns.  Minutes of Icarus, so it
    runs only when asked for."""

    def test_full_size_recurrent_token(self) -> None:
        import torch
        from fixed_llm_poc import ASICDecoderLayer, ASICLMConfig
        cfg = ASICLMConfig.qwen3_5_9b()
        torch.manual_seed(0)
        t0 = time.time()
        layer = ASICDecoderLayer(cfg, 0)
        w = {k: v.detach().double().numpy() for k, v in layer.state_dict().items()}
        spec, mm = TileSpec(), MemoryMap.from_config(cfg)
        rng = np.random.default_rng(9)
        nv, hk, hv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
        conv_dim = layer.linear_attn.conv_dim
        xs = [rng.standard_normal(cfg.hidden_size) * 2.0 for _ in range(3)]
        s, hist, runs = np.zeros((nv, hk, hv)), np.zeros((conv_dim, cfg.linear_conv_kernel - 1)), []
        for x in xs:
            r = L.recurrent_layer_float(w, cfg, x, s, hist)
            s, hist = r["s_next"], r["hist_next"]
            runs.append(r)
        cal = L.calibrate(runs, L.RECURRENT_CAL_KEYS)
        cal["x2"] = max(cal["x2"], max(float(np.abs(x).max()) for x in xs))
        c = L.compile_recurrent_layer(w, cfg, spec, cal)
        prog = S.recurrent_program(cfg, c, spec, mm)
        # The context after one token, then the second on the engine.
        s_i = np.zeros((nv, hk, hv), dtype=np.int64)
        sc_i = np.tile([L.ONE_U, 0, 0, 0], (nv, 1)).astype(np.int64)
        hist_i = np.zeros((conv_dim, cfg.linear_conv_kernel - 1), dtype=np.int64)
        r = L.recurrent_layer_int(c, cfg, spec, np.rint(xs[0] / c.s_h).astype(np.int64), s_i, hist_i, scale=sc_i)
        inputs = {"x": np.rint(xs[1] / c.s_h).astype(np.int64), "s_mem": r["s_next"], "scale_mem": r["scale_next"], "hist_mem": r["hist_next"]}
        print(f"\nfull size: layer built and compiled in {time.time() - t0:.0f} s, {len(prog)} steps")
        cycles = run_engine(self, cfg, c, spec, mm, prog, inputs, model_tiles=True, log=print)
        print(f"full size: {cycles} engine cycles for one token (the timing model said {S.schedule(prog).cycles})")
        self.assertGreater(cycles, S.schedule(prog).cycles // 2)


if __name__ == "__main__":
    unittest.main()
