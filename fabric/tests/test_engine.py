import os
import re
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
SOURCES = [RTL / name for name in ("fabric_sram.sv", "fabric_vector.sv", "fabric_norm.sv", "fabric_recurrent.sv", "fabric_ffn.sv", "fabric_attention.sv",
                                   "fabric_memory.sv", "fabric_tile.sv", "fabric_sequencer.sv", "fabric_engine.sv",
                                   "fabric_phy.sv", "fabric_cdc.sv", "fabric_hpi.sv", "fabric_controller.sv", "fabric_ring.sv",
                                   "fabric_die_link.sv", "tb_layer_engine.sv")]


def run_engine(case: unittest.TestCase, cfg, c, spec, mm, steps: list[S.Step], inputs: dict, memory=None, ndev: int = 0,
               model_tiles: bool = False, log=None, first: bool = False, base_page: int = 0, ring: bool = False,
               position: int = 0) -> int:
    """Emit, simulate and check one program; returns the engine's cycle count.  With ``ndev`` the memory is the HPI path,
    with ``model_tiles`` the tiles' behavioural columns (full-size runs), with ``first`` the token is FIRST, with ``ring``
    the tokens come and go as packets through the die's ring link."""
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        t0 = time.time()
        run = E.EngineRun(work, cfg, c, spec, mm, steps, inputs, memory, ndev, model_tiles, first, base_page, ring, position)
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
    took = int(passed.split(" in ")[1].split()[0])
    # The timing model is the engine's, command for command: every adapter's
    # latency and every lane count in `sequencer.Timing` is the RTL's own, so
    # the two cycle counts are equal and not merely close.  This is the
    # assertion that keeps them from drifting -- for a long time nothing
    # compared them and the model had settled at about half the real count.
    # Its port is the testbench's memory, a beat a cycle after a short
    # latency; with the HPI devices behind the port the memory steps are the
    # PSRAM's and the model has no calibration for them.
    if not ndev:
        case.assertEqual(took, S.schedule(steps).cycles, passed)
    return took


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


class PortMapTest(unittest.TestCase):
    """The crossbar's port face is folded in Python and wired in Verilog, and
    the fold is only sound if the two number the ports the same way.  Reading
    the numbering out of the RTL rather than restating it is the point: a port
    map that has drifted folds two live ports onto one."""

    def rtl_ports(self, prefix: str, tmax: int) -> dict[str, int]:
        text = (RTL / "fabric_engine.sv").read_text()
        names: dict[str, int] = {"TMAX": tmax}
        for statement in re.findall(r"localparam int (" + prefix + r"_NORM\b.*?);", text, re.S):
            for assignment in statement.split(","):
                name, expression = assignment.split("=", 1)
                names[name.strip()] = eval(expression.strip(), {"__builtins__": {}}, names)  # noqa: S307 - our own RTL
        return names

    def test_the_lanes_are_the_ones_the_rtl_has(self) -> None:
        # The timing model's lane counts are the RTL's own, not a second
        # estimate of them: the programs' beat counts come from the same
        # fields, and a model that thinks the units are wider than they are
        # is what made it half the engine's real cycle count.
        text = (RTL / "fabric_engine.sv").read_text()
        localparams = re.search(r"localparam int NU = .*?;", text, re.S).group(0)
        t = S.Timing()
        for name, value in (("NL", t.lanes), ("CL", t.l_conv)):
            self.assertIn(f"{name} = {value}", localparams, name)
        # ATT_L follows the head, by the same rule on both sides.
        from fixed_llm_poc import ASICLMConfig, tiny_config
        for cfg in (tiny_config(), ASICLMConfig.qwen3_5_9b()):
            lanes = t.head_lanes(cfg.head_dim)
            self.assertEqual(cfg.head_dim % lanes, 0)
            self.assertLessEqual(lanes * 8, 128)                 # a buffer beat

    def test_the_port_map_is_the_one_the_rtl_wires(self) -> None:
        for chunk in (1, 2, 3):
            for prefix, table, total_name in (("R", E.RD_PORT_MAP, "NR"), ("W", E.WR_PORT_MAP, "NW")):
                rtl = self.rtl_ports(prefix, chunk)
                index, total = E._port_index(table, chunk)
                self.assertEqual(total, rtl[total_name], (prefix, chunk))
                for unit, _, _ in table:
                    name = f"{prefix}_{'SWIGLU' if unit == 'swiglu' else unit.upper()}"
                    self.assertEqual(index[(unit, 0, 0)], rtl[name], (name, chunk))


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

    def test_the_same_program_serves_any_slot(self) -> None:
        # The program carries its memory operands as offsets in the token's
        # slot, and the engine adds the slot's page: the image is the same
        # byte for byte wherever the context lives, and the token is exact there.
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            images = []
            for directory, page in ((a, 0), (b, 37)):
                run = E.EngineRun(Path(directory), self.cfg, self.c, self.spec, self.mm, self.prog, self.context_after(2),
                                  base_page=page)
                self.assertEqual(run.params["SLOT0"], page)
                images.append((Path(directory) / "program.hex").read_text())
            self.assertEqual(images[0], images[1])
        self.run_engine_at(37)

    def run_engine_at(self, page: int) -> None:
        run_engine(self, self.cfg, self.c, self.spec, self.mm, self.prog, self.context_after(2), base_page=page)

    def test_a_first_token_over_a_used_slot(self) -> None:
        # FIRST: a new context's first token in a slot that holds another's
        # state and history.  It must start from zero -- the integer layer's
        # fresh state -- whatever the slot held, and write that back.
        cfg, nv, hk, hv = self.cfg, self.cfg.linear_num_value_heads, self.cfg.linear_key_head_dim, self.cfg.linear_value_head_dim
        inputs = self.context_after(2)
        inputs["x"] = np.rint(self.xs[0] / self.c.s_h).astype(np.int64)
        self.assertGreater(np.abs(inputs["s_mem"]).max(), 0)
        self.assertGreater(np.abs(inputs["hist_mem"]).max(), 0)
        fresh = L.recurrent_layer_int(self.c, cfg, self.spec, inputs["x"], np.zeros((nv, hk, hv), dtype=np.int64),
                                      np.zeros_like(inputs["hist_mem"]), scale=np.tile([L.ONE_U, 0, 0, 0], (nv, 1)).astype(np.int64))
        prog = S.recurrent_program(cfg, self.c, self.spec, self.mm, first=True)
        env = S.run_program(prog, {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in inputs.items()})
        np.testing.assert_array_equal(env["x2"], fresh["x2"])
        np.testing.assert_array_equal(env["s_mem"], fresh["s_next"])
        np.testing.assert_array_equal(env["hist_mem"], fresh["hist_next"])
        run_engine(self, cfg, self.c, self.spec, self.mm, prog, inputs, first=True)

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

    def test_two_contexts_through_the_ring(self) -> None:
        # The die as the ring sees it: two packets in, a lane each, from
        # contexts in slots other than the first; the link writes their
        # vectors into the buffer, starts the stream's program with their
        # slots, and sends their outputs on as the model's packets.
        two = S.stream(self.prog, 2)
        inputs = {}
        for token, tokens in ((0, 3), (1, 1)):
            inputs.update({f"{k}@{token}": v for k, v in self.context_after(tokens).items()})
        run_engine(self, self.cfg, self.c, self.spec, self.mm, two, inputs, base_page=37, ring=True, position=3)


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

    def test_a_chunk_through_the_ring(self) -> None:
        # A prompt's chunk as one packet of three tokens: its three vectors
        # into the buffer, the chunk's program, and three out.
        inputs, images = self.chunk_at(30, 3)
        prog = S.global_program(self.cfg, self.c, self.spec, self.mm, 30, chunk=3)
        run_engine(self, self.cfg, self.c, self.spec, self.mm, prog, inputs, {"m_ctx": images}, ring=True, position=30)

    def test_a_first_token_over_a_used_slot(self) -> None:
        # FIRST at position 0 in a context image another context left behind:
        # its block sums are garbage, and must be taken as zero, and so are
        # the window records the token never reads.  The sums are written back
        # whole; the rest stays as it was.
        inputs, (before, after) = self.context_at(0)
        rng = np.random.default_rng(9)
        before, after = bytearray(before), bytearray(after)
        regions = self.mm.regions()
        off, size = regions["sums0"]
        before[off:off + size] = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
        woff, _ = regions["window0"]
        rec = self.mm.kv_record_bytes
        for n in range(self.cfg.num_key_value_heads):
            for p in range(1, self.mm.local_window):
                a = woff + (n * self.mm.local_window + p) * rec
                junk = rng.integers(0, 256, rec, dtype=np.uint8).tobytes()
                before[a:a + rec] = junk
                after[a:a + rec] = junk
        prog = S.global_program(self.cfg, self.c, self.spec, self.mm, 0, first=True)
        run_engine(self, self.cfg, self.c, self.spec, self.mm, prog, inputs, {"m_ctx": (bytes(before), bytes(after))}, first=True)

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
