import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fabric import layer as L
from fabric import sequencer as S
from fabric.memory import MemoryMap
from fabric.tile import TileSpec

RTL = Path(__file__).parents[1] / "rtl"


def steps_of(*specs) -> list[S.Step]:
    return [S.Step(name, "norm", 0, tuple(src), tuple(dst), 1) for name, src, dst in specs]


class LinkTest(unittest.TestCase):
    """The dependency rules that turn buffer names into the program's edges."""

    def test_read_after_write_and_write_after_read(self) -> None:
        steps = steps_of(("w", (), ("a",)), ("r", ("a",), ("b",)), ("w2", (), ("a",)), ("r2", ("a",), ()))
        S.link(steps)
        self.assertEqual(steps[1].deps, [0])          # reads a after its writer
        self.assertEqual(steps[2].deps, [0, 1])       # overwrites a after its writer and its reader
        self.assertEqual(steps[3].deps, [2])          # reads the new a only

    def test_contributions_to_a_whole_vector(self) -> None:
        steps = steps_of(("p0", (), ("+y",)), ("p1", (), ("+y",)), ("all", ("y",), ("z",)), ("again", ("y",), ()),
                         ("p0b", (), ("+y",)), ("p1b", (), ("+y",)), ("all2", ("y",), ()))
        S.link(steps)
        self.assertEqual(steps[1].deps, [])           # contributions do not wait for each other
        self.assertEqual(steps[2].deps, [0, 1])       # the whole waits for every contribution
        self.assertEqual(steps[4].deps, [2, 3])       # the next version waits for the readers of the last
        self.assertEqual(steps[5].deps, [])
        self.assertEqual(steps[6].deps, [4, 5])

    def test_encoding_carries_ids_and_contribution_bits(self) -> None:
        steps = steps_of(("w", (), ("a",)), ("c", ("a",), ("+y",)), ("r", ("a", "y"), ()))
        S.link(steps)
        words = S.encode(steps)
        ids = S.buffer_ids(steps)
        base = 64 + 4 * S.ADDR_BITS                                          # the ids follow the four address operands
        self.assertEqual((words[1] >> base) & 0xFF, ids["a"])                # consumed
        self.assertEqual((words[1] >> (base + 8)) & 0xFF, 0xFF)              # no second consumed buffer
        self.assertEqual((words[1] >> (base + 8 * S.MAX_CONSUME)) & 0xFF, ids["y"])            # produced
        self.assertEqual((words[1] >> (base + 8 * (S.MAX_CONSUME + 2))) & 0b11, 0b01)         # as a contribution
        self.assertEqual((words[0] >> (base + 8 * (S.MAX_CONSUME + 2))) & 0b11, 0)
        self.assertEqual((words[2] >> 8) & 1, 1)                           # the last step


class ScheduleTest(unittest.TestCase):
    def test_in_order_issue_engine_reuse_and_dependencies(self) -> None:
        steps = [S.Step("a", "norm", 0, (), ("x",), 10), S.Step("b", "norm", 1, (), ("y",), 5),
                 S.Step("c", "norm", 0, (), ("z",), 3), S.Step("d", "conv", 0, ("x", "y"), ("w",), 4)]
        S.link(steps)
        sched = S.schedule(steps)
        # A step's cycles are its whole span, issue to completion, as the
        # engine's own spans measure it: a issues at 0, completes at 10, and
        # its drain is picked there.  The drain is registered and the counters
        # are registers, so it applies at 11 and is readable at 12 -- c takes
        # engine 0 at 11, when the port is free because a's ids were
        # registered out of its slot at the pick, and d waits for x to be
        # readable at 12.
        self.assertEqual(sched.issue, [0, 1, 11, 12])
        self.assertEqual(sched.end, [9, 5, 13, 15])          # the last cycle each was working
        self.assertEqual(sched.release, [10, 6, 14, 16])     # and the cycle its buffers came back
        self.assertEqual(sched.last_done, 16)
        self.assertEqual(sched.cycles, 18)                   # done is two after the last drain

    def test_full_size_layers_are_memory_bound(self) -> None:
        from fixed_llm_poc import ASICLMConfig
        cfg = ASICLMConfig.qwen3_5_9b()
        mm = MemoryMap.from_config(cfg)
        rec = S.recurrent_program(cfg, None, TileSpec(), mm)
        glob = S.global_program(cfg, None, TileSpec(), mm, mm.context_tokens - 1)
        for steps in (rec, glob):
            sched = S.schedule(steps)
            # The port is still the busiest unit by far, though the mover's two
            # beats a transfer took a recurrent token from two thirds of it to
            # just under three fifths.
            self.assertGreater(sched.busy("mem") / sched.cycles, 0.55)
            self.assertEqual(max(S.UNITS, key=lambda u: sched.busy(u) / S.UNITS[u][1]), "mem")
            self.assertEqual(sched.busy("tiles"), sum(s.cycles for s in steps if s.unit == "tiles"))
        self.assertEqual(len([s for s in rec if s.unit == "tiles"]), 4)              # four passes
        self.assertEqual(len([s for s in rec if s.name.startswith("delta")]), cfg.linear_num_value_heads)
        self.assertEqual(sum(s.nbytes for s in rec), 2 * mm.state_bytes + 2 * mm.hist_bytes)     # the int8 state and its scales
        self.assertLess(sum(s.nbytes for s in rec), 1.1 * 2 ** 20)
        self.assertIn("stream of tokens", S.report_markdown(cfg, mm, 4095))

    def test_a_stream_of_tokens_runs_at_the_memory_port(self) -> None:
        from fixed_llm_poc import ASICLMConfig
        cfg = ASICLMConfig.qwen3_5_9b()
        mm = MemoryMap.from_config(cfg)
        for steps in (S.recurrent_program(cfg, None, TileSpec(), mm), S.global_program(cfg, None, TileSpec(), mm, mm.context_tokens - 1)):
            single = S.schedule(steps)
            interval = S.token_interval(steps)
            port = single.busy("mem")
            # The port bounds the stream.  Not a floor: a command's span is its
            # beats and its latency, and consecutive commands overlap the latency,
            # so the interval can come in a little under the sum of the spans.
            # The global layer's stream sits on it; since the mover moves two
            # beats a transfer the recurrent one no longer does -- its port time
            # halved, and the in-order issue of its passes and heads now leaves
            # the port idle for about a sixth of the interval.
            self.assertGreater(interval, 0.97 * port)
            self.assertLess(interval, 1.25 * port)
            self.assertLess(interval, 0.85 * single.cycles)             # below one token at a time
            two = S.stream(steps, 2)
            self.assertEqual(len(two), 2 * len(steps))
            self.assertLessEqual(len(S.buffer_ids(two)), S.MAX_IDS)
            # The merge keeps each token's own order and every dependency it had.
            for token in (0, 1):
                own = [s.name for s in two if s.token == token]
                self.assertEqual(own, [s.name for s in steps])
            self.assertEqual(S.schedule(two).cycles, S.schedule(S.stream(steps, 2)).cycles)

    def test_a_stream_never_reads_into_a_slot_another_token_holds(self) -> None:
        # The state slots are shared by every token in a stream.  From a
        # token's read into a slot to its write-back the slot is that token's:
        # anything another token does to it in between is the first token's
        # update running on the wrong context's state.  Only the full-size
        # merge ever did it -- three times a token in a stream of three -- so
        # the small configurations' bit-exact streams could not see it.
        from fixed_llm_poc import ASICLMConfig
        cfg = ASICLMConfig.qwen3_5_9b()
        mm = MemoryMap.from_config(cfg)
        merged = S.stream(S.recurrent_program(cfg, None, TileSpec(), mm), 3)
        held: dict[str, int] = {}
        for step in merged:
            for name in (*step.src, *step.dst):
                name = S._plain(name)
                if not name.startswith(S.SHARED_PREFIX):
                    continue
                if name in held:
                    self.assertEqual(held[name], step.token, f"{step.name} of token {step.token} in a slot token {held[name]} holds")
                if step.unit == "mem" and name in step.dst:
                    held[name] = step.token
                elif step.unit == "mem" and name in step.src:
                    del held[name]


class ProgramTest(unittest.TestCase):
    """The programs reproduce the integer layers bit for bit on the tiny geometry."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
            from fixed_llm_poc import ASICDecoderLayer, tiny_config
        except ImportError:  # pragma: no cover
            raise unittest.SkipTest("PyTorch not installed")
        cls.cfg = tiny_config()
        torch.manual_seed(0)
        cls.layers = {idx: ASICDecoderLayer(cls.cfg, idx) for idx in (0, 3)}
        cls.spec = TileSpec(rows=cls.cfg.hidden_size, cols=16)
        cls.mm = MemoryMap.from_config(cls.cfg)

    @staticmethod
    def weights(layer) -> dict:
        return {k: v.detach().double().numpy() for k, v in layer.state_dict().items()}

    def test_recurrent_program(self) -> None:
        cfg, spec = self.cfg, self.spec
        w = self.weights(self.layers[0])
        rng = np.random.default_rng(3)
        nv, hk, hv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
        conv_dim = self.layers[0].linear_attn.conv_dim
        xs = [rng.standard_normal(cfg.hidden_size) * 2.0 for _ in range(6)]
        s, hist, runs = np.zeros((nv, hk, hv)), np.zeros((conv_dim, cfg.linear_conv_kernel - 1)), []
        for x in xs:
            r = L.recurrent_layer_float(w, cfg, x, s, hist)
            s, hist = r["s_next"], r["hist_next"]
            runs.append(r)
        cal = L.calibrate(runs, L.RECURRENT_CAL_KEYS)
        cal["x2"] = max(cal["x2"], max(float(np.abs(x).max()) for x in xs))
        c = L.compile_recurrent_layer(w, cfg, spec, cal)
        prog = S.recurrent_program(cfg, c, spec, self.mm)
        s_i = np.zeros((nv, hk, hv), dtype=np.int64)                  # the int8 state and its per-head scale beats
        sc_i = np.tile([L.ONE_U, 0, 0, 0], (nv, 1)).astype(np.int64)
        hist_i = np.zeros((conv_dim, cfg.linear_conv_kernel - 1), dtype=np.int64)
        s_mem, sc_mem, hist_mem = s_i.copy(), sc_i.copy(), hist_i.copy()
        for x in xs:
            xi = np.rint(x / c.s_h).astype(np.int64)
            ri = L.recurrent_layer_int(c, cfg, spec, xi, s_i, hist_i, scale=sc_i)
            s_i, sc_i, hist_i = ri["s_next"], ri["scale_next"], ri["hist_next"]
            env = S.run_program(prog, {"x": xi, "s_mem": s_mem, "scale_mem": sc_mem, "hist_mem": hist_mem})
            s_mem, sc_mem, hist_mem = env["s_mem"], env["scale_mem"], env["hist_mem"]
            np.testing.assert_array_equal(env["x2"], ri["x2"])
            np.testing.assert_array_equal(env["mixer"], ri["mixer"])
            np.testing.assert_array_equal(s_mem, ri["s_next"])
            np.testing.assert_array_equal(np.asarray(sc_mem), ri["scale_next"])
            np.testing.assert_array_equal(hist_mem, ri["hist_next"])
        # Two contexts' tokens as one stream: each context's result is its own layer's.
        two = S.stream(prog, 2)
        env = {}
        expect = []
        for token in (0, 1):
            xi = np.rint(xs[token] / c.s_h).astype(np.int64)
            s0 = np.zeros((nv, hk, hv), dtype=np.int64)
            sc0 = np.tile([L.ONE_U, 0, 0, 0], (nv, 1)).astype(np.int64)
            h0 = np.zeros((conv_dim, cfg.linear_conv_kernel - 1), dtype=np.int64)
            env.update({f"x@{token}": xi, f"s_mem@{token}": s0.copy(), f"scale_mem@{token}": sc0.copy(), f"hist_mem@{token}": h0.copy()})
            expect.append(L.recurrent_layer_int(c, cfg, spec, xi, s0, h0, scale=sc0))
        out = S.run_program(two, env)
        for token in (0, 1):
            np.testing.assert_array_equal(out[f"x2@{token}"], expect[token]["x2"])
            np.testing.assert_array_equal(out[f"s_mem@{token}"], expect[token]["s_next"])
            np.testing.assert_array_equal(np.asarray(out[f"scale_mem@{token}"]), expect[token]["scale_next"])
        # A chunk of three tokens (prefill) is three tokens in turn, with each head's state read and written once.
        chunk = S.recurrent_program(cfg, c, spec, self.mm, chunk=3)
        self.assertEqual(len([s for s in chunk if s.unit == "tiles"]), 4)
        self.assertEqual(len([s for s in chunk if s.name.startswith("dma.s_")]), 2 * nv)
        self.assertEqual(len([s for s in chunk if s.name.startswith("delta")]), 3 * nv)
        s_i = np.zeros((nv, hk, hv), dtype=np.int64)
        sc_i = np.tile([L.ONE_U, 0, 0, 0], (nv, 1)).astype(np.int64)
        hist_i = np.zeros((conv_dim, cfg.linear_conv_kernel - 1), dtype=np.int64)
        env = S.run_program(chunk, {"x": np.stack([np.rint(x / c.s_h).astype(np.int64) for x in xs[:3]]), "s_mem": s_i.copy(),
                                    "scale_mem": sc_i.copy(), "hist_mem": hist_i.copy()})
        for k, x in enumerate(xs[:3]):
            ri = L.recurrent_layer_int(c, cfg, spec, np.rint(x / c.s_h).astype(np.int64), s_i, hist_i, scale=sc_i)
            s_i, sc_i, hist_i = ri["s_next"], ri["scale_next"], ri["hist_next"]
            np.testing.assert_array_equal(env["x2"][k], ri["x2"])
        np.testing.assert_array_equal(env["s_mem"], s_i)
        np.testing.assert_array_equal(np.asarray(env["scale_mem"]), sc_i)
        np.testing.assert_array_equal(env["hist_mem"], hist_i)
        # The int16 program is the same list with the plain state.
        mm16 = MemoryMap.from_config(cfg, state_bits=16)
        prog16 = S.recurrent_program(cfg, c, spec, mm16)
        s16 = np.zeros((nv, hk, hv), dtype=np.int64)
        xi = np.rint(xs[0] / c.s_h).astype(np.int64)
        ri = L.recurrent_layer_int(c, cfg, spec, xi, s16, np.zeros_like(hist_i))
        env = S.run_program(prog16, {"x": xi, "s_mem": s16.copy(), "hist_mem": np.zeros_like(hist_i)})
        np.testing.assert_array_equal(env["x2"], ri["x2"])

    def test_global_program(self) -> None:
        cfg, spec = self.cfg, self.spec
        w = self.weights(self.layers[3])
        rng = np.random.default_rng(4)
        nkv, hd = cfg.num_key_value_heads, cfg.head_dim
        xs = [rng.standard_normal(cfg.hidden_size) * 2.0 for _ in range(6)]
        kf, vf, runs = [], [], []
        for pos, x in enumerate(xs):
            own = L.global_layer_float(w, cfg, x, pos, np.zeros((nkv, 1, hd)), np.zeros((nkv, 1, hd)))
            kf.append(own["k"])
            vf.append(own["v"])
            runs.append(L.global_layer_float(w, cfg, x, pos, np.stack(kf, axis=1), np.stack(vf, axis=1)))
        cal = L.calibrate(runs, L.GLOBAL_CAL_KEYS)
        cal["x2"] = max(cal["x2"], max(float(np.abs(x).max()) for x in xs))
        c = L.compile_global_layer(w, cfg, spec, cal)
        ki, vi = [], []
        for pos, x in enumerate(xs):
            xi = np.rint(x / c.s_h).astype(np.int64)
            zero = np.zeros((nkv, 1, hd), dtype=np.int64)
            own = L.global_layer_int(c, cfg, spec, xi, pos, zero, zero)
            ki.append(own["k"])
            vi.append(own["v"])
            k_rows, v_rows = np.stack(ki, axis=1), np.stack(vi, axis=1)
            ri = L.global_layer_int(c, cfg, spec, xi, pos, k_rows, v_rows)
            env = S.run_program(S.global_program(cfg, c, spec, self.mm, pos), {"x": xi, "k_rows": k_rows, "v_rows": v_rows})
            np.testing.assert_array_equal(env["x2"], ri["x2"])
            for n in range(nkv):
                np.testing.assert_array_equal(env[f"k[{n}]"], ri["k"][n])
                np.testing.assert_array_equal(env[f"att[{n}]"], ri["att"].reshape(cfg.num_attention_heads, hd)[n * 2:(n + 1) * 2])
        # A chunk of the last three positions: each token over the rows it would see, the passes shared.
        chunk = S.global_program(cfg, c, spec, self.mm, 3, chunk=3)
        self.assertEqual(len([s for s in chunk if s.unit == "tiles"]), 4)
        self.assertEqual(len([s for s in chunk if s.name.startswith("mem.append")]), 3)
        xi = np.stack([np.rint(x / c.s_h).astype(np.int64) for x in xs[3:6]])
        rows_k = [np.stack(ki[:p + 1], axis=1) for p in range(3, 6)]
        rows_v = [np.stack(vi[:p + 1], axis=1) for p in range(3, 6)]
        env = S.run_program(chunk, {"x": xi, "k_rows": rows_k, "v_rows": rows_v})
        for k, pos in enumerate(range(3, 6)):
            ri = L.global_layer_int(c, cfg, spec, xi[k], pos, rows_k[k], rows_v[k])
            np.testing.assert_array_equal(env["x2"][k], ri["x2"])


@unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog not installed")
class SequencerRtlTest(unittest.TestCase):
    """The controller runs each program over stub units in exactly the cycles
    the model predicts, and the trace respects every dependency and engine."""

    def run_program(self, steps: list[S.Step]) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            params = S.emit_program(work, steps)
            args = [f"-Ptb_sequencer.{name}={value}" for name, value in params.items()]
            subprocess.run(["iverilog", "-g2012", "-I", str(RTL), "-s", "tb_sequencer", "-o", "sim.vvp", *args,
                            str(RTL / "fabric_sram.sv"), str(RTL / "fabric_sequencer.sv"), str(RTL / "tb_sequencer.sv")],
                           cwd=work, check=True, capture_output=True, text=True)
            out = subprocess.run(["vvp", "sim.vvp"], cwd=work, check=True, capture_output=True, text=True).stdout
            trace = (work / "trace.txt").read_text()
        self.assertIn("PASS", out, out)
        self.assertEqual(S.check_trace(steps, trace), [])

    def test_tiny_layers(self) -> None:
        from fixed_llm_poc import tiny_config
        cfg = tiny_config()
        mm = MemoryMap.from_config(cfg)
        self.run_program(S.recurrent_program(cfg, None, TileSpec(), mm))
        self.run_program(S.global_program(cfg, None, TileSpec(), mm, 5))

    def test_full_size_layers(self) -> None:
        from fixed_llm_poc import ASICLMConfig
        cfg = ASICLMConfig.qwen3_5_9b()
        mm = MemoryMap.from_config(cfg)
        self.run_program(S.recurrent_program(cfg, None, TileSpec(), mm))
        self.run_program(S.global_program(cfg, None, TileSpec(), mm, mm.context_tokens - 1))

    def test_streams_of_two_tokens(self) -> None:
        from fixed_llm_poc import ASICLMConfig, tiny_config
        cfg = tiny_config()
        mm = MemoryMap.from_config(cfg)
        self.run_program(S.stream(S.recurrent_program(cfg, None, TileSpec(), mm), 2))
        cfg = ASICLMConfig.qwen3_5_9b()
        mm = MemoryMap.from_config(cfg)
        self.run_program(S.stream(S.recurrent_program(cfg, None, TileSpec(), mm), 2))          # 346 steps, tags wrap
        self.run_program(S.stream(S.global_program(cfg, None, TileSpec(), mm, mm.context_tokens - 1), 2))


if __name__ == "__main__":
    unittest.main()
