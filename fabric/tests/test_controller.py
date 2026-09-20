import unittest

import numpy as np

from fabric import controller as C
from fabric.layer import FF


def fake_model(rng, vocab=64, d=16):
    """A stateless stand-in for the ring: the embedding, an identity layer die,
    and two head dies over a random LM head, each answering for its half."""
    emb = C.EmbeddingTable(rng.integers(-2000, 2000, size=(vocab, d)).astype(np.int16))
    w = rng.standard_normal((vocab, d)) * 0.02
    half = vocab // 2

    def logits_of(hidden):
        return C.to_fixed(w @ (hidden / 256.0))

    def head(die):
        def run(item):
            return C.head_list(die, logits_of(item.hidden)[die * half:(die + 1) * half], 8, die * half)
        return run
    return emb, logits_of, head


class PacketTest(unittest.TestCase):
    def test_round_trip_and_crc(self):
        rng = np.random.default_rng(1)
        hidden = rng.integers(-32768, 32767, size=16).astype(np.int16)
        item = C.WorkItem(3, 77, hidden, C.FLAG_SAMPLE | C.FLAG_FIRST)
        packet = C.pack_item(item)
        self.assertEqual(len(packet), C.HEADER_BYTES + 32)
        back, lists = C.unpack_item(packet, 16)
        self.assertEqual((back.context, back.position, back.flags), (3, 77, item.flags))
        self.assertTrue(np.array_equal(back.hidden, hidden.astype(np.int64)))
        self.assertEqual(lists, [])
        bad = bytearray(packet)
        bad[C.HEADER_BYTES + 5] ^= 0x10
        with self.assertRaises(ValueError):
            C.unpack_item(bytes(bad), 16)

    def test_head_dies_append_under_the_crc(self):
        rng = np.random.default_rng(2)
        hidden = rng.integers(-100, 100, size=16).astype(np.int16)
        packet = C.pack_item(C.WorkItem(1, 5, hidden))
        l0 = C.HeadList(0, np.array([4, 9], dtype=np.uint32), np.array([1000, -20], dtype=np.int32), 1234)
        l1 = C.HeadList(1, np.array([40, 33, 35], dtype=np.uint32), np.array([900, 800, 700], dtype=np.int32), -5)
        packet = C.append_head_list(C.append_head_list(packet, l0), l1)
        back, lists = C.unpack_item(packet, 16)
        self.assertTrue(np.array_equal(back.hidden, hidden.astype(np.int64)))
        self.assertEqual([hl.die for hl in lists], [0, 1])
        self.assertEqual(lists[1].lse, -5)
        self.assertTrue(np.array_equal(lists[1].rows, l1.rows))
        self.assertTrue(np.array_equal(lists[0].logits, l0.logits))
        self.assertEqual(len(packet), C.HEADER_BYTES + 32 + 2 * C.LIST_HEADER.size + 5 * C.ENTRY.size)


class MergeTest(unittest.TestCase):
    def test_merged_lists_are_the_global_top_and_the_lse_is_the_whole(self):
        rng = np.random.default_rng(3)
        logits = C.to_fixed(rng.standard_normal(200) * 3)
        lists = [C.head_list(0, logits[:100], 16, 0), C.head_list(1, logits[100:], 16, 100)]
        rows, merged, total = C.merge_lists(lists)
        top = np.argsort(-logits, kind="stable")[:16]
        self.assertEqual(list(rows[:16]), list(top))              # the 16 best of both halves lead the merge
        self.assertTrue(np.all(np.diff(merged) <= 0))
        self.assertLessEqual(abs(total - C.lse_fixed(logits)), 1)  # one LSB of rounding at most


class SampleTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        logits = C.to_fixed(rng.standard_normal(40) * 2)
        self.rows, self.logits, _ = C.merge_lists([C.head_list(0, logits[:20], 20, 0), C.head_list(1, logits[20:], 20, 20)])

    def test_top_one_and_cold_are_the_argmax(self):
        self.assertEqual(C.sample(self.rows, self.logits, C.SamplingParams(top_k=1), 0xDEADBEEF)[0], int(self.rows[0]))
        cold = C.SamplingParams.of(temperature=1e-4)
        self.assertEqual(C.sample(self.rows, self.logits, cold, 0xDEADBEEF)[0], int(self.rows[0]))

    def test_the_word_decides(self):
        p = C.SamplingParams.of(temperature=1.0, top_k=40)
        a = C.sample(self.rows, self.logits, p, 12345)
        self.assertEqual(a, C.sample(self.rows, self.logits, p, 12345))
        seen = {C.sample(self.rows, self.logits, p, int(r))[0] for r in np.random.default_rng(5).integers(0, 1 << 32, 400)}
        self.assertGreater(len(seen), 5)

    def test_frequencies_follow_the_weights(self):
        p = C.SamplingParams.of(temperature=1.0, top_k=8)
        w = C.weights(self.logits[:8], p.inv_t).astype(np.float64)
        counts = np.zeros(8)
        for r in np.random.default_rng(6).integers(0, 1 << 32, 20000):
            counts[C.sample(self.rows, self.logits, p, int(r))[1]] += 1
        expect = w / w.sum() * 20000
        self.assertLess(np.abs(counts - expect).max() / np.sqrt(expect.max()), 4.0)

    def test_top_p_keeps_the_head_of_the_mass(self):
        p = C.SamplingParams.of(temperature=1.0, top_k=40, top_p=0.5)
        w = C.weights(self.logits, p.inv_t)
        cum = np.cumsum(w)
        keep = int(np.searchsorted(cum * 0x10000, p.top_p * cum[-1], side="left")) + 1
        picked = {C.sample(self.rows, self.logits, p, int(r))[1] for r in np.random.default_rng(7).integers(0, 1 << 32, 2000)}
        self.assertLessEqual(max(picked), keep - 1)


class ControllerTest(unittest.TestCase):
    def test_one_context_samples_exactly_what_the_model_says(self):
        rng = np.random.default_rng(8)
        emb, logits_of, head = fake_model(rng)
        ring = C.Ring(3, lambda item: item.hidden, [head(0), head(1)])
        ctl = C.Controller(emb, ring, slots=4, seed=9)
        prompt = [5, 17, 3]
        cid = ctl.submit(prompt, max_new=6, params=C.SamplingParams.of(0.8, top_k=6))
        ctl.run()
        got = ctl.generated(cid)
        self.assertEqual(len(got), 6)
        # The same draws, by hand: the item's token is the last prompt token, then each sampled one.
        draws = np.random.default_rng(9)
        tokens = list(prompt)
        for i in range(6):
            logits = logits_of(emb.lookup(tokens[-1]))
            lists = [C.head_list(0, logits[:32], 8, 0), C.head_list(1, logits[32:], 8, 32)]
            rows, merged, total = C.merge_lists(lists)
            token, j = C.sample(rows, merged, C.SamplingParams.of(0.8, top_k=6), int(draws.integers(0, 1 << 32)))
            tokens.append(token)
            self.assertAlmostEqual(ctl.contexts[cid].logprobs[i], C.logprob(int(merged[j]), total))
        self.assertEqual(got, tokens[3:])

    def test_contexts_share_the_ring_round_robin(self):
        rng = np.random.default_rng(10)
        emb, _, head = fake_model(rng)
        ring = C.Ring(4, lambda item: item.hidden, [head(0), head(1)])
        ctl = C.Controller(emb, ring, slots=8, seed=11)
        ids = [ctl.submit([1, 2], 5), ctl.submit([7], 5), ctl.submit([3, 4, 5], 5)]
        ctl.run()
        for cid in ids:
            self.assertEqual(len(ctl.generated(cid)), 5)
            self.assertTrue(all(0 <= t < 64 for t in ctl.generated(cid)))
        self.assertEqual(ctl.table.evicted, [])
        self.assertEqual(ctl.table.holder, {})                       # every slot given back
        self.assertLess(ctl.steps, 3 * (2 + 1 + 3 + 5) * 6 + 50)     # the ring kept more than one item in flight

    def test_a_short_table_evicts_and_the_evicted_start_over(self):
        rng = np.random.default_rng(12)
        emb, _, head = fake_model(rng)
        ring = C.Ring(2, lambda item: item.hidden, [head(0), head(1)])
        ctl = C.Controller(emb, ring, slots=1, seed=13)
        a, b = ctl.submit([1, 2, 3], 3), ctl.submit([4, 5], 3)
        ctl.run()
        self.assertGreater(len(ctl.table.evicted), 0)
        self.assertEqual(len(ctl.generated(a)), 3)
        self.assertEqual(len(ctl.generated(b)), 3)


import shutil
import subprocess
import tempfile
from pathlib import Path

RTL = Path(__file__).parents[1] / "rtl"
SOURCES = [RTL / name for name in ("fabric_vector.sv", "fabric_controller.sv")]


@unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog not installed")
class ControllerRtlTest(unittest.TestCase):
    """The gateware's arithmetic against the model: the sampler draws the same
    token from the same lists and word, the CRC is zlib's."""

    def check(self, top: str, emit) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            params = emit(work)
            args = [f"-P{top}.{name}={value}" for name, value in params.items()]
            subprocess.run(["iverilog", "-g2012", "-I", str(RTL), "-s", top, "-o", "sim.vvp", *args,
                            *map(str, SOURCES), str(RTL / f"{top}.sv")], cwd=work, check=True, capture_output=True, text=True)
            result = subprocess.run(["vvp", "sim.vvp"], cwd=work, check=True, capture_output=True, text=True)
        self.assertIn("PASS", result.stdout, result.stdout)

    def test_sampler(self) -> None:
        rng = np.random.default_rng(30)
        self.check("tb_sampler", lambda d: C.emit_sampler_vectors(d, rng, 32, 40))
        self.check("tb_sampler", lambda d: C.emit_sampler_vectors(d, rng, 8, 30))

    def test_crc32(self) -> None:
        rng = np.random.default_rng(31)
        self.check("tb_crc32", lambda d: C.emit_crc_vectors(d, rng, 24))


if __name__ == "__main__":
    unittest.main()
