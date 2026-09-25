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
        self.assertEqual(len(packet), C.HEADER_BYTES + 32 + C.TRAILER_BYTES)
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
        self.assertEqual(len(packet), C.HEADER_BYTES + 32 + 2 * C.LIST_HEADER.size + 5 * C.ENTRY.size + C.TRAILER_BYTES)

    def test_a_chunk_is_one_packet(self):
        # A chunk's vectors back to back, its token count in the header's top
        # byte; a head die appends to it as to any other.
        rng = np.random.default_rng(3)
        hidden = rng.integers(-32768, 32767, size=(5, 16)).astype(np.int16)
        packet = C.pack_item(C.WorkItem(2, 100, hidden, C.FLAG_SAMPLE))
        self.assertEqual(len(packet), C.HEADER_BYTES + 5 * 32 + C.TRAILER_BYTES)
        self.assertEqual(packet[11], 5)
        self.assertEqual(int.from_bytes(packet[8:11], "little"), 5 * 32)
        l0 = C.HeadList(0, np.array([4], dtype=np.uint32), np.array([7], dtype=np.int32), 3)
        back, lists = C.unpack_item(C.append_head_list(packet, l0), 16)
        self.assertEqual((back.position, back.tokens, back.last.position), (100, 5, 104))
        self.assertTrue(np.array_equal(back.hidden, hidden.astype(np.int64)))
        self.assertTrue(np.array_equal(back.last.hidden, hidden[-1].astype(np.int64)))
        self.assertEqual([hl.lse for hl in lists], [3])
        with self.assertRaises(ValueError):
            C.pack_item(C.WorkItem(2, 100, np.zeros((256, 16), dtype=np.int16)))


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


def recorded(ring, emb):
    """The ring, with every packet sent into it recorded."""
    sent, inject = [], ring.inject
    ring.inject = lambda packet: (sent.append(C.unpack_item(packet, emb.hidden)[0]), inject(packet))[1]
    return sent


class ControllerTest(unittest.TestCase):
    def test_one_slot_samples_exactly_what_the_model_says(self):
        rng = np.random.default_rng(8)
        emb, logits_of, head = fake_model(rng)
        ring = C.Ring(3, lambda item: item.hidden, [head(0), head(1)])
        ctl = C.Controller(emb, ring, slots=4, seed=9)
        prompt = [5, 17, 3]
        ctl.append(2, prompt, max_new=6, params=C.SamplingParams.of(0.8, top_k=6))
        ctl.run()
        got = ctl.slots[2].generated
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
            self.assertAlmostEqual(ctl.slots[2].logprobs[i], C.logprob(int(merged[j]), total))
        self.assertEqual(got, tokens[3:])

    def test_slots_share_the_ring_round_robin(self):
        rng = np.random.default_rng(10)
        emb, _, head = fake_model(rng)
        ring = C.Ring(4, lambda item: item.hidden, [head(0), head(1)])
        ctl = C.Controller(emb, ring, slots=8, seed=11)
        for slot, prompt in ((0, [1, 2]), (5, [7]), (3, [3, 4, 5])):
            ctl.append(slot, prompt, 5)
        ctl.run()
        for slot in (0, 5, 3):
            self.assertEqual(len(ctl.slots[slot].generated), 5)
            self.assertTrue(all(0 <= t < 64 for t in ctl.slots[slot].generated))
        self.assertLess(ctl.steps, 3 * (2 + 1 + 3 + 5) * 6 + 50)     # the ring kept more than one item in flight

    def test_a_prompt_fills_the_ring_by_itself(self):
        # A prompt's tokens are all known, so one slot's go in back to back
        # rather than one per trip round the ring: in order, the last of them
        # drawn from, and a sampled token sent only once it has been drawn.
        rng = np.random.default_rng(16)
        emb, _, head = fake_model(rng)
        ring = C.Ring(8, lambda item: item.hidden, [head(0), head(1)])
        sent = recorded(ring, emb)
        ctl = C.Controller(emb, ring, slots=4, seed=17)
        prompt = list(range(1, 41))
        ctl.append(0, prompt, max_new=3)
        ctl.run()
        self.assertEqual(len(ctl.slots[0].generated), 3)
        # The turn's last token is drawn and not sent: it goes in first next turn.
        self.assertEqual([s.position for s in sent], list(range(len(prompt) + 2)))
        self.assertEqual([s.position for s in sent if s.flags & C.FLAG_SAMPLE], [39, 40, 41])
        self.assertEqual([s.flags & C.FLAG_FIRST for s in sent][:2], [C.FLAG_FIRST, 0])
        # Forty prompt tokens through a ten-stage ring: about forty steps and a
        # trip, not forty trips; then each sampled token a trip of its own.
        self.assertLess(ctl.steps, len(prompt) + 4 * 10 + 5)
        self.assertEqual(emb.lookup(ctl.slots[0].generated[0]).tolist(), sent[40].hidden.tolist())
        self.assertEqual(ctl.slots[0].pending, [ctl.slots[0].generated[-1]])

    def test_a_turn_goes_on_from_where_the_slot_is(self):
        # The next turn carries only the new tokens; the last token drawn goes
        # in first, and nothing is FIRST.
        rng = np.random.default_rng(22)
        emb, _, head = fake_model(rng)
        ring = C.Ring(2, lambda item: item.hidden, [head(0), head(1)])
        sent = recorded(ring, emb)
        ctl = C.Controller(emb, ring, slots=2, seed=23)
        ctl.append(1, [4, 5, 6], 2)
        ctl.run()
        drawn = ctl.slots[1].generated[-1]
        sent.clear()
        ctl.append(1, [9, 10], 2)
        ctl.run()
        self.assertEqual([s.position for s in sent], [4, 5, 6, 7])
        self.assertEqual(emb.lookup(drawn).tolist(), sent[0].hidden.tolist())
        self.assertFalse(any(s.flags & C.FLAG_FIRST for s in sent))
        self.assertEqual([s.position for s in sent if s.flags & C.FLAG_SAMPLE], [6, 7])

    def test_fresh_starts_the_slot_over(self):
        # The host gives a slot to another conversation by starting it fresh:
        # position 0, FIRST, and what the slot held before is forgotten.
        rng = np.random.default_rng(24)
        emb, _, head = fake_model(rng)
        ring = C.Ring(2, lambda item: item.hidden, [head(0), head(1)])
        sent = recorded(ring, emb)
        ctl = C.Controller(emb, ring, slots=1, seed=25)
        ctl.append(0, [1, 2, 3], 3)
        ctl.run()
        sent.clear()
        ctl.append(0, [7, 8], 1, fresh=True)
        ctl.run()
        self.assertEqual([(s.position, s.flags & C.FLAG_FIRST) for s in sent], [(0, C.FLAG_FIRST), (1, 0)])
        self.assertEqual(emb.lookup(7).tolist(), sent[0].hidden.tolist())

    def test_a_slot_mid_turn_is_refused_and_a_turn_must_fit(self):
        rng = np.random.default_rng(26)
        emb, _, head = fake_model(rng)
        ctl = C.Controller(emb, C.Ring(2, lambda item: item.hidden, [head(0), head(1)]), slots=1, max_tokens=10)
        ctl.append(0, [1, 2], 3)
        with self.assertRaises(ValueError):
            ctl.append(0, [3], 1)
        ctl.run()
        self.assertTrue(ctl.fits(0, 2, 3))                  # five used of ten: the drawn token, two, three
        self.assertFalse(ctl.fits(0, 2, 4))
        self.assertTrue(ctl.fits(0, 6, 4, fresh=True))

    def test_a_prompt_goes_in_chunks(self):
        # Eight tokens a packet while eight are known, then one: the dies see
        # the same tokens in the same order, so a layer with state -- here a
        # running sum a slot, from zero at FIRST -- gives the same draws as a
        # token a packet, in a fraction of the packets.
        def run(chunk):
            rng = np.random.default_rng(18)
            emb, _, head = fake_model(rng)
            sums = {}
            def layer(item):
                if item.flags & C.FLAG_FIRST and item.position == 0:
                    sums[item.context] = np.zeros(emb.hidden, dtype=np.int64)
                sums[item.context] = sums[item.context] + item.hidden
                return np.clip(sums[item.context] // 4, -32768, 32767)
            ring = C.Ring(1, layer, [head(0), head(1)])          # one layer die: the state is its
            sent = recorded(ring, emb)
            positions = []
            ctl = C.Controller(emb, ring, slots=4, seed=19, chunk=chunk,
                               on_token=lambda slot, token, lp, pos, last: positions.append(pos))
            greedy = C.SamplingParams(top_k=1)      # the draws' words go to slots in retire order
            ctl.append(0, list(range(1, 44)), 4, greedy)
            ctl.append(1, list(range(20, 37)), 3, greedy)
            ctl.run()
            return ctl, sent, positions, ctl.slots[0].generated + ctl.slots[1].generated

        one, sent1, pos1, got1 = run(1)
        eight, sent8, pos8, got8 = run(8)
        self.assertEqual(got8, got1)
        self.assertEqual(sorted(pos8), sorted(pos1))
        chunks = sorted((s.position, s.tokens) for s in sent8 if s.tokens > 1)
        self.assertEqual(chunks, sorted([(p, 8) for p in range(0, 40, 8)] + [(0, 8), (8, 8)]))
        # Five chunks and three single prompt tokens for the first slot, two
        # and one for the second; neither prompt ends on a chunk, so no chunk
        # is drawn from.
        self.assertFalse(any(s.flags & C.FLAG_SAMPLE for s in sent8 if s.tokens > 1))
        self.assertLess(len(sent8), len(sent1) - 40)
        self.assertLess(eight.steps, one.steps)

    def test_a_chunk_that_ends_the_prompt_is_drawn_from(self):
        rng = np.random.default_rng(20)
        emb, logits_of, head = fake_model(rng)
        ring = C.Ring(3, lambda item: item.hidden, [head(0), head(1)])
        sent = recorded(ring, emb)
        ctl = C.Controller(emb, ring, slots=2, seed=21, chunk=4)
        ctl.append(0, [9, 8, 7, 6, 5, 4, 3, 2], 2)
        ctl.run()
        self.assertEqual([(s.position, s.tokens, s.flags & C.FLAG_SAMPLE) for s in sent],
                         [(0, 4, 0), (4, 4, C.FLAG_SAMPLE), (8, 1, C.FLAG_SAMPLE)])
        # The head dies read the chunk's last token: the draw is from token 2's logits.
        draws = np.random.default_rng(21)
        logits = logits_of(emb.lookup(2))
        rows, merged, _ = C.merge_lists([C.head_list(0, logits[:32], 8, 0), C.head_list(1, logits[32:], 8, 32)])
        self.assertEqual(ctl.slots[0].generated[0], C.sample(rows, merged, C.SamplingParams(), int(draws.integers(0, 1 << 32)))[0])


import shutil
import subprocess
import tempfile
from pathlib import Path

RTL = Path(__file__).parents[1] / "rtl"
SOURCES = [RTL / name for name in ("fabric_vector.sv", "fabric_controller.sv", "fabric_ring.sv",
                                   "fabric_controller_top.sv")]


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

    def test_ring_link(self) -> None:
        rng = np.random.default_rng(32)
        self.check("tb_ring", lambda d: C.emit_ring_vectors(d, rng, 12))
        self.check("tb_ring", lambda d: C.emit_ring_vectors(d, rng, 9, d=32))

    def test_datapath_end_to_end(self) -> None:
        """A request in, the model's packet out, the reply's lists back, the
        model's token drawn for the slot it belongs to."""
        rng = np.random.default_rng(33)
        self.check("tb_controller_top", lambda d: C.emit_top_vectors(d, rng, 8))
        self.check("tb_controller_top", lambda d: C.emit_top_vectors(d, rng, 6, d=16, k=16))


if __name__ == "__main__":
    unittest.main()
