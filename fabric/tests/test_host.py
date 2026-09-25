import struct
import unittest

import numpy as np

from fabric import controller as C
from fabric import host as H
from fabric.tests.test_controller import fake_model, recorded


def setup(slots=4, seed=21, sq=16, cq=64, max_tokens=1 << 17):
    rng = np.random.default_rng(20)
    emb, logits_of, head = fake_model(rng)
    ring = C.Ring(3, lambda item: item.hidden, [head(0), head(1)])
    ctl = C.Controller(emb, ring, slots=slots, seed=seed, max_tokens=max_tokens)
    mem = H.HostMemory(1 << 20)
    dev = H.Device(ctl, mem)
    return dev, H.Driver(dev, mem, sq_entries=sq, cq_entries=cq), emb, head


def last_of(cmd_id):
    return lambda got: any(c.cmd_id == cmd_id and c.flags & H.LAST for c in got)


def tokens_of(got, cmd_id):
    return [c for c in got if c.cmd_id == cmd_id and c.flags & H.TOKEN]


class FormatTest(unittest.TestCase):
    def test_entries_round_trip(self):
        cmd = H.Command(H.APPEND, 7, 123, 0x1_2345_6780, 17, 5, 2048, 40, 0x8000, (2, 3), H.FRESH)
        raw = cmd.pack()
        self.assertEqual(len(raw), H.SQ_ENTRY)
        self.assertEqual(H.Command.unpack(raw), cmd)
        cmp = H.Completion(9, 7, H.OK, H.TOKEN | H.LAST | H.PHASE, 42, -123456, 3)
        self.assertEqual(len(cmp.pack()), H.CQ_ENTRY)
        self.assertEqual(H.Completion.unpack(cmp.pack()), cmp)

    def test_documented_offsets(self):
        raw = H.Command(H.APPEND, cmd_id=0x0102, slot=0x03040506, addr=0x0708090A0B0C0D0E, count=0x11, max_new=0x22,
                        inv_t=0x33, top_k=0x44, top_p=0x55, stops=(0x66, 0x77), flags=H.FRESH).pack()
        self.assertEqual((raw[0], raw[1]), (H.APPEND, H.FRESH))
        self.assertEqual(struct.unpack_from("<H", raw, 2)[0], 0x0102)
        self.assertEqual(struct.unpack_from("<I", raw, 4)[0], 0x03040506)
        self.assertEqual(struct.unpack_from("<Q", raw, 8)[0], 0x0708090A0B0C0D0E)
        self.assertEqual(struct.unpack_from("<IIIHHH", raw, 16), (0x11, 0x22, 0x33, 0x44, 0x55, 2))
        self.assertEqual(struct.unpack_from("<2I", raw, 36), (0x66, 0x77))
        raw = H.Completion(0x0102, 0x0304, 5, 6, 0x0708090A, -2, 0x0B0C).pack()
        self.assertEqual(struct.unpack("<IiHHHBB", raw), (0x0708090A, -2, 0x0102, 0x0304, 0x0B0C, 5, 6))


class QueueTest(unittest.TestCase):
    def test_identify(self):
        dev, drv, emb, _ = setup(max_tokens=4096)
        cid = drv.identify()
        got = H.run(dev, drv, last_of(cid))
        self.assertEqual([(c.cmd_id, c.status) for c in got], [(cid, H.OK)])
        fields = struct.unpack_from("<6I", drv.mem.read(drv.identify_page, 24))
        self.assertEqual(fields, (H.MAGIC, H.VERSION, 4, emb.vocab, emb.hidden, 4096))
        self.assertEqual(dev.mmio_read(H.REG_SLOTS), 4)

    def test_a_turn_streams_what_the_controller_draws(self):
        # The same turn on a bare controller with the same seed gives the same
        # tokens; the queues add nothing and lose nothing.
        dev, drv, emb, head = setup()
        params = C.SamplingParams.of(0.8, top_k=6)
        turn = drv.append(2, [5, 17, 3], 6, params, fresh=True)
        got = tokens_of(H.run(dev, drv, last_of(turn)), turn)
        self.assertEqual([c.slot for c in got], [2] * 6)
        self.assertEqual([bool(c.flags & H.LAST) for c in got], [False] * 5 + [True])

        ref = C.Controller(emb, C.Ring(3, lambda item: item.hidden, [head(0), head(1)]), slots=4, seed=21)
        ref.append(2, [5, 17, 3], 6, params)
        ref.run()
        self.assertEqual([c.token for c in got], ref.slots[2].generated)
        self.assertEqual([c.logprob for c in got], [H.logprob_fixed(lp) for lp in ref.slots[2].logprobs])

    def test_a_second_turn_goes_on_from_the_slot(self):
        # The slot keeps its context: the second turn carries only the new
        # tokens, and nothing goes round the ring as FIRST again.
        dev, drv, emb, _ = setup()
        sent = recorded(dev.ctl.ring, emb)
        first = tokens_of(H.run(dev, drv, last_of(t := drv.append(0, [1, 2], 3, fresh=True))), t)
        before = len(sent)
        second = drv.append(0, [9], 2)
        got = tokens_of(H.run(dev, drv, last_of(second)), second)
        self.assertEqual(len(got), 2)
        later = sent[before:]
        self.assertEqual([s.position for s in later], list(range(4, 4 + len(later))))   # the third drawn token, then 9, ...
        self.assertFalse(any(s.flags & C.FLAG_FIRST for s in later))
        self.assertEqual(emb.lookup(first[-1].token).tolist(), later[0].hidden.tolist())

    def test_fresh_gives_the_slot_to_another_conversation(self):
        dev, drv, emb, _ = setup(slots=1)
        sent = recorded(dev.ctl.ring, emb)
        H.run(dev, drv, last_of(drv.append(0, [1, 2, 3], 2, fresh=True)))
        sent.clear()
        H.run(dev, drv, last_of(drv.append(0, [7, 8], 1, fresh=True)))
        self.assertEqual([(s.position, s.flags & C.FLAG_FIRST) for s in sent], [(0, C.FLAG_FIRST), (1, 0)])

    def test_a_short_completion_queue_wraps_and_holds_back(self):
        # Four completion entries and three slots' tokens: the device waits
        # for room rather than overwrite, and the phase bit carries the host
        # round the ring many times.
        dev, drv, _, _ = setup(cq=4)
        turns = {drv.append(k, [k + 1, k + 2], 7, fresh=True): k for k in range(3)}
        done = lambda got: all(any(x.cmd_id == t and x.flags & H.LAST for x in got) for t in turns)
        got = H.run(dev, drv, done)
        for t, k in turns.items():
            mine = tokens_of(got, t)
            self.assertEqual(len(mine), 7)
            self.assertTrue(all(x.slot == k for x in mine))
            self.assertEqual([x.token for x in mine], dev.ctl.slots[k].generated)
        self.assertGreater(dev.interrupts, 3)

    def test_a_stop_token_ends_the_turn(self):
        dev, drv, _, _ = setup()
        t = drv.append(0, [4, 5], 5, fresh=True)
        tokens = [c.token for c in tokens_of(H.run(dev, drv, last_of(t)), t)]
        # The same stream, now with its third token as a stop: two ordinary tokens, then the stop, LAST.
        dev, drv, _, _ = setup()
        t = drv.append(0, [4, 5], 5, stops=[tokens[2]], fresh=True)
        got = tokens_of(H.run(dev, drv, last_of(t)), t)
        stop_at = tokens.index(tokens[2])
        self.assertEqual([c.token for c in got], tokens[:stop_at + 1])
        self.assertTrue(got[-1].flags & H.LAST)

    def test_cancel_ends_the_turn_and_keeps_the_slot(self):
        dev, drv, _, _ = setup()
        turn = drv.append(1, [1, 2, 3], 1000, fresh=True)
        got = H.run(dev, drv, lambda g: len(tokens_of(g, turn)) >= 3)
        drv.cancel(1)
        got += H.run(dev, drv, last_of(turn))
        mine = [c for c in got if c.cmd_id == turn]
        self.assertLess(len(mine), 10)
        self.assertTrue(mine[-1].flags & H.LAST and mine[-1].flags & H.CANCELLED)
        again = drv.append(1, [4], 2)                         # the slot keeps its context
        self.assertEqual(len(tokens_of(H.run(dev, drv, last_of(again)), again)), 2)

    def test_errors(self):
        dev, drv, _, _ = setup(slots=2, max_tokens=64)
        status = lambda cid: [c.status for c in H.run(dev, drv, last_of(cid)) if c.cmd_id == cid][-1]
        self.assertEqual(status(drv.append(2, [1], 1)), H.BAD_SLOT)
        self.assertEqual(status(drv.append(0, [1], 0)), H.BAD_ARGUMENT)
        self.assertEqual(status(drv.append(0, [10_000], 1)), H.BAD_ARGUMENT)
        self.assertEqual(status(drv.append(0, [], 1, fresh=True)), H.BAD_ARGUMENT)     # nothing to start from
        self.assertEqual(status(drv.append(0, list(range(60)), 5, fresh=True)), H.TOO_LONG)
        turn = drv.append(0, [1], 50, fresh=True)
        busy = drv.append(0, [2], 1)                          # mid-turn
        got = H.run(dev, drv, last_of(busy))
        self.assertEqual([c.status for c in got if c.cmd_id == busy], [H.BUSY])
        H.run(dev, drv, last_of(turn))
        self.assertEqual(status(drv.submit(H.Command(0x02))), H.BAD_OPCODE)         # OPEN is gone: slots are the host's
        self.assertEqual(status(drv.append(0, [], 1)), H.OK)                        # the last drawn token goes on


class SlotsTest(unittest.TestCase):
    def test_the_driver_gives_the_least_recently_used_idle_slot_away(self):
        # The policy is the driver's: a conversation keeps its slot until the
        # slot is wanted, and one that lost it comes back fresh, with its
        # history, which the driver keeps.
        slots = H.Slots(2)
        self.assertEqual(slots.place("a"), (0, True))
        self.assertEqual(slots.place("b"), (1, True))
        self.assertEqual(slots.place("a"), (0, False))
        self.assertEqual(slots.place("c", busy=lambda k: False), (1, True))      # b was the least recently used
        self.assertEqual(slots.place("b", busy=lambda k: k == 0), (1, True))     # a is mid-turn, so c's goes
        self.assertIsNone(slots.place("d", busy=lambda k: True))


if __name__ == "__main__":
    unittest.main()
