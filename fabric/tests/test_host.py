import struct
import unittest

import numpy as np

from fabric import controller as C
from fabric import host as H
from fabric.tests.test_controller import fake_model


def setup(slots=4, seed=21, sq=16, cq=64, contexts=1024):
    rng = np.random.default_rng(20)
    emb, logits_of, head = fake_model(rng)
    ring = C.Ring(3, lambda item: item.hidden, [head(0), head(1)])
    ctl = C.Controller(emb, ring, slots=slots, seed=seed)
    mem = H.HostMemory(1 << 20)
    dev = H.Device(ctl, mem, max_contexts=contexts)
    return dev, H.Driver(dev, mem, sq_entries=sq, cq_entries=cq), emb, head


def last_of(cmd_id):
    return lambda got: any(c.cmd_id == cmd_id and c.flags & H.LAST for c in got)


class FormatTest(unittest.TestCase):
    def test_entries_round_trip(self):
        cmd = H.Command(H.APPEND, 7, 123456, 0x1_2345_6780, 17, 5, 2048, 40, 0x8000, (2, 3), 0)
        raw = cmd.pack()
        self.assertEqual(len(raw), H.SQ_ENTRY)
        self.assertEqual(H.Command.unpack(raw), cmd)
        cmp = H.Completion(9, 7, H.OK, H.TOKEN | H.LAST | H.PHASE, 42, -123456, 3)
        self.assertEqual(len(cmp.pack()), H.CQ_ENTRY)
        self.assertEqual(H.Completion.unpack(cmp.pack()), cmp)

    def test_documented_offsets(self):
        raw = H.Command(H.APPEND, cmd_id=0x0102, ctx=0x03040506, addr=0x0708090A0B0C0D0E, count=0x11, max_new=0x22,
                        inv_t=0x33, top_k=0x44, top_p=0x55, stops=(0x66, 0x77)).pack()
        self.assertEqual(raw[0], H.APPEND)
        self.assertEqual(struct.unpack_from("<H", raw, 2)[0], 0x0102)
        self.assertEqual(struct.unpack_from("<I", raw, 4)[0], 0x03040506)
        self.assertEqual(struct.unpack_from("<Q", raw, 8)[0], 0x0708090A0B0C0D0E)
        self.assertEqual(struct.unpack_from("<IIIHHH", raw, 16), (0x11, 0x22, 0x33, 0x44, 0x55, 2))
        self.assertEqual(struct.unpack_from("<2I", raw, 36), (0x66, 0x77))
        raw = H.Completion(0x0102, 0x0304, 5, 6, 0x0708090A, -2, 0x0B0C).pack()
        self.assertEqual(struct.unpack("<IiHHHBB", raw), (0x0708090A, -2, 0x0102, 0x0304, 0x0B0C, 5, 6))


class QueueTest(unittest.TestCase):
    def test_identify(self):
        dev, drv, emb, _ = setup()
        cid = drv.identify()
        got = H.run(dev, drv, last_of(cid))
        self.assertEqual([(c.cmd_id, c.status) for c in got], [(cid, H.OK)])
        magic, version, contexts, vocab, slots, hidden = struct.unpack_from("<6I", drv.mem.read(drv.identify_page, 24))
        self.assertEqual((magic, version, contexts, vocab, slots, hidden), (H.MAGIC, H.VERSION, 1024, emb.vocab, 4, emb.hidden))

    def test_a_turn_streams_what_the_controller_draws(self):
        # The same operations on a bare controller with the same seed give the
        # same tokens; the queues add nothing and lose nothing.
        dev, drv, emb, head = setup()
        params = C.SamplingParams.of(0.8, top_k=6)
        opened = drv.open(params)
        ctx = H.run(dev, drv, last_of(opened))[0].ctx
        turn = drv.append(ctx, [5, 17, 3], 6)
        got = [c for c in H.run(dev, drv, last_of(turn)) if c.flags & H.TOKEN]
        self.assertEqual([c.cmd_id for c in got], [turn] * 6)
        self.assertEqual([bool(c.flags & H.LAST) for c in got], [False] * 5 + [True])

        ring = C.Ring(3, lambda item: item.hidden, [head(0), head(1)])
        ref = C.Controller(emb, ring, slots=4, seed=21)
        rid = ref.open(params)
        ref.append(rid, [5, 17, 3], 6)
        ref.run()
        self.assertEqual([c.token for c in got], ref.contexts[rid].tokens[3:])
        self.assertEqual([c.logprob for c in got], [H.logprob_fixed(lp) for lp in ref.contexts[rid].logprobs])

    def test_a_second_turn_continues_the_context(self):
        # A resident context keeps its state: the second turn carries only the
        # new tokens, and nothing goes round the ring as FIRST again.
        dev, drv, _, _ = setup()
        sent = []
        inject = dev.ctl.ring.inject
        dev.ctl.ring.inject = lambda packet: (sent.append(C.unpack_item(packet, dev.ctl.embedding.hidden)[0]), inject(packet))[1]
        ctx = H.run(dev, drv, last_of(drv.open()))[0].ctx
        H.run(dev, drv, last_of(drv.append(ctx, [1, 2], 3)))
        first_turn = len(sent)
        second = drv.append(ctx, [9], 2)
        got = [c for c in H.run(dev, drv, last_of(second)) if c.flags & H.TOKEN]
        self.assertEqual(len(got), 2)
        later = sent[first_turn:]
        self.assertEqual([s.position for s in later], list(range(4, 4 + len(later))))   # the third sampled token, then 9, ...
        self.assertFalse(any(s.flags & C.FLAG_FIRST for s in later))
        self.assertEqual(dev.ctl.contexts[ctx].tokens[:6], [1, 2] + dev.ctl.contexts[ctx].tokens[2:5] + [9])

    def test_a_short_completion_queue_wraps_and_holds_back(self):
        # Four completion entries and three contexts' tokens: the device waits
        # for room rather than overwrite, and the phase bit carries the host
        # round the ring many times.
        dev, drv, _, _ = setup(cq=4)
        ctxs = []
        for _ in range(3):
            ctxs.append(H.run(dev, drv, last_of(drv.open()))[0].ctx)
        turns = {drv.append(c, [c + 1, c + 2], 7): c for c in ctxs}
        done = lambda got: all(any(x.cmd_id == t and x.flags & H.LAST for x in got) for t in turns)
        got = H.run(dev, drv, done)
        for t, c in turns.items():
            mine = [x for x in got if x.cmd_id == t]
            self.assertEqual(len(mine), 7)
            self.assertTrue(all(x.ctx == c for x in mine))
            self.assertEqual([x.token for x in mine], dev.ctl.contexts[c].tokens[2:9])
        self.assertGreater(dev.interrupts, 3)

    def test_a_stop_token_ends_the_turn(self):
        dev, drv, _, _ = setup()
        probe = H.run(dev, drv, last_of(drv.open()))[0].ctx
        tokens = [c.token for c in H.run(dev, drv, last_of(drv.append(probe, [4, 5], 5))) if c.flags & H.TOKEN]
        # The same stream, now with its third token as a stop: two ordinary tokens, then the stop, LAST.
        dev, drv, _, _ = setup()
        ctx = H.run(dev, drv, last_of(drv.open(stops=[tokens[2]])))[0].ctx
        got = [c for c in H.run(dev, drv, last_of(drv.append(ctx, [4, 5], 5))) if c.flags & H.TOKEN]
        stop_at = tokens.index(tokens[2])
        self.assertEqual([c.token for c in got], tokens[:stop_at + 1])
        self.assertTrue(got[-1].flags & H.LAST)

    def test_cancel_ends_the_turn_and_keeps_the_context(self):
        dev, drv, _, _ = setup()
        ctx = H.run(dev, drv, last_of(drv.open()))[0].ctx
        turn = drv.append(ctx, [1, 2, 3], 1000)
        got = H.run(dev, drv, lambda g: sum(1 for c in g if c.flags & H.TOKEN) >= 3)
        drv.cancel(ctx)
        got += H.run(dev, drv, last_of(turn))
        tokens = [c for c in got if c.cmd_id == turn and c.flags & H.TOKEN]
        self.assertLess(len(tokens), 10)
        self.assertTrue(tokens[-1].flags & H.LAST and tokens[-1].flags & H.CANCELLED)
        again = drv.append(ctx, [4], 2)                       # the context is still open
        self.assertEqual(sum(1 for c in H.run(dev, drv, last_of(again)) if c.flags & H.TOKEN), 2)

    def test_errors(self):
        dev, drv, _, _ = setup(contexts=1)
        bad = drv.append(99, [1], 1)
        self.assertEqual(H.run(dev, drv, last_of(bad))[0].status, H.BAD_CONTEXT)
        ctx = H.run(dev, drv, last_of(drv.open()))[0].ctx
        self.assertEqual(H.run(dev, drv, last_of(drv.open()))[0].status, H.NO_CONTEXTS)
        self.assertEqual(H.run(dev, drv, last_of(drv.append(ctx, [1], 0)))[0].status, H.BAD_ARGUMENT)
        self.assertEqual(H.run(dev, drv, last_of(drv.append(ctx, [10_000], 1)))[0].status, H.BAD_ARGUMENT)
        turn = drv.append(ctx, [1], 50)
        busy = drv.append(ctx, [2], 1)                        # mid-turn
        got = H.run(dev, drv, last_of(busy))
        self.assertEqual([c.status for c in got if c.cmd_id == busy], [H.BUSY])
        H.run(dev, drv, last_of(turn))
        self.assertEqual(H.run(dev, drv, last_of(drv.submit(H.Command(0x7F))))[0].status, H.BAD_OPCODE)
        drv.close(ctx)
        closed = drv.append(ctx, [1], 1)
        self.assertEqual([c.status for c in H.run(dev, drv, last_of(closed)) if c.cmd_id == closed], [H.BAD_CONTEXT])


if __name__ == "__main__":
    unittest.main()
