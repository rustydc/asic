import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fabric import controller as C
from fabric import die
from fabric import layer as L

RTL = Path(__file__).parents[1] / "rtl"


def packet(context, position, tokens, d=8, flags=0, seed=0):
    rng = np.random.default_rng(seed)
    hidden = rng.integers(-100, 100, size=(tokens, d) if tokens > 1 else d).astype(np.int16)
    return C.pack_item(C.WorkItem(context, position, hidden, flags))


class DieLinkModelTest(unittest.TestCase):
    def test_batches_close_when_full_on_a_new_shape_and_on_quiet(self):
        link = die.DieLink(8, chunk=4, lanes=2)
        singles = [packet(c, 10, 1) for c in range(3)]
        chunks = [packet(7, 0, 4), packet(8, 0, 4)]
        got = link.batches([singles + chunks, [packet(9, 3, 1)]])
        self.assertEqual([(b.chunked, [it.context for it in b.items]) for b in got],
                         [(False, [0, 1]), (False, [2]), (True, [7, 8]), (False, [9])])
        self.assertEqual([b.entry for b in got], [1, 0, 5, 0])

    def test_a_slot_takes_one_lane(self):
        # A prompt's packets come back to back from one slot; each is a batch
        # of its own, since two tokens of one context cannot run at once.
        link = die.DieLink(8, chunk=4, lanes=4)
        got = link.batches([[packet(3, 0, 1), packet(3, 1, 1), packet(5, 0, 1), packet(3, 2, 1)]])
        self.assertEqual([[(it.context, it.position) for it in b.items] for b in got], [[(3, 0)], [(3, 1), (5, 0)], [(3, 2)]])

    def test_what_is_dropped(self):
        link = die.DieLink(8, chunk=4, lanes=4)
        bad = bytearray(packet(1, 0, 1))
        bad[C.HEADER_BYTES] ^= 1
        three = packet(2, 0, 3)                              # no program for three tokens
        got = link.batches([[packet(0, 0, 1), bytes(bad), three, packet(3, 0, 1)]])
        self.assertEqual([[it.context for it in b.items] for b in got], [[0, 3]])
        self.assertEqual((link.crc_errors, link.malformed), (1, 1))

    def test_slot_and_first(self):
        link = die.DieLink(8, lanes=4, slot_pages=37, page_base=5)
        b = link.batches([[packet(2, 0, 1, flags=C.FLAG_FIRST), packet(3, 9, 1)]])[0]
        self.assertEqual([link.slot_page(it) for it in b.items], [5 + 74, 5 + 111])
        self.assertEqual(link.first_mask(b), 0b01)


@unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog not installed")
class DieLinkRtlTest(unittest.TestCase):
    def check(self, **kw) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            params = die.emit_die_link_vectors(work, np.random.default_rng(kw.pop("seed")), **kw)
            L.write_luts(work)
            args = [f"-Ptb_die_link.{name}={value}" for name, value in params.items()]
            sources = [RTL / n for n in ("fabric_vector.sv", "fabric_controller.sv", "fabric_ring.sv", "fabric_die_link.sv", "tb_die_link.sv")]
            subprocess.run(["iverilog", "-g2012", "-I", str(RTL), "-s", "tb_die_link", "-o", "sim.vvp", *args, *map(str, sources)],
                           cwd=work, check=True, capture_output=True, text=True)
            result = subprocess.run(["vvp", "sim.vvp"], cwd=work, check=True, capture_output=True, text=True)
        self.assertIn("PASS", result.stdout, result.stdout)

    def test_batches_and_forwarding(self) -> None:
        """Packets in, the model's batches started, the model's packets out:
        full batches, a shape change, quiet, CRC failures and packets the die
        has no program for, with the next die stalling."""
        self.check(seed=40)

    def test_two_lanes_and_a_chunk_of_two(self) -> None:
        self.check(seed=41, d=32, chunk=2, lanes=2)


if __name__ == "__main__":
    unittest.main()
