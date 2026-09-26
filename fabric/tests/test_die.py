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
    def test_what_is_dropped(self):
        link = die.DieLink(8, chunk=4, lanes=4)
        bad = bytearray(packet(1, 0, 1))
        bad[C.HEADER_BYTES] ^= 1
        three = packet(2, 0, 3)                              # no program for three tokens
        got = link.admit([packet(0, 0, 1), bytes(bad), three, packet(3, 0, 1)])
        self.assertEqual([it.context for it in got], [0, 3])
        self.assertEqual((link.crc_errors, link.malformed), (1, 1))

    def test_a_lane_is_given_its_layers(self):
        # Four layers, kinds recurrent, recurrent, recurrent, global, each at
        # its part of the slot; the lane's token with each run.
        link = die.DieLink(8, chunk=4, lanes=4, slot_pages=37, page_base=5, layers=4)
        table = [10 | (3 << 16), 20 | (4 << 16), 30 | (5 << 16), 40 | (6 << 16)]      # {kind, chunked}
        layers = [0 | (0 << 1), 0 | (9 << 1), 0 | (18 << 1), 1 | (27 << 1)]
        item = link.admit([packet(2, 7, 4, flags=C.FLAG_FIRST)])[0]
        self.assertEqual(link.pushes(item, table, layers),
                         [die.Push(20, 4, 0, 0, True, 5 + 74, 7), die.Push(20, 4, 1, 9, True, 5 + 74, 7),
                          die.Push(20, 4, 2, 18, True, 5 + 74, 7), die.Push(40, 6, 3, 27, True, 5 + 74, 7)])

    def test_a_context_leaves_in_order(self):
        a, b, c = packet(3, 0, 1), packet(3, 1, 1, seed=1), packet(5, 0, 1, seed=2)
        self.assertEqual(die.order_problems([a, b, c], [c, a, b]), [])
        self.assertEqual(len(die.order_problems([a, b, c], [b, a, c])), 1)
        words = [int.from_bytes(p[i:i + 4], "little") for p in (a, c) for i in range(0, len(p), 4)]
        self.assertEqual(die.packets_of(words), [a, c])


@unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog not installed")
class DieLinkRtlTest(unittest.TestCase):
    def check(self, **kw) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            model = die.emit_die_link_vectors(work, np.random.default_rng(kw.pop("seed")), **kw)
            L.write_luts(work)
            args = [f"-Ptb_die_link.{name}={value}" for name, value in model["params"].items()]
            sources = [RTL / n for n in ("fabric_vector.sv", "fabric_controller.sv", "fabric_ring.sv", "fabric_die_link.sv", "tb_die_link.sv")]
            subprocess.run(["iverilog", "-g2012", "-I", str(RTL), "-s", "tb_die_link", "-o", "sim.vvp", *args, *map(str, sources)],
                           cwd=work, check=True, capture_output=True, text=True)
            result = subprocess.run(["vvp", "sim.vvp"], cwd=work, check=True, capture_output=True, text=True)
            self.assertIn("PASS", result.stdout, result.stdout)
            self.assertEqual(die.check_die_link_run(work, model), [])

    def test_lanes_and_forwarding(self) -> None:
        """Packets in, each item's lane given the model's runs, the model's
        packets out: more packets than lanes, shapes side by side, CRC
        failures and packets the die has no program for, a prompt's packets
        from one slot one at a time, lanes finishing out of order, and the
        next die stalling."""
        self.check(seed=40)

    def test_two_lanes_and_a_chunk_of_two(self) -> None:
        self.check(seed=41, d=32, chunk=2, lanes=2)

    def test_four_layers(self) -> None:
        self.check(seed=42, layers=4)


if __name__ == "__main__":
    unittest.main()
