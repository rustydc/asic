import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fabric import hpi as H

RTL = Path(__file__).parents[1] / "rtl"


class DeviceTest(unittest.TestCase):
    def test_part_and_geometry(self) -> None:
        d = H.DEVICE
        self.assertEqual(d.part, "APS512XXN-OB9-BG")
        self.assertEqual(d.words * 2, d.bytes)
        self.assertEqual(1 << (d.row_bits + d.col_bits), d.words)
        self.assertEqual(d.page_words, 1 << d.col_bits)
        self.assertEqual(H.STRIPE_BYTES, d.page_words * 2)

    def test_latency_codes_follow_the_datasheet_tables(self) -> None:
        self.assertEqual(H.latency_codes(250), (0b110, 10, 0b011, 9))
        self.assertEqual(H.latency_codes(200), (0b100, 7, 0b001, 7))
        self.assertEqual(H.latency_codes(133), (0b010, 5, 0b010, 5))
        self.assertEqual(H.register_read_latency(250), 9)
        self.assertEqual(H.register_read_latency(200), 7)
        mrs = H.mode_registers(250)
        self.assertEqual(mrs[0], 0x18)          # variable latency, code 110, full drive
        self.assertEqual(mrs[4], 0x60)          # write latency code 011
        self.assertEqual(mrs[8], 0x43)          # x16, 1K-word wrap
        self.assertEqual(H.mode_registers(250, fixed_latency=True)[0] & 0x20, 0x20)

    def test_frame_bytes_split_row_and_column(self) -> None:
        word = (0b101010101010101 << 10) | 0x2A6   # RA = 0x5555, CA = 0x2A6
        instr, a3, a2, a1, a0 = H.ca_bytes(H.CMD_LINEAR_READ, word)
        self.assertEqual(instr, 0x20)
        self.assertEqual(a3, 0b10)                # RA[14:13]
        self.assertEqual(a2, (0x5555 >> 5) & 0xFF)
        self.assertEqual(a1, ((0x5555 & 0x1F) << 3) | 0b010)
        self.assertEqual(a0, 0xA6)
        with self.assertRaises(ValueError):
            H.ca_bytes(H.CMD_LINEAR_READ, 3)
        self.assertEqual(H.register_frame(H.CMD_MR_WRITE, 8), [0xC0, 0, 0, 0, 8])

    def test_striping_and_chunks(self) -> None:
        self.assertEqual(H.split_address(0, 16), (0, 0))
        self.assertEqual(H.split_address(2048, 16), (1, 0))
        self.assertEqual(H.split_address(16 * 2048 + 100, 16), (0, 2048 + 100))
        parts = H.chunks(0x7680, 269, 4)
        self.assertEqual([n for _, _, n in parts], [24, 128, 117])
        self.assertEqual([d for d, _, _ in parts], [2, 3, 0])
        self.assertEqual(sum(n for _, _, n in parts), 269)
        # A head's 32 KB state runs on all sixteen devices.
        self.assertEqual(sorted(d for d, _, _ in H.chunks(0, 2048, 16)), list(range(16)))
        for _, daddr, n in H.chunks(12345 * 16, 500, 16):
            self.assertLessEqual(daddr % 2048 + n * 16, 2048)

    def test_efficiency(self) -> None:
        self.assertGreater(H.efficiency(128), 0.95)      # a full page
        self.assertGreater(H.efficiency(32), 0.85)       # a 512 B record
        self.assertLess(H.efficiency(5), 0.55)           # an 80 B index record
        data, total = H.burst_cycles(128)
        self.assertEqual(data, 512)
        self.assertEqual(total, 532)

    def test_striped_image_round_trips(self) -> None:
        rng = np.random.default_rng(0)
        image = H.StripedImage(4, 8192)
        payload = bytes(rng.integers(0, 256, 4800, dtype=np.uint8))
        image.write(1000 * 16, payload)
        self.assertEqual(image.read(1000 * 16, 4800), payload)
        touched = {H.split_address(a, 4)[0] for a in range(1000 * 16, 1000 * 16 + 4800, 16)}
        self.assertEqual(sum(1 for d in image.devices if any(d)), len(touched))


@unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog not installed")
class HpiRtlTest(unittest.TestCase):
    def run_rtl(self, ndev: int, seed: int, transactions: int = 20) -> str:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            params = H.emit_hpi_vectors(work, np.random.default_rng(seed), ndev=ndev, device_kb=16,
                                        transactions=transactions, max_beats=300)
            args = [f"-Ptb_hpi.{name}={value}" for name, value in params.items()]
            subprocess.run(["iverilog", "-g2012", "-I", str(RTL), "-s", "tb_hpi", "-o", "sim.vvp", *args,
                            str(RTL / "fabric_hpi.sv"), str(RTL / "tb_hpi.sv")],
                           cwd=work, check=True, capture_output=True, text=True)
            result = subprocess.run(["vvp", "sim.vvp"], cwd=work, check=True, capture_output=True, text=True)
        return result.stdout + result.stderr

    def test_bursts_over_four_devices(self) -> None:
        out = self.run_rtl(4, 30)
        self.assertIn("PASS", out, out)
        self.assertNotIn("ERROR", out, out)          # the device model's protocol checks

    def test_bursts_on_one_device_and_sixteen(self) -> None:
        for ndev in (1, 16):
            out = self.run_rtl(ndev, 31, transactions=12)
            self.assertIn("PASS", out, out)
            self.assertNotIn("ERROR", out, out)


if __name__ == "__main__":
    unittest.main()
