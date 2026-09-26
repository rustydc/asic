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
        # The stripe divides the page, because a burst never crosses one, and
        # the RTL's stripe unit has to be told the same number.
        self.assertEqual((d.page_words * 2) % H.STRIPE_BYTES, 0)
        rtl = (Path(__file__).parents[1] / "rtl" / "fabric_hpi.sv").read_text()
        self.assertIn(f"localparam int STRIPE = {H.STRIPE_BYTES};", rtl)

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
        # Written against STRIPE_BYTES rather than a number, because the size
        # is a lever: it decides how many devices a transfer of a given size
        # reaches, and a test that restates it stops the lever being pulled.
        stripe, bps = H.STRIPE_BYTES, H.STRIPE_BYTES // H.BEAT_BYTES
        self.assertEqual(H.split_address(0, 16), (0, 0))
        self.assertEqual(H.split_address(stripe, 16), (1, 0))
        self.assertEqual(H.split_address(16 * stripe + 100, 16), (0, stripe + 100))
        parts = H.chunks(0x7680, 269, 4)
        self.assertEqual(sum(n for _, _, n in parts), 269)
        self.assertTrue(all(n <= bps for _, _, n in parts))
        self.assertEqual([d for d, _, _ in parts][:2], [(0x7680 // stripe) % 4, (0x7680 // stripe + 1) % 4])
        # A head's 16 KB state reaches every one of sixteen devices.
        self.assertEqual(sorted(d for d, _, _ in H.chunks(0, 16 * 1024 // H.BEAT_BYTES, 16)), list(range(16)))
        for _, daddr, n in H.chunks(12345 * 16, 500, 16):
            self.assertLessEqual(daddr % stripe + n * H.BEAT_BYTES, stripe)

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
    def run_rtl(self, ndev: int, seed: int, transactions: int = 20, **extra: object) -> str:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            params = H.emit_hpi_vectors(work, np.random.default_rng(seed), ndev=ndev, device_kb=16,
                                        transactions=transactions, max_beats=300)
            params.update(extra)
            args = [f"-Ptb_hpi.{name}={value}" for name, value in params.items()]
            subprocess.run(["iverilog", "-g2012", "-I", str(RTL), "-s", "tb_hpi", "-o", "sim.vvp", *args,
                            str(RTL / "fabric_phy.sv"), str(RTL / "fabric_hpi.sv"), str(RTL / "tb_hpi.sv")],
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


@unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog not installed")
class CdcRtlTest(unittest.TestCase):
    """The asynchronous FIFO on its own, then the memory path across the bridge."""

    def test_async_fifo_at_three_clock_ratios(self) -> None:
        for aw, wclk, rclk, seed in ((3, 1.3, 4.0, 1), (2, 4.0, 1.3, 2), (4, 2.0, 2.1, 3)):
            with tempfile.TemporaryDirectory() as directory:
                args = [f"-Ptb_async_fifo.AW={aw}", f"-Ptb_async_fifo.WCLK={wclk}", f"-Ptb_async_fifo.RCLK={rclk}",
                        f"-Ptb_async_fifo.SEED={seed}", "-Ptb_async_fifo.N=2000"]
                subprocess.run(["iverilog", "-g2012", "-I", str(RTL), "-s", "tb_async_fifo", "-o", "sim.vvp", *args,
                                str(RTL / "fabric_cdc.sv"), str(RTL / "tb_async_fifo.sv")],
                               cwd=directory, check=True, capture_output=True, text=True)
                out = subprocess.run(["vvp", "sim.vvp"], cwd=directory, check=True, capture_output=True, text=True).stdout
            self.assertIn("PASS", out, out)
            self.assertIn(f"peak occupancy {1 << aw}", out, out)      # the flags let it fill completely

    def test_memory_path_across_the_bridge(self) -> None:
        for ndev, core_ns in ((4, 1.3), (16, 1.25), (2, 3.0)):
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                params = H.emit_hpi_vectors(work, np.random.default_rng(40 + ndev), ndev=ndev, device_kb=16,
                                            transactions=16, max_beats=300)
                args = [f"-Ptb_mem_bridge.{name}={value}" for name, value in params.items()] + [f"-Ptb_mem_bridge.CORE_NS={core_ns}"]
                subprocess.run(["iverilog", "-g2012", "-I", str(RTL), "-s", "tb_mem_bridge", "-o", "sim.vvp", *args,
                                str(RTL / "fabric_phy.sv"), str(RTL / "fabric_cdc.sv"), str(RTL / "fabric_hpi.sv"),
                                str(RTL / "tb_mem_bridge.sv")],
                               cwd=work, check=True, capture_output=True, text=True)
                result = subprocess.run(["vvp", "sim.vvp"], cwd=work, check=True, capture_output=True, text=True)
            out = result.stdout + result.stderr
            self.assertIn("PASS", out, out)
            self.assertNotIn("ERROR", out, out)

    def test_memory_path_four_beats_a_transfer(self) -> None:
        # The engine's path: the controller side moves four beats a transfer,
        # the core side two for a wide request (every other one here) and one
        # for a narrow; and a narrow core side, which drains a read slower than
        # the controller fills it, so the read FIFO's push-back is exercised.
        for ndev, core_ns, cxb in ((4, 1.67, 2), (16, 1.25, 2), (4, 1.3, 1)):
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                params = H.emit_hpi_vectors(work, np.random.default_rng(50 + ndev), ndev=ndev, device_kb=16,
                                            transactions=16, max_beats=300)
                args = [f"-Ptb_mem_bridge.{name}={value}" for name, value in params.items()]
                args += [f"-Ptb_mem_bridge.CORE_NS={core_ns}", f"-Ptb_mem_bridge.CXB={cxb}", "-Ptb_mem_bridge.MXB=4"]
                subprocess.run(["iverilog", "-g2012", "-I", str(RTL), "-s", "tb_mem_bridge", "-o", "sim.vvp", *args,
                                str(RTL / "fabric_phy.sv"), str(RTL / "fabric_cdc.sv"), str(RTL / "fabric_hpi.sv"),
                                str(RTL / "tb_mem_bridge.sv")],
                               cwd=work, check=True, capture_output=True, text=True)
                result = subprocess.run(["vvp", "sim.vvp"], cwd=work, check=True, capture_output=True, text=True)
            out = result.stdout + result.stderr
            self.assertIn("PASS", out, out)
            self.assertNotIn("ERROR", out, out)


@unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog not installed")
@unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog not installed")
class PathTimingTest(unittest.TestCase):
    """``PathModel`` against the RTL: each request through the bridge, the
    stripe unit, the channels and the device models, taken on the core clock
    to complete, one at a time; and runs of requests back to back, which the
    controller overlaps across its devices.  The RTL is the pipelined
    controller; the model's ``asbuilt`` mode is the one before it."""

    CORE_NS = 1.25                             # the engine testbench's core clock, 800 MHz

    def measure(self, ndev: int) -> list[tuple[tuple, int]]:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            params, spec = H.emit_timing_vectors(work, ndev)
            args = [f"-Ptb_mem_bridge.{name}={value}" for name, value in params.items()]
            args.append(f"-Ptb_mem_bridge.CORE_NS={self.CORE_NS}")
            subprocess.run(["iverilog", "-g2012", "-I", str(RTL), "-s", "tb_mem_bridge", "-o", "sim.vvp", *args,
                            *(str(RTL / n) for n in ("fabric_cdc.sv", "fabric_phy.sv", "fabric_hpi.sv", "tb_mem_bridge.sv"))],
                           cwd=work, check=True, capture_output=True, text=True)
            out = subprocess.run(["vvp", "sim.vvp"], cwd=work, check=True, capture_output=True, text=True).stdout
            self.assertIn("PASS", out, out)
            rows = [[int(v) for v in line.split()] for line in (work / "req_times.txt").read_text().splitlines()]
        return [(spec[n], done - take) for n, _, _, take, done in rows]

    def check(self, ndev: int) -> None:
        model = H.PathModel(ndev, mode="pipelined", core_mhz=1000 / self.CORE_NS)
        for (write, beats, addr), took in self.measure(ndev):
            want = model.request(write, beats, addr)
            # The device model draws a read's refresh push-out at random, 0
            # to the latency, and the model takes its mean, so a read may be
            # off by half the latency besides.
            spread = 0 if write else model.lc / 2 * model.core_mhz / model.f_mhz
            self.assertLessEqual(abs(want - took), 0.03 * took + spread,
                                 f"{'write' if write else 'read'} of {beats} beats at {addr}: the RTL took {took}, the model says {want}")

    def test_sixteen_devices(self) -> None:
        self.check(16)

    def test_four_devices(self) -> None:
        self.check(4)

    RUNS = {
        "window pages": [(False, 128, i * 2048) for i in range(8)],
        "half pages": [(False, 64, i * 1024) for i in range(8)],
        "index pages": [(False, 125, 5000 * 16 + i * 2000) for i in range(6)],
        "a state slot out and in": [(True, 1024, 0), (False, 1024, 1 << 20)],
        "two": [(True, 1024, 0), (False, 1024, 1 << 20), (True, 1024, 16384), (False, 1024, (1 << 20) + 16384)],
        "block records": [(False, 16, (i * 37 % 64) * 1024) for i in range(8)],
        "state slots read ahead": [(False, 1024, (1 << 20) + i * 16384) for i in range(3)],
    }

    def run_whole(self, ndev: int, spec) -> int:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            params = H.emit_request_vectors(work, ndev, spec, pipe=True)
            args = [f"-Ptb_mem_bridge.{name}={value}" for name, value in params.items()]
            args.append(f"-Ptb_mem_bridge.CORE_NS={self.CORE_NS}")
            subprocess.run(["iverilog", "-g2012", "-I", str(RTL), "-s", "tb_mem_bridge", "-o", "sim.vvp", *args,
                            *(str(RTL / n) for n in ("fabric_cdc.sv", "fabric_phy.sv", "fabric_hpi.sv", "tb_mem_bridge.sv"))],
                           cwd=work, check=True, capture_output=True, text=True)
            out = subprocess.run(["vvp", "sim.vvp"], cwd=work, check=True, capture_output=True, text=True).stdout
            self.assertIn("PASS", out, out)
            first, last = map(int, (work / "req_times.txt").read_text().split())
        return last - first

    def test_runs_of_requests_overlap(self) -> None:
        # Requests back to back: the stripe takes each once the last one's
        # chunks are issued, and a device takes its next chunk while the
        # last waits its turn on the port.  One at a time they cost several
        # times as much.
        for ndev in (16, 4):
            model = H.PathModel(ndev, mode="pipelined", core_mhz=1000 / self.CORE_NS)
            for name, spec in self.RUNS.items():
                took = self.run_whole(ndev, spec)
                want = model.sequence(spec)
                self.assertLessEqual(abs(want - took), 0.05 * took, f"{name} over {ndev}: the RTL took {took}, the model says {want}")
                if ndev == 16 and name != "a state slot out and in":
                    self.assertLess(took, 0.8 * sum(model.request(w, b, a) for w, b, a in spec), name)


class PhyRtlTest(unittest.TestCase):
    """The DLL on its own across tap lengths, then the controller through its delay lines."""

    def run_dll(self, **params: object) -> str:
        with tempfile.TemporaryDirectory() as directory:
            args = [f"-Ptb_dll.{name}={value}" for name, value in params.items()]
            subprocess.run(["iverilog", "-g2012", "-I", str(RTL), "-s", "tb_dll", "-o", "sim.vvp", *args,
                            str(RTL / "fabric_phy.sv"), str(RTL / "tb_dll.sv")],
                           cwd=directory, check=True, capture_output=True, text=True)
            return subprocess.run(["vvp", "sim.vvp"], cwd=directory, check=True, capture_output=True, text=True).stdout

    def test_dll_locks_from_25_to_90_ps_taps(self) -> None:
        for tap_ps in (25.0, 40.0, 60.0, 90.0):
            out = self.run_dll(TAP_PS=tap_ps)
            self.assertIn("PASS", out, out)
            self.assertIn(f"quarter {round(4000 / tap_ps / 4)} taps", out, out)

    def test_dll_reports_a_line_too_short_for_the_period(self) -> None:
        out = self.run_dll(TAP_PS=25.0, TAPS=128, EXPECT_RANGE_ERR=1)
        self.assertIn("PASS", out, out)

    def test_bursts_through_the_delay_lines(self) -> None:
        for tap_ps in (25.0, 90.0):
            out = HpiRtlTest.run_rtl(self, 4, 32, transactions=12, USE_DLL=1, TAP_PS=tap_ps)
            self.assertIn("PASS", out, out)
            # The code as the run ends, which the DLL may have moved a tap
            # while the channels were quiet.
            quarter = int(out.split("quarter ")[1].split()[0])
            self.assertLessEqual(abs(quarter - 4000 / tap_ps / 4), 1, out)
            self.assertNotIn("ERROR", out, out)


if __name__ == "__main__":
    unittest.main()
