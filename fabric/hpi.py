"""The local memory device and its controller: AP Memory APS512XXN-OB9-BG.

The PSRAM board (``hw/board_psram.yaml``) gives every layer die sixteen
x16 HPI PSRAMs.  The part is the AP Memory APS512XXN-OB9-BG: 512 Mb, the
Xccela DDR OPI/HPI interface in its 16-bit (HPI) mode, 250 MHz, 1.62 to
1.98 V, a 24-ball 6 x 8 mm BGA at 1.0 mm pitch (datasheet rev 1.0,
November 2025).  This module holds what the controller in
``fabric/rtl/fabric_hpi.sv`` is built from and the model its testbench
checks against:

* the device's geometry, timing and mode-register encodings as the
  datasheet states them (``DEVICE``, ``mode_registers``);
* the command/address frame (``ca_bytes``) and the address split between
  row and column;
* the striping of the die's one memory port over the sixteen devices in
  2 KB stripes, one device page each (``split_address``, ``chunks``);
* a byte-image model of the striped devices and the transaction vectors
  for ``tb_hpi``.

Two protocol details come from the datasheet's figures rather than its
text and are stated here as the assumption the RTL and the device model
share: the first data word of a burst sits ``latency`` clocks after the
third command clock (the one carrying A1/A0), and the 512 Mb part's extra
row bit RA[14] rides in bit 1 of the A3 byte.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fabric.tile import write_hex

BEAT_BYTES = 16
STRIPE_BYTES = 2048             # one device page at x16: 1024 words


@dataclass(frozen=True)
class Device:
    part: str = "APS512XXN-OB9-BG"
    package: str = "BGA24 6 x 8 x 1.2 mm, 1.0 mm pitch, 0.4 mm balls"
    bytes: int = 64 << 20
    words: int = 32 << 20
    page_words: int = 1024
    row_bits: int = 15
    col_bits: int = 10
    f_max_mhz: int = 250
    v_dd: tuple[float, float] = (1.62, 1.98)
    t_pu_us: float = 150.0      # power-up self-initialisation
    t_rst_us: float = 2.0       # global reset to command
    t_cph_ns: float = 28.0      # CE# high between bursts at 250 MHz
    t_cem_us: float = 4.0       # CE# low maximum, standard temperature
    t_rc_ns: float = 60.0       # minimum cycle for consecutive short writes
    t_dqsck_ns: tuple[float, float] = (2.0, 6.5)
    t_hz_ns: float = 6.0
    i_cc_ma_x16_250: float = 60.0
    signals: tuple[str, ...] = ("CLK", "CE#", "DQ0-15", "DQS/DM0", "DQS/DM1")


DEVICE = Device()

# Commands (the instruction byte on the first rising edge).
CMD_SYNC_READ = 0x00
CMD_SYNC_WRITE = 0x80
CMD_LINEAR_READ = 0x20
CMD_LINEAR_WRITE = 0xA0
CMD_MR_READ = 0x40
CMD_MR_WRITE = 0xC0
CMD_GLOBAL_RESET = 0xFF

# Read latency codes MR0[4:2]: (variable latency, maximum push-out and fixed latency, Fmax MHz).
READ_LATENCY = {0b000: (3, 6, 66), 0b001: (4, 8, 109), 0b010: (5, 10, 133), 0b011: (6, 12, 166),
                0b100: (7, 14, 200), 0b101: (9, 16, 225), 0b110: (10, 18, 250)}
# Write latency codes MR4[7:5]: (latency, Fmax MHz).
WRITE_LATENCY = {0b000: (3, 66), 0b100: (4, 109), 0b010: (5, 133), 0b110: (6, 166), 0b001: (7, 200),
                 0b101: (8, 225), 0b011: (9, 250)}
VENDOR_ID = 0b01101             # MR1[4:0]
DENSITY_512MB = 0b110           # MR2[2:0]
GOOD_DIE = 0b110                # MR2[7:5]


def latency_codes(f_mhz: float) -> tuple[int, int, int, int]:
    """``(read code, read latency, write code, write latency)`` for a clock."""
    rc = min((c for c, (_, _, fmax) in READ_LATENCY.items() if fmax >= f_mhz), key=lambda c: READ_LATENCY[c][2])
    wc = min((c for c, (_, fmax) in WRITE_LATENCY.items() if fmax >= f_mhz), key=lambda c: WRITE_LATENCY[c][1])
    return rc, READ_LATENCY[rc][0], wc, WRITE_LATENCY[wc][0]


def mode_registers(f_mhz: float = 250.0, drive: int = 0, fixed_latency: bool = False) -> dict[int, int]:
    """MR0, MR4 and MR8 for x16 operation at ``f_mhz``.

    MR0: [7:6] 0, [5] latency type, [4:2] read latency code, [1:0] drive
    strength (0 full 25 ohm, 1 half, 2 quarter, 3 eighth).  MR4: [7:5] write
    latency code, [4:3] refresh (00 always 4x), [2:0] PASR full array.  MR8:
    [7] 0, [6] x16, [3] RBX off, [2] wrap (not hybrid), [1:0] 1K-word wrap.
    """
    rc, _, wc, _ = latency_codes(f_mhz)
    mr0 = (int(fixed_latency) << 5) | (rc << 2) | (drive & 3)
    mr4 = wc << 5
    mr8 = (1 << 6) | 0b11
    return {0: mr0, 4: mr4, 8: mr8}


def register_read_latency(f_mhz: float) -> int:
    """Register reads take LC clocks up to 200 MHz and LC - 1 above."""
    _, lc, _, _ = latency_codes(f_mhz)
    return lc if f_mhz <= 200 else lc - 1


def ca_bytes(command: int, word_addr: int, device: Device = DEVICE) -> list[int]:
    """The five bytes of a frame: the instruction, then A3, A2, A1, A0 on the
    second and third clocks' rising and falling edges.  RA is the row, CA
    the column (10 bits at x16; CA[10] is unused there)."""
    if word_addr & 1:
        raise ValueError("memory accesses start on even addresses")
    ra = word_addr >> device.col_bits
    ca = word_addr & ((1 << device.col_bits) - 1)
    a3 = (ra >> 13) & 0x3                         # RA[14:13]: RA[13] in bit 0 as the datasheet's table, RA[14] above it
    a2 = (ra >> 5) & 0xFF                         # RA[12:5]
    a1 = ((ra & 0x1F) << 3) | ((ca >> 8) & 0x7)   # RA[4:0], CA[10:8]
    a0 = ca & 0xFF
    return [command & 0xFF, a3, a2, a1, a0]


def register_frame(command: int, mr: int) -> list[int]:
    """A mode-register frame carries the register address in A0."""
    return [command & 0xFF, 0, 0, 0, mr & 0xFF]


# ---------------------------------------------------------------------------
# Striping of the port over the devices
# ---------------------------------------------------------------------------


def split_address(byte_addr: int, ndev: int, stripe: int = STRIPE_BYTES) -> tuple[int, int]:
    """``(device, byte address within the device)``: consecutive stripes go
    to consecutive devices, so a burst of ``ndev`` stripes runs on all of them."""
    s, off = divmod(byte_addr, stripe)
    return s % ndev, (s // ndev) * stripe + off


def chunks(byte_addr: int, beats: int, ndev: int, stripe: int = STRIPE_BYTES) -> list[tuple[int, int, int]]:
    """A burst as ``[(device, device byte address, beats)]`` chunks, each within one stripe."""
    out = []
    while beats:
        dev, daddr = split_address(byte_addr, ndev, stripe)
        room = (stripe - byte_addr % stripe) // BEAT_BYTES
        n = min(beats, room)
        out.append((dev, daddr, n))
        byte_addr += n * BEAT_BYTES
        beats -= n
    return out


def burst_cycles(beats: int, f_mhz: float = 250.0, device: Device = DEVICE) -> tuple[int, int]:
    """``(data clocks, total clocks)`` of one chunk on one device: three
    command clocks, the latency (the variable read latency at its minimum),
    two words per clock, and tCPH before the next burst."""
    _, lc, _, _ = latency_codes(f_mhz)
    data = beats * BEAT_BYTES // 2 // 2
    tcph = -(-int(device.t_cph_ns * f_mhz) // 1000)
    return data, 3 + lc + data + tcph


def efficiency(beats: int, f_mhz: float = 250.0) -> float:
    data, total = burst_cycles(beats, f_mhz)
    return data / total


class StripedImage:
    """The bytes of ``ndev`` devices addressed through the stripe map."""

    def __init__(self, ndev: int, device_bytes: int, stripe: int = STRIPE_BYTES):
        self.ndev, self.stripe = ndev, stripe
        self.devices = [bytearray(device_bytes) for _ in range(ndev)]

    def write(self, byte_addr: int, payload: bytes) -> None:
        for i in range(0, len(payload), BEAT_BYTES):
            dev, daddr = split_address(byte_addr + i, self.ndev, self.stripe)
            self.devices[dev][daddr:daddr + BEAT_BYTES] = payload[i:i + BEAT_BYTES]

    def read(self, byte_addr: int, nbytes: int) -> bytes:
        out = bytearray()
        for i in range(0, nbytes, BEAT_BYTES):
            dev, daddr = split_address(byte_addr + i, self.ndev, self.stripe)
            out += self.devices[dev][daddr:daddr + BEAT_BYTES]
        return bytes(out)

    def to_hex(self, path: Path) -> None:
        """Every device's words in turn, little-endian, one image."""
        words = [int.from_bytes(image[2 * w:2 * w + 2], "little") for image in self.devices for w in range(len(image) // 2)]
        write_hex(path, words, 16)


def emit_hpi_vectors(directory: Path, rng: np.random.Generator, ndev: int = 4, device_kb: int = 16,
                     transactions: int = 24, max_beats: int = 300) -> dict:
    """Random write and read bursts over the striped devices: the requests,
    the write data, the read data expected beat by beat, and the device
    images expected at the end."""
    directory.mkdir(parents=True, exist_ok=True)
    device_bytes = device_kb << 10
    total = ndev * device_bytes
    image = StripedImage(ndev, device_bytes)
    reqs, wbeats, rbeats = [], [], []
    for _ in range(transactions):
        beats = int(rng.integers(1, max_beats + 1))
        addr = int(rng.integers(0, (total - beats * BEAT_BYTES) // BEAT_BYTES + 1)) * BEAT_BYTES
        write = bool(rng.integers(0, 2)) or not wbeats
        if write:
            payload = bytes(rng.integers(0, 256, beats * BEAT_BYTES, dtype=np.uint8))
            image.write(addr, payload)
            wbeats += [int.from_bytes(payload[i:i + BEAT_BYTES], "little") for i in range(0, len(payload), BEAT_BYTES)]
        else:
            payload = image.read(addr, beats * BEAT_BYTES)
            rbeats += [int.from_bytes(payload[i:i + BEAT_BYTES], "little") for i in range(0, len(payload), BEAT_BYTES)]
        reqs.append((int(write) << 44) | (beats << 32) | addr)
    write_hex(directory / "reqs.hex", reqs, 48)
    write_hex(directory / "wdata.hex", wbeats or [0], 128)
    write_hex(directory / "expected_rdata.hex", rbeats or [0], 128)
    image.to_hex(directory / "expected_devs.hex")
    mrs = mode_registers()
    (directory / "params.json").write_text(json.dumps(dict(
        NDEV=ndev, DEV_WORDS=device_bytes // 2, N=transactions, NW=max(len(wbeats), 1), NR=max(len(rbeats), 1),
        MR0=mrs[0], MR4=mrs[4], MR8=mrs[8]), indent=2), encoding="utf-8")
    return json.loads((directory / "params.json").read_text(encoding="utf-8"))
