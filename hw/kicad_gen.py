"""Generate a KiCad 7 project for the appliance board from ``hw/board.yaml``.

    python -m hw.kicad_gen            # writes hw/kicad/appliance.kicad_{pro,pcb,sch} and report.md
    python -m hw.kicad_gen --check    # generate into a temporary directory and print the report

What it produces is a floorplan-level board, not a finished layout:

* the full-length PCIe card outline with the x8 edge fingers, bracket
  keep-outs and the 12V-2x6 auxiliary connector;
* every part of ``board.yaml`` placed (two rows of layer ASICs with their
  LPDDR5X devices, the head ASICs and FPGA at the bracket end, DDR4, clock
  generator, SFP cage, one core regulator block per ASIC and shared-rail
  regulator blocks) on generated footprints;
* placeholder BGA ball maps: the ASIC package has no ball map yet, so the
  generator assigns each interface to a package edge and keeps the same map
  for all ten ASICs;
* every net at the signal level, including the memory channels, PCIe and the
  management buses, so the schematic and the PCB agree;
* a twelve-layer stackup, ground and power zones, and the activation ring
  fully routed as 36-lane ribbons with escape vias, checked against the
  80 mm link limit;
* a report with link lengths, escape density and core-rail current density.

The memory, PCIe and management nets are present but unrouted: they need
length matching and signal-integrity work that an autorouter would only
imitate.  Decoupling capacitors and the small housekeeping parts are not
placed.

The files are written directly in the KiCad 7 S-expression formats, so the
generator needs only PyYAML; KiCad itself is used to open and check them.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

from hw import pinout
from hw.board import Board, Endpoint

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "kicad"
PROJECT = "appliance"

# Board geometry, millimetres.  Board frame: origin at the bottom-left corner
# of the card, x along the card length from the bracket, y up.  KiCad's frame
# has y down; conversion happens only when emitting.
CARD_LENGTH = 312.0
CARD_HEIGHT = 111.15
FINGER_ZONE_HEIGHT = 8.0        # card-edge contacts and keep-out along the bottom
FINGER_X0 = 12.5
BRACKET_KEEPOUT_X = 5.0
KICAD_ORIGIN = (20.0, 20.0)     # page offset so the outline sits inside the sheet

LINK_LAYERS = ("In2.Cu", "In5.Cu")   # straight hops on the first; bent hops split by escape depth over both
LINK_LAYER = LINK_LAYERS[0]
TRACK_WIDTH = 0.1
VIA_SIZE, VIA_DRILL = 0.4, 0.2
SMALL_VIA_SIZE = 0.3          # via-in-pad on the 0.35 mm memory balls
ESCAPE_OUT = 2.7              # outer-column escape via distance beyond the ball, mm
ESCAPE_IN = 0.5               # inner-column dogbone offset
LANE_PITCH = 0.5
MAX_LINK_MM = 80.0

# Twelve-layer stackup: signal / ground / link signals / 12 V and I/O rails /
# ground / memory signals x2 / ground / core rails / signals / ground / signal.
COPPER_LAYERS = ["F.Cu", "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu", "In5.Cu", "In6.Cu", "In7.Cu",
                 "In8.Cu", "In9.Cu", "In10.Cu", "B.Cu"]
LAYER_ROLES = {"F.Cu": "signal", "In1.Cu": "GND", "In2.Cu": "link signals", "In3.Cu": "12 V and I/O rails",
               "In4.Cu": "GND", "In5.Cu": "link signals (second half of bent hops), memory signals",
               "In6.Cu": "memory signals", "In7.Cu": "GND",
               "In8.Cu": "core rails", "In9.Cu": "signal", "In10.Cu": "GND", "B.Cu": "signal"}
GND_LAYERS = ["In1.Cu", "In4.Cu", "In7.Cu", "In10.Cu"]
TOP_RAIL_LAYER, CORE_RAIL_LAYER = "In3.Cu", "In8.Cu"

ROW_LETTERS = list("ABCDEFGHJKLMNPRTUVWY") + ["A" + c for c in "ABCDEFGHJKLMNPRTUVWY"]


def uid() -> str:
    return str(uuid.uuid4())


def fmt(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


# --------------------------------------------------------------------------
# Packages and parts
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Package:
    name: str
    cols: int
    rows: int
    pitch: float
    body_w: float
    body_h: float
    ball: float = 0.45

    def ball_xy(self, i: int, j: int) -> tuple[float, float]:
        """Local position of ball row i (0 = north), column j (0 = west), y up."""
        x = (j - (self.cols - 1) / 2) * self.pitch
        y = ((self.rows - 1) / 2 - i) * self.pitch
        return x, y

    def ball_name(self, i: int, j: int) -> str:
        return f"{ROW_LETTERS[i]}{j + 1}"


@dataclass
class Pad:
    name: str
    x: float            # local, y up, before rotation
    y: float
    net: str | None
    shape: str = "circle"
    size: tuple[float, float] = (0.45, 0.45)
    layers: str = '"F.Cu" "F.Paste" "F.Mask"'
    escape: tuple[str, int] | None = None   # (edge, depth) for ring-link balls


@dataclass
class Part:
    ref: str
    part_class: str
    footprint: str
    x: float
    y: float
    rotation: int
    body_w: float
    body_h: float
    pads: list[Pad] = field(default_factory=list)
    value: str = ""
    package: Package | None = None

    def local_to_board(self, x: float, y: float) -> tuple[float, float]:
        if self.rotation == 180:
            x, y = -x, -y
        elif self.rotation == 90:
            x, y = -y, x
        elif self.rotation == 270:
            x, y = y, -x
        return self.x + x, self.y + y

    def pad(self, name: str) -> Pad:
        for pad in self.pads:
            if pad.name == name:
                return pad
        raise KeyError(f"{self.ref} has no pad {name}")


PACKAGES = {
    "fpga": Package("FFVB676_26x26_P1.0", 26, 26, 1.0, 27.0, 27.0, 0.5),
    "lpddr5x": Package("FBGA315_15x21_P0.8", 15, 21, 0.8, 12.0, 16.8, 0.35),
    "ddr4": Package("FBGA96_8x12_P0.8", 8, 12, 0.8, 7.5, 10.6, 0.35),
}
MAX_ASIC_BODY_MM = 23.5   # the placement below (column pitch, row gap, memory offsets) assumes this


def asic_package(pin: pinout.Pinout) -> Package:
    """The ASIC package is whatever hw/pinout.py selected from the power model."""
    p = pin.package
    return Package(p.name, p.cols, p.rows, p.pitch, p.body, p.body, p.ball)


def expand_signals(signals: dict[str, int]) -> list[str]:
    """Signal names from an interface kind: dq: 32 -> DQ0..DQ31, dqs_pairs: 4 -> DQS0_P, DQS0_N, ..."""
    names = []
    for name, count in signals.items():
        base = name.upper()
        if base.endswith("_PAIRS") or base.endswith("_PAIR"):
            base = base.rsplit("_", 1)[0]
            for k in range(count):
                suffix = f"{k}" if count > 1 else ""
                names += [f"{base}{suffix}_P", f"{base}{suffix}_N"]
        elif count == 1:
            names.append(base)
        else:
            names += [f"{base}{k}" for k in range(count)]
    return names


# --------------------------------------------------------------------------
# Net naming
# --------------------------------------------------------------------------

def short_ref(ref: str) -> str:
    return ref.replace("U_", "")


def ring_net(hop: int, signal: str) -> str:
    return f"LINK{hop}_{signal}"


def memory_net(asic_ref: str, channel: str, signal: str) -> str:
    return f"{short_ref(asic_ref)}_{channel.upper()}_{signal}"


# --------------------------------------------------------------------------
# Ball maps
# --------------------------------------------------------------------------

def link_signals(board: Board) -> list[str]:
    return expand_signals(board.kinds["link"]["signals"])


def misc_net(ref: str, interface: str, signal: str) -> str:
    """Net names of the small interfaces: shared SPI and JTAG lines, per-chip selects,
    chained TDI/TDO (rewired in build_design), per-chip reference clock and straps."""
    me = short_ref(ref)
    if interface == "mgmt":
        return f"MGMT_{signal}_{me}" if signal in ("CS", "IRQ") else f"MGMT_{signal}"
    if interface == "jtag":
        return f"JTAG_{signal}_{me}" if signal in ("TDI", "TDO") else f"JTAG_{signal}"
    if interface == "refclk":
        return f"REFCLK_{me}_{signal[-1]}"
    return f"{signal}_{me}"


def asic_ball_map(board: Board, pin: pinout.Pinout, part_class: str, ref: str, hop_in: int, hop_out: int,
                  memory_channels: dict[str, str]) -> list[Pad]:
    """Pads of one ASIC from the derived ball map: the same map for every
    ASIC, with this chip's nets on it.  Head ASICs leave the memory balls and
    memory-PHY rails unconnected."""
    rails = set(board.classes[part_class]["rails"])
    pads = []
    for ball in pin.balls:
        net: str | None
        if ball.kind == "signal":
            if ball.interface == "link_in":
                net = ring_net(hop_in, ball.signal)
            elif ball.interface == "link_out":
                net = ring_net(hop_out, ball.signal)
            elif ball.interface.startswith("lpddr_"):
                net = memory_net(ref, ball.interface[-3:], ball.signal) if memory_channels.get(ball.interface) else None
            else:
                net = misc_net(ref, ball.interface, ball.signal)
        elif ball.kind == "ground":
            net = "GND"
        elif ball.interface == "VDD_CORE":
            net = f"VDD_CORE_{short_ref(ref)}"
        else:
            net = ball.interface if ball.interface in rails else None
        pads.append(Pad(ball.name, ball.x, ball.y, net, size=(pin.package.ball, pin.package.ball), escape=ball.escape))
    return pads


def fpga_ball_map(board: Board, hop_in: int, hop_out: int) -> list[Pad]:
    """FPGA: link_out on the east edge, link_in on the south edge (both
    assigned by the ribbon router), PCIe on the west edge, DDR4 x64 on the
    north rows, management and small interfaces on the south-west."""
    pkg = PACKAGES["fpga"]
    assigned: dict[tuple[int, int], tuple[str | None, tuple[str, int] | None]] = {}
    for k in range(36):
        assigned[(4 + k // 2, 25 if k % 2 == 0 else 24)] = (None, ("E", k % 2))
        assigned[(25 if k % 2 == 0 else 24, k // 2)] = (None, ("S", k % 2))     # west end of the south edge
    pcie = expand_signals(board.kinds["pcie_gen4_x8"]["signals"])
    west = iter((i, j) for i in range(5, 23) for j in range(2))
    for signal in pcie:
        assigned[next(west)] = (f"PCIE_{signal}", None)
    ddr = expand_signals(board.kinds["ddr4_x64"]["signals"])
    north = iter([(i, j) for i in range(4) for j in range(26)] + [(4, j) for j in range(20)])
    for signal in ddr:
        assigned[next(north)] = (f"DDR4_{signal}", None)
    misc = ["MGMT_SCLK", "MGMT_MOSI", "MGMT_MISO"]
    for ref in board.nets["mgmt"]["slaves"]:
        misc += [f"MGMT_CS_{short_ref(ref)}", f"MGMT_IRQ_{short_ref(ref)}"]
    misc += ["JTAG_TCK", "JTAG_TMS", "JTAG_TDI_FPGA", "JTAG_TDO_FPGA", "JTAG_TRST", "REFCLK_FPGA_P", "REFCLK_FPGA_N"]
    misc += [f"SFP_{s}" for s in expand_signals(board.kinds["sfp_plus"]["signals"])]
    misc += [f"QSPI{k}" for k in range(6)] + ["UART_TX", "UART_RX"]
    south = iter((i, j) for i in (22, 23) for j in range(24))
    for net in misc:
        assigned[next(south)] = (net, None)
    rails = [r for r in board.classes["fpga"]["rails"] if r != "VCCINT_0V85"]
    pads, idx = [], 0
    for i in range(pkg.rows):
        for j in range(pkg.cols):
            x, y = pkg.ball_xy(i, j)
            if (i, j) in assigned:
                net, escape = assigned[(i, j)]
            else:
                escape = None
                if (i + j) % 2 == 0:
                    net = "GND"
                elif (i * 5 + j * 3) % 9 == 0:
                    net = rails[idx % len(rails)]
                    idx += 1
                else:
                    net = "VCCINT_0V85"
            pads.append(Pad(pkg.ball_name(i, j), x, y, net, size=(pkg.ball, pkg.ball), escape=escape))
    return pads


def memory_ball_map(board: Board, package: Package, signal_nets: list[str], rails: list[str]) -> list[Pad]:
    """Signals fill the rows nearest the ASIC (south), rails and ground the rest."""
    pads, nets = [], iter(signal_nets)
    idx = 0
    for i in reversed(range(package.rows)):
        for j in range(package.cols):
            x, y = package.ball_xy(i, j)
            net = next(nets, None)
            if net is None:
                if (i + j) % 2 == 0:
                    net = "GND"
                else:
                    net = rails[idx % len(rails)]
                    idx += 1
            pads.append(Pad(package.ball_name(i, j), x, y, net, size=(package.ball, package.ball)))
    return pads


def block_pads(width: float, height: float, nets: list[str], pitch: float = 2.0,
               pad_size: tuple[float, float] = (1.2, 1.6)) -> list[Pad]:
    """Rectangular placeholder module: one row of pads along each long edge."""
    pads = []
    per_side = math.ceil(len(nets) / 2)
    for k, net in enumerate(nets):
        side = -1 if k < per_side else 1
        col = k if k < per_side else k - per_side
        x = (col - (per_side - 1) / 2) * pitch
        pads.append(Pad(str(k + 1), x, side * (height / 2 - pad_size[1] / 2 - 0.2), net, shape="rect", size=pad_size))
    return pads


def edge_finger_pads() -> list[Pad]:
    """PCIe x8 card-edge contacts: 49 positions per side, 1.0 mm pitch, key after pin 11."""
    a_side = {1: "PRSNT1#", 2: "+12V", 3: "+12V", 4: "GND", 5: "SMB_CLK", 6: "SMB_DAT", 7: "GND", 8: "+3V3",
              9: "+3V3", 10: "+3V3AUX", 11: "WAKE#", 12: "PCIE_RESERVED", 13: "GND", 14: "PCIE_REFCLK_P",
              15: "PCIE_REFCLK_N", 16: "GND", 17: "PCIE_PRSNT2#"}
    b_side = {1: "+12V", 2: "+12V", 3: "+12V", 4: "GND", 5: "SMB_CLK", 6: "SMB_DAT", 7: "GND", 8: "+3V3",
              9: "JTAG_TRST", 10: "+3V3AUX", 11: "WAKE#", 12: "PCIE_PERST#", 13: "GND"}
    pads = []
    lane = 0
    for n in range(1, 50):
        x = FINGER_X0 + (n - 1) * 1.0 + (2.0 if n > 11 else 0.0)
        for side, table, layers in (("A", a_side, '"F.Cu" "F.Mask"'), ("B", b_side, '"B.Cu" "B.Mask"')):
            net = table.get(n, "GND")
            if n >= 18 and side == "A":
                k = n - 18
                net = ["GND", f"PCIE_RX{{}}_P", f"PCIE_RX{{}}_N", "GND"][k % 4].format(min(lane + k // 4, 7))
            if n >= 14 and side == "B":
                k = n - 14
                net = ["GND", f"PCIE_TX{{}}_P", f"PCIE_TX{{}}_N", "GND"][k % 4].format(min(k // 4, 7))
            pads.append(Pad(f"{side}{n}", x - FINGER_ORIGIN[0], 3.25 - FINGER_ORIGIN[1], net, shape="rect",
                            size=(0.7, 5.5), layers=layers))
    return pads


FINGER_ORIGIN = (45.0, 4.0)   # footprint origin of the edge connector, inside the board


# --------------------------------------------------------------------------
# Design assembly
# --------------------------------------------------------------------------

@dataclass
class Track:
    layer: str
    net: str
    points: list[tuple[float, float]]
    width: float = TRACK_WIDTH

    def length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.points, self.points[1:]))


@dataclass
class Via:
    x: float
    y: float
    net: str
    size: float = VIA_SIZE
    drill: float = VIA_DRILL


@dataclass
class Zone:
    layer: str
    net: str
    polygon: list[tuple[float, float]]
    priority: int = 0


@dataclass
class Design:
    board: Board
    pinout: pinout.Pinout | None = None
    parts: list[Part] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    hop_lengths: dict[str, tuple[float, float]] = field(default_factory=dict)   # hop -> (min, max) mm
    notes: list[str] = field(default_factory=list)

    def part(self, ref: str) -> Part:
        for part in self.parts:
            if part.ref == ref:
                return part
        raise KeyError(ref)

    def nets(self) -> list[str]:
        names: set[str] = set()
        for part in self.parts:
            names.update(pad.net for pad in part.pads if pad.net)
        for track in self.tracks:
            names.add(track.net)
        return sorted(names)


def ring_hops(board: Board) -> list[tuple[str, str]]:
    return [(a.component, b.component) for a, b in board.ring()]


# Placement (board frame, y up).  Row A is unrotated and flows left to
# right; row B is rotated 180 degrees so the ring flows back right to left
# with link_in facing the previous chip.
ROW_Y = {"A": 72.0, "B": 41.0}
COL_X = [100.0, 160.0, 220.0, 273.0]
FPGA_XY = (42.0, 84.5)
HEAD_XY = {"U_H0": (70.0, 41.0), "U_H1": (40.0, 41.0)}   # under the FPGA, so the closing hop stays short
VRM_W, VRM_H = 26.0, 5.5
VRM_Y = 56.5                     # one band of core regulators between the rows
REG_W, REG_H = 20.0, 6.5
PATH_MARGIN = 0.6                # a ribbon path starts this far past the outer escape vias
BEND_RUN = 12.5                  # straight run needed before or after a corner: taper plus half the ribbon width


def escape_reach(package: Package) -> float:
    """Distance from a package centre to a ribbon path end: past the outer escape vias."""
    return (package.cols - 1) / 2 * package.pitch + ESCAPE_OUT + PATH_MARGIN


def build_design(board: Board) -> Design:
    pin = pinout.derive(board)
    asic_pkg = asic_package(pin)
    if asic_pkg.body_w > MAX_ASIC_BODY_MM:
        raise ValueError(f"the derived package {asic_pkg.name} is {asic_pkg.body_w:.0f} mm; the placement in "
                         f"hw/kicad_gen.py (column pitch, row gap, memory offsets) fits at most {MAX_ASIC_BODY_MM} mm. "
                         "Lower package_selection.rated_tokens_per_second or rework the floorplan.")
    design = Design(board, pinout=pin)
    hops = ring_hops(board)
    hop_in = {sink: h for h, (_, sink) in enumerate(hops)}
    hop_out = {source: h for h, (source, _) in enumerate(hops)}
    memory_of: dict[str, dict[str, str]] = {}
    for a, b in board.nets["memory"]["channels"]:
        asic, device = Endpoint.parse(a), Endpoint.parse(b)
        memory_of.setdefault(asic.component, {})[asic.interface] = device.component

    # ASICs.
    for ref in board.instances("layer_asic") + board.instances("head_asic"):
        comp = board.components[ref]
        if ref in HEAD_XY:
            x, y = HEAD_XY[ref]
            rotation = 180
        else:
            row = comp["place"].split("row ")[1][0]
            col = int(comp["place"].split("col ")[1])
            x, y = COL_X[col], ROW_Y[row]
            rotation = 180 if row == "B" else 0
        pkg = asic_pkg
        pads = asic_ball_map(board, pin, board.class_of(ref), ref, hop_in[ref], hop_out[ref], memory_of.get(ref, {}))
        design.parts.append(Part(ref, board.class_of(ref), pkg.name, x, y, rotation, pkg.body_w, pkg.body_h, pads,
                                 value=board.class_of(ref), package=pkg))
        # Core regulator block between the rows, beside its ASIC.
        vrot = 0
        if ref in HEAD_XY:
            vx, vy = (66.0 if ref == "U_H0" else 38.0), 14.0   # heads have no memory below them; the regulator goes there
        elif ref == "U_A4":
            # The end column's row-B regulator would sit in the row-change
            # ribbon's channel; it stands on end beside the card edge instead.
            vx, vy, vrot = CARD_LENGTH - 7.0, VRM_Y, 90
        else:
            # Row A's regulator left of its chip, row B's to the right, in one band between the rows.
            # The end column is closer to its neighbour, so its row-A regulator sits nearer the chip.
            left = 11.0 if col == 3 else 14.5
            vx, vy = (x - left, VRM_Y) if rotation == 0 else (x + 14.5, VRM_Y)
        core = f"VDD_CORE_{short_ref(ref)}"
        nets = ["+12V"] * 4 + [core] * 6 + ["GND"] * 6 + ["PMB_SCL", "PMB_SDA", f"VRM_EN_{short_ref(ref)}"]
        design.parts.append(Part(f"VRM_{short_ref(ref)}", "vrm_core", f"VRM_MODULE_{VRM_W:.0f}x{VRM_H:.0f}", vx, vy, vrot,
                                 VRM_W, VRM_H, block_pads(VRM_W, VRM_H, nets, pitch=2.3, pad_size=(1.2, 1.2)),
                                 value=f"core VRM {core}"))

    # LPDDR5X beside their ASICs: row A above, row B below (rotated rows face them).
    lpddr_signals = expand_signals(board.kinds["lpddr5x_x32"]["signals"])
    for asic_ref, channels in memory_of.items():
        asic = design.part(asic_ref)
        for index, (channel, device) in enumerate(sorted(channels.items())):
            dx = -9.5 if index == 0 else 9.5
            if asic.rotation == 0:
                x, y = asic.x + dx, asic.y + asic.body_h / 2 + 2.0 + 6.0
            else:
                x, y = asic.x - dx, asic.y - asic.body_h / 2 - 2.0 - 6.0
            pkg = PACKAGES["lpddr5x"]
            nets = [memory_net(asic_ref, channel[-3:], s) for s in lpddr_signals]
            pads = memory_ball_map(board, pkg, nets, [r for r in board.classes["lpddr5x"]["rails"]])
            design.parts.append(Part(device, "lpddr5x", pkg.name, x, y, 90, pkg.body_w, pkg.body_h, pads,
                                     value="LPDDR5X x32", package=pkg))

    # FPGA, DDR4, clock, SFP, connectors, shared regulators.
    fpga = board.instances("fpga")[0]
    pkg = PACKAGES["fpga"]
    design.parts.append(Part(fpga, "fpga", pkg.name, FPGA_XY[0], FPGA_XY[1], 0, pkg.body_w, pkg.body_h,
                             fpga_ball_map(board, hop_in[fpga], hop_out[fpga]), value="FPGA", package=pkg))
    ddr_signals = expand_signals(board.kinds["ddr4_x16"]["signals"])
    for n, device in enumerate(board.instances("ddr4")):
        pkg = PACKAGES["ddr4"]
        nets = []
        for s in ddr_signals:
            if s.startswith("DQS"):
                idx = int(s[3:].split("_")[0])
                nets.append(f"DDR4_DQS{2 * n + idx}_{s[-1]}")
            elif s.startswith("DQ"):
                nets.append(f"DDR4_DQ{16 * n + int(s[2:])}")
            elif s.startswith("DM"):
                nets.append(f"DDR4_DM{2 * n + int(s[2:])}")
            else:
                nets.append(f"DDR4_{s}")
        pads = memory_ball_map(board, pkg, nets, board.classes["ddr4"]["rails"])
        x, y = (30.0 + 8.5 * n, 105.0)      # a row above the FPGA, clear of the ring ribbon
        design.parts.append(Part(device, "ddr4", pkg.name, x, y, 0, pkg.body_w, pkg.body_h, pads,
                                 value="DDR4 x16", package=pkg))
    clk_nets = ["VDD_IO_1V8", "GND"] + [f"REFCLK_{short_ref(r)}_{p}" for r in board.nets["refclk"]["sinks"] for p in "PN"]
    design.parts.append(Part("U_CLK", "clock_gen", "QFN64_9x9", 10.0, 20.0, 0, 9.0, 9.0,
                             block_pads(9.0, 9.0, clk_nets[:24], pitch=0.7, pad_size=(0.3, 0.9)), value="Si5345"))
    sfp_nets = [f"SFP_{s}" for s in expand_signals(board.kinds["sfp_plus"]["signals"])] + ["VDD_3V3"] * 2 + ["GND"] * 4
    design.parts.append(Part("J_SFP", "sfp_cage", "SFP_CAGE", 12.0, 85.0, 0, 14.0, 50.0,
                             block_pads(50.0, 14.0, sfp_nets, pitch=2.0), value="SFP+ cage"))
    design.parts.append(Part("J_PCIE", "pcie_edge", "PCIE_X8_EDGE", FINGER_ORIGIN[0], FINGER_ORIGIN[1], 0, 0.0, 0.0,
                             edge_finger_pads(), value="PCIe x8 edge"))
    aux_nets = ["+12V"] * 6 + ["GND"] * 6 + ["AUX_SENSE0", "AUX_SENSE1", "AUX_SENSE2", "AUX_SENSE3"]
    design.parts.append(Part("J_AUX", "aux_power", "12V-2x6", 302.0, 105.0, 0, 18.0, 10.0,
                             block_pads(18.0, 10.0, aux_nets, pitch=2.0), value="12V-2x6 aux"))
    shared = [("REG_1V8", "VDD_IO_1V8", (130.0, 105.0)), ("REG_1V05", "VDD2H_1V05", (190.0, 100.0)),
              ("REG_0V9", "VDD2L_0V9", (130.0, 11.5)), ("REG_0V3", "VDDQ_0V3", (190.0, 11.5)),
              ("REG_FPGA", "VCCINT_0V85", (60.0, 59.0)), ("REG_DDR", "VDD_1V2", (68.0, 95.0)),
              ("REG_3V3", "VDD_3V3", (71.0, 106.0))]
    for ref, rail, (x, y) in shared:
        nets = ["+12V"] * 3 + [rail] * 4 + ["GND"] * 3
        design.parts.append(Part(ref, "regulator", f"REG_MODULE_{REG_W:.0f}x{REG_H:.0f}", x, y, 0, REG_W, REG_H,
                                 block_pads(REG_W, REG_H, nets, pitch=1.8, pad_size=(1.2, 1.2)), value=f"{rail} regulator"))
    # Aliases so the memory-only rails have a source too.
    design.part("REG_1V8").pads[3].net = "VDD1_1V8"
    design.part("REG_DDR").pads[5].net = "VPP_2V5"
    design.part("REG_FPGA").pads[4].net = "VCCAUX_1V8"
    design.part("REG_FPGA").pads[5].net = "MGTAVCC"
    design.part("REG_FPGA").pads[6].net = "MGTAVTT"
    design.part("REG_1V05").pads[4].net = "VCCO_1V2"
    design.part("REG_1V8").pads[4].net = "VCCO_1V8"
    design.part("REG_0V9").pads[4].net = "VDD_PLL_0V9"

    # Daisy chains: JTAG TDO -> next TDI, mode straps.
    chain = board.nets["jtag"]["chain"]
    for a, b in zip(chain, chain[1:]):
        for pad in design.part(b).pads:
            if pad.net == f"JTAG_TDI_{short_ref(b)}":
                pad.net = f"JTAG_TDO_{short_ref(a)}"

    route_ring(design)
    add_zones(design)
    add_power_vias(design)
    return design


def is_rail(net: str | None, board: Board) -> bool:
    if not net:
        return False
    return net == "GND" or net == "+12V" or net.startswith("VDD_CORE_") or net in board.data["power_tree"]["rails"]


def add_power_vias(design: Design) -> None:
    """Via-in-pad on every ground and rail pad (as an FCBGA on an HDI board would
    have) so the planes connect to the packages, regulators and connectors."""
    def inside(zone: Zone, x: float, y: float) -> bool:
        xs, ys = [p[0] for p in zone.polygon], [p[1] for p in zone.polygon]
        return min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys)

    for part in design.parts:
        if part.part_class == "pcie_edge":
            continue
        for pad in part.pads:
            if not is_rail(pad.net, design.board):
                continue
            x, y = part.local_to_board(pad.x, pad.y)
            # Only where a zone of that net will actually pick the via up.
            if any(zone.net == pad.net and inside(zone, x, y) for zone in design.zones):
                size = VIA_SIZE if min(pad.size) >= VIA_SIZE else SMALL_VIA_SIZE
                design.vias.append(Via(x, y, pad.net, size=size, drill=size / 2))


# --------------------------------------------------------------------------
# Ring routing: escape vias plus orthogonal ribbons of 36 lanes
# --------------------------------------------------------------------------

def escape_via_local(pad: Pad, pitch: float) -> tuple[float, float]:
    """Escape via of a link ball: outer column straight out past the package
    edge, inner column a dogbone half a pitch diagonally, so the lanes of the
    two columns interleave at half the ball pitch."""
    edge, depth = pad.escape
    dog = pitch / 2
    if edge == "E":
        return (pad.x + ESCAPE_OUT, pad.y) if depth == 0 else (pad.x + dog, pad.y + dog)
    if edge == "W":
        return (pad.x - ESCAPE_OUT, pad.y) if depth == 0 else (pad.x - dog, pad.y + dog)
    if edge == "S":
        # Mirrored dogbone: the south edge receives lanes from a rotated chip,
        # whose inner-column lanes sit half a pitch the other way.
        return (pad.x, pad.y - ESCAPE_OUT) if depth == 0 else (pad.x - dog, pad.y - dog)
    return (pad.x, pad.y + ESCAPE_OUT) if depth == 0 else (pad.x + dog, pad.y + dog)


def escape_via(part: Part, pad: Pad) -> tuple[float, float]:
    return part.local_to_board(*escape_via_local(pad, part.package.pitch))


def escape_pads(part: Part, edge_local: str) -> list[Pad]:
    return [pad for pad in part.pads if pad.escape and pad.escape[0] == edge_local]


def lane_centre(part: Part, edge_local: str) -> tuple[float, float]:
    """Board-frame centre of a link port's escape vias; a ribbon path ends here."""
    vias = [escape_via(part, pad) for pad in escape_pads(part, edge_local)]
    return sum(v[0] for v in vias) / len(vias), sum(v[1] for v in vias) / len(vias)


BODY_LANE_PITCH = 0.25   # lane pitch in the body of a bent ribbon: a bent hop is split by escape depth
                         # onto two layers, so each half has one-ball-pitch lanes at the vias and
                         # converges to this in its body (0.1 mm tracks, 0.15 mm gaps)
TAPER_MM = 10.0          # along-path length over which the lanes converge or spread (keeps the
                         # outermost lane's diagonal shallow enough for 0.1 mm clearance)


def corner_at(b0: tuple[float, float], ua: tuple[float, float], ub: tuple[float, float],
              da: float, db: float) -> tuple[float, float]:
    """Corner of two orthogonal offset lines: lateral ``da`` on the segment with
    direction ``ua`` and ``db`` on the one with ``ub``, meeting at centreline point ``b0``."""
    na, nb = (-ua[1], ua[0]), (-ub[1], ub[0])
    pa = (b0[0] + da * na[0], b0[1] + da * na[1])
    pb = (b0[0] + db * nb[0], b0[1] + db * nb[1])
    return (pb[0], pa[1]) if abs(ua[0]) > 0.5 else (pa[0], pb[1])


def ribbon(path: list[tuple[float, float]], starts: dict[str, tuple[float, float]],
           end_vias: list[tuple[str | None, tuple[float, float]]], body_compression: float = 1.0,
           end_scale: float = 1.0) -> tuple[dict[str, list[tuple[float, float]]], dict[str, str]]:
    """Route each start via along the orthogonal centreline ``path`` at its own
    lateral offset and finish on the end via lying on the same lane.

    Between the first and last waypoints of a multi-segment path the lanes
    converge to ``body_compression`` of their offset (when the first and last
    segments are long enough for a shallow taper), so a bend costs the outer
    lane little.  ``end_scale`` is the ratio of the end part's lane pitch to
    the start part's, for hops between packages of different ball pitch.
    Returns the polylines per net and the mapping end-via-pad -> net.
    """
    def unit(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        return dx / length, dy / length

    segments = list(zip(path, path[1:]))
    lengths = [math.dist(a, b) for a, b in segments]
    compress = 1.0
    if len(segments) >= 2 and lengths[0] >= TAPER_MM and lengths[-1] >= TAPER_MM:
        compress = body_compression
    taper = compress < 1.0 or end_scale != 1.0
    tracks: dict[str, list[tuple[float, float]]] = {}
    assignment: dict[str, str] = {}
    used: set[int] = set()
    for net, start in starts.items():
        u0 = unit(*segments[0])
        n0 = (-u0[1], u0[0])                              # left normal
        d = (start[0] - path[0][0]) * n0[0] + (start[1] - path[0][1]) * n0[1]
        dc = d * compress
        d_end = d * end_scale
        points = [start]
        if taper:
            points.append((path[0][0] + d * n0[0], path[0][1] + d * n0[1]))
            if compress < 1.0:
                points.append((path[0][0] + u0[0] * TAPER_MM + dc * n0[0], path[0][1] + u0[1] * TAPER_MM + dc * n0[1]))
        for (a0, b0), (a1, b1) in zip(segments, segments[1:]):
            points.append(corner_at(b0, unit(a0, b0), unit(a1, b1), dc, dc))
        ul = unit(*segments[-1])
        nl = (-ul[1], ul[0])
        if taper:
            if compress < 1.0:
                points.append((path[-1][0] - ul[0] * TAPER_MM + dc * nl[0], path[-1][1] - ul[1] * TAPER_MM + dc * nl[1]))
            points.append((path[-1][0] + d_end * nl[0], path[-1][1] + d_end * nl[1]))
        # End via: the one on this lane (same lateral offset on the last segment).
        best, best_err = None, 1e9
        for index, (pad_name, via) in enumerate(end_vias):
            if index in used:
                continue
            lateral = (via[0] - path[-1][0]) * nl[0] + (via[1] - path[-1][1]) * nl[1]
            err = abs(lateral - d_end)
            if err < best_err:
                best, best_err = index, err
        if best is None or best_err > 1e-3:
            raise ValueError(f"no end via on the lane of {net} (offset {d:.3f}, nearest error {best_err:.3f})")
        used.add(best)
        pad_name, via = end_vias[best]
        points.append(via)
        tracks[net] = points
        assignment[pad_name] = net
    return tracks, assignment


def hop_path(design: Design, source: Part, sink: Part) -> tuple[list[tuple[float, float]], str, str]:
    """Centreline and the local edges used for each hop.  Returns (path, source_edge, sink_edge)."""
    # A path starts just past the outer escape vias; when it bends it needs
    # TAPER_MM plus half the compressed ribbon width before the first corner.
    out_x = escape_reach(source.package)
    in_x = escape_reach(sink.package)
    # Paths run through the centre of each port's escape vias, so the lane
    # offsets are symmetric at both ends whatever rows the ball map uses.
    if source.part_class == "fpga":
        # FPGA east edge to A0 west edge with a jog to A0's row.
        x0 = source.x + out_x
        x1 = sink.x - in_x
        mid = (x0 + x1) / 2
        y0 = lane_centre(source, "E")[1]
        y1 = lane_centre(sink, "W")[1]
        return [(x0, y0), (mid, y0), (mid, y1), (x1, y1)], "E", "W"
    if sink.part_class == "fpga":
        # H1 (rotated: link_out on its physical west) to the FPGA south edge.
        x0 = source.x - out_x
        y0 = lane_centre(source, "E")[1]
        channel_x = x0 - BEND_RUN
        y1 = sink.y - in_x
        x1 = lane_centre(sink, "S")[0]
        y_mid = y1 - BEND_RUN
        return [(x0, y0), (channel_x, y0), (channel_x, y_mid), (x1, y_mid), (x1, y1)], "E", "S"
    y0, y1 = lane_centre(source, "E")[1], lane_centre(sink, "W")[1]
    if source.rotation == sink.rotation:
        if abs(y0 - y1) > 1e-6:
            raise ValueError(f"{source.ref} -> {sink.ref}: ports are not aligned ({y0:.2f} vs {y1:.2f})")
        if source.rotation == 0:
            return [(source.x + out_x, y0), (sink.x - in_x, y1)], "E", "W"
        return [(source.x - out_x, y0), (sink.x + in_x, y1)], "E", "W"
    # Row change: east out of the last row-A chip, south, west into the rotated chip below.
    channel_x = source.x + out_x + BEND_RUN
    return [(source.x + out_x, y0), (channel_x, y0), (channel_x, y1), (sink.x + in_x, y1)], "E", "W"


def route_ring(design: Design) -> None:
    board = design.board
    for hop, (src_ref, dst_ref) in enumerate(ring_hops(board)):
        source, sink = design.part(src_ref), design.part(dst_ref)
        path, src_edge, dst_edge = hop_path(design, source, sink)
        src_pads = escape_pads(source, src_edge)
        dst_pads = escape_pads(sink, dst_edge)
        # Escape stubs and vias on both ends.
        for part, pads in ((source, src_pads), (sink, dst_pads)):
            for pad in pads:
                vx, vy = escape_via(part, pad)
                px, py = part.local_to_board(pad.x, pad.y)
                net = pad.net or f"__{part.ref}_{pad.name}"
                design.vias.append(Via(vx, vy, net))
                design.tracks.append(Track("F.Cu", net, [(px, py), (vx, vy)]))
        known_is_source = all(pad.net for pad in src_pads)
        if not known_is_source:
            path = list(reversed(path))
            src_pads, dst_pads = dst_pads, src_pads
            source, sink = sink, source
        # A straight hop is one ribbon on the first link layer.  A bent hop is
        # split by escape depth (outer-column lanes, dogbone lanes) onto the
        # two link layers so each half is narrow enough to bend within the
        # length limit.  Lanes of one depth are one ball pitch apart, and the
        # end part may have a different pitch.
        groups = [(LINK_LAYERS[0], (0, 1))] if len(path) == 2 else [(LINK_LAYERS[0], (0,)), (LINK_LAYERS[1], (1,))]
        body_compression = BODY_LANE_PITCH / source.package.pitch
        end_scale = sink.package.pitch / source.package.pitch
        lengths = []
        for layer, depths in groups:
            starts = {pad.net: escape_via(source, pad) for pad in src_pads if pad.escape[1] in depths}
            ends = [(pad.name, escape_via(sink, pad)) for pad in dst_pads if pad.escape[1] in depths]
            tracks, assignment = ribbon(path, starts, ends, body_compression, end_scale)
            for pad_name, net in assignment.items():
                pad = sink.pad(pad_name)
                if pad.net is None:
                    pad.net = net
                    for via in design.vias:
                        if via.net == f"__{sink.ref}_{pad_name}":
                            via.net = net
                    for track in design.tracks:
                        if track.net == f"__{sink.ref}_{pad_name}":
                            track.net = net
                elif pad.net != net:
                    raise ValueError(f"hop {hop}: lane of {net} arrives at {sink.ref}.{pad_name} carrying {pad.net}")
            for net, points in tracks.items():
                track = Track(layer, net, points)
                design.tracks.append(track)
                lengths.append(track.length())
        design.hop_lengths[f"{src_ref} -> {dst_ref}"] = (min(lengths), max(lengths))


# --------------------------------------------------------------------------
# Zones
# --------------------------------------------------------------------------

def rect(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def add_zones(design: Design) -> None:
    outline = rect(1.0, FINGER_ZONE_HEIGHT, CARD_LENGTH - 1.0, CARD_HEIGHT - 1.0)
    for layer in GND_LAYERS:
        design.zones.append(Zone(layer, "GND", outline))
    # 12 V from the aux connector and the slot across the top rail layer; the
    # I/O rail as a band along the top where the memories and regulators sit.
    design.zones.append(Zone(TOP_RAIL_LAYER, "+12V", outline, priority=0))
    # Memory PHY rail under both memory bands, I/O rail along the regulator strip.
    design.zones.append(Zone(TOP_RAIL_LAYER, "VDD2H_1V05", rect(85.0, FINGER_ZONE_HEIGHT, 300.0, 21.0), priority=1))
    design.zones.append(Zone(TOP_RAIL_LAYER, "VDD2H_1V05", rect(85.0, 92.0, 300.0, 105.0), priority=1))
    design.zones.append(Zone(TOP_RAIL_LAYER, "VDD_IO_1V8", rect(85.0, 105.5, 300.0, 110.0), priority=1))
    # Core rails: one island per ASIC covering the package and its regulator.
    for part in design.parts:
        if part.part_class in ("layer_asic", "head_asic"):
            vrm = design.part(f"VRM_{short_ref(part.ref)}")
            x0 = min(part.x - part.body_w / 2, vrm.x - vrm.body_w / 2) - 1.0
            x1 = max(part.x + part.body_w / 2, vrm.x + vrm.body_w / 2) + 1.0
            y0 = min(part.y - part.body_h / 2, vrm.y - vrm.body_h / 2) - 1.0
            y1 = max(part.y + part.body_h / 2, vrm.y + vrm.body_h / 2) + 1.0
            design.zones.append(Zone(CORE_RAIL_LAYER, f"VDD_CORE_{short_ref(part.ref)}", rect(x0, y0, x1, y1), 1))
    fpga = design.part(design.board.instances("fpga")[0])
    design.zones.append(Zone(CORE_RAIL_LAYER, "VCCINT_0V85",
                             rect(fpga.x - 15.0, fpga.y - 15.0, fpga.x + 15.0, fpga.y + 27.0), 1))


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

def core_current_density(design: Design) -> list[tuple[str, float, float, float]]:
    """(ASIC, amps at 50K tok/s, plane cross-section mm2 per layer, A/mm2 on one 2 oz plane)."""
    board = design.board
    rows = []
    amps = board.core_current_a("layer_asic", 50_000)
    for part in design.parts:
        if part.part_class == "layer_asic":
            section = part.body_w * 0.07     # one 2 oz plane across the package width
            rows.append((part.ref, amps, section, amps / section))
    return rows


def report_markdown(design: Design) -> str:
    board = design.board
    lines = ["# KiCad floorplan report", "",
             "Generated by `python -m hw.kicad_gen` from `hw/board.yaml`. Do not edit.", "",
             f"Card {CARD_LENGTH:.0f} x {CARD_HEIGHT:.2f} mm, {len(COPPER_LAYERS)} copper layers, "
             f"{len(design.parts)} footprints, {len(design.nets())} nets, {len(design.tracks)} track segments, "
             f"{len(design.vias)} vias, {len(design.zones)} zones.", "",
             "## Stackup", "", "| Layer | Role |", "| --- | --- |"]
    lines += [f"| {layer} | {role} |" for layer, role in LAYER_ROLES.items()]
    pin = design.pinout
    pitch = pin.package.pitch
    lines += ["", "## Activation ring", "", f"36 lanes per hop, {TRACK_WIDTH} mm tracks. A straight hop is one "
              f"ribbon on {LINK_LAYERS[0]} at {pitch / 2} mm lane pitch (outer-column and dogbone vias interleaved). "
              f"A bent hop is split by escape depth into two 18-lane ribbons on {LINK_LAYERS[0]} and {LINK_LAYERS[1]}, "
              f"each tapering from {pitch} mm at the vias to {BODY_LANE_PITCH} mm in its body so the corners cost "
              f"the outer lane little; the FPGA end rescales to its own {PACKAGES['fpga'].pitch} mm pitch. "
              f"Limit {MAX_LINK_MM:.0f} mm per `board.yaml`.", "",
              "| Hop | Shortest lane (mm) | Longest lane (mm) | Within limit |", "| --- | ---: | ---: | --- |"]
    for hop, (lo, hi) in design.hop_lengths.items():
        lines.append(f"| {hop} | {lo:.1f} | {hi:.1f} | {'yes' if hi <= MAX_LINK_MM else 'NO'} |")
    pkg = pin.package
    need = pin.requirements
    deep = sum(1 for b in pin.balls if b.kind == "signal" and
               min(b.row, b.col, pkg.rows - 1 - b.row, pkg.cols - 1 - b.col) >= int(design.board.data["package_selection"]["signal_rows"]))
    memory_rows = 1 + max(b.row for b in pin.balls if b.kind == "signal" and b.interface.startswith("lpddr"))
    lines += ["", "## ASIC package and escape density", "",
              f"{pkg.name}: {pkg.cols}x{pkg.rows} balls at {pkg.pitch} mm, {pkg.body:.0f} mm body, selected by "
              f"`hw/pinout.py` for {need.core_amps:.0f} A of core current at {need.rated_tokens_per_second:.0f} tokens/s "
              f"({need.total} balls needed; see `hw/pinout/report.md`). "
              f"Per edge: 36 link signals on two columns of 18 rows ({36 / (18 * pkg.pitch):.1f} signals per mm of edge); "
              f"130 LPDDR signals on the north {memory_rows} rows, {130 / pkg.body:.1f} per mm of edge, of which {deep} "
              f"sit deeper than the outer {design.board.data['package_selection']['signal_rows']} rows and need "
              "microvias or a build-up layer pair to escape. The memory nets are not routed here.", "",
              "## Core rail current", "",
              "One 2 oz (70 um) plane across the package width, at 50K tokens/s from the board power model:", "",
              "| ASIC | Current (A) | Section per plane (mm2) | A/mm2 on one plane | Planes for 30 A/mm2 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for ref, amps, section, density in core_current_density(design):
        lines.append(f"| {ref} | {amps:.0f} | {section:.2f} | {density:.0f} | {math.ceil(density / 30)} |")
    lines += ["", "The core rail therefore needs the regulator directly beside the package with several "
              "plane layers or thick copper in between; the single In8.Cu island here is a placeholder.", "",
              "## Not done here", "",
              "* LPDDR5X, DDR4 and PCIe are present as nets and unrouted (length matching and impedance work).",
              "* No decoupling capacitors, no VRM internals, no thermal vias, no mounting holes.",
              "* The ASIC ball map is a placeholder shared by all ten parts; the real one comes from the package design.",
              "* The FPGA link pins are assigned by the ribbon router; every other FPGA pin is a placeholder."]
    lines += ["", "## Notes", ""] + [f"* {note}" for note in design.notes]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# KiCad writers
# --------------------------------------------------------------------------

def kx(x: float) -> float:
    return x + KICAD_ORIGIN[0]


def ky(y: float) -> float:
    return KICAD_ORIGIN[1] + (CARD_HEIGHT - y)


def write_pcb(design: Design, path: Path) -> None:
    nets = ["", *design.nets()]
    net_id = {name: index for index, name in enumerate(nets)}
    out = ["(kicad_pcb (version 20221018) (generator asic_kicad_gen)", "",
           "  (general (thickness 1.6))", '  (paper "A0")']
    layers = []
    for index, name in enumerate(COPPER_LAYERS):
        kid = 0 if index == 0 else 31 if index == len(COPPER_LAYERS) - 1 else index
        kind = "power" if LAYER_ROLES[name] in ("GND", "12 V and I/O rails", "core rails") else "signal"
        layers.append(f'    ({kid} "{name}" {kind})')
    layers += [f'    ({kid} "{name}" user)' for kid, name in
               ((32, "B.Adhes"), (33, "F.Adhes"), (34, "B.Paste"), (35, "F.Paste"), (36, "B.SilkS"), (37, "F.SilkS"),
                (38, "B.Mask"), (39, "F.Mask"), (40, "Dwgs.User"), (41, "Cmts.User"), (42, "Eco1.User"),
                (43, "Eco2.User"), (44, "Edge.Cuts"), (45, "Margin"), (46, "B.CrtYd"), (47, "F.CrtYd"),
                (48, "B.Fab"), (49, "F.Fab"))]
    out += ["  (layers", *layers, "  )", ""]
    out += ["  (setup", "    (stackup"]
    for index, name in enumerate(COPPER_LAYERS):
        out.append(f'      (layer "{name}" (type "copper") (thickness 0.035))')
        if index < len(COPPER_LAYERS) - 1:
            kind = "prepreg" if index % 2 == 0 else "core"
            out.append(f'      (layer "dielectric {index + 1}" (type "{kind}") (thickness 0.1) (material "FR4") (epsilon_r 4.2))')
        else:
            pass
    out += ["      (copper_finish \"ENIG\")", "    )", "    (pad_to_mask_clearance 0)", "  )", ""]
    for name, index in net_id.items():
        out.append(f'  (net {index} "{name}")')
    out.append("")
    # Outline and finger zone.
    corners = [(0.0, 0.0), (CARD_LENGTH, 0.0), (CARD_LENGTH, CARD_HEIGHT), (0.0, CARD_HEIGHT)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        out.append(f'  (gr_line (start {fmt(kx(a[0]))} {fmt(ky(a[1]))}) (end {fmt(kx(b[0]))} {fmt(ky(b[1]))}) '
                   f'(stroke (width 0.1) (type default)) (layer "Edge.Cuts") (tstamp {uid()}))')
    out.append(f'  (gr_rect (start {fmt(kx(0))} {fmt(ky(FINGER_ZONE_HEIGHT))}) (end {fmt(kx(90))} {fmt(ky(0))}) '
               f'(stroke (width 0.1) (type dash)) (fill none) (layer "Dwgs.User") (tstamp {uid()}))')
    out.append(f'  (gr_rect (start {fmt(kx(0))} {fmt(ky(CARD_HEIGHT))}) (end {fmt(kx(BRACKET_KEEPOUT_X))} {fmt(ky(0))}) '
               f'(stroke (width 0.1) (type dash)) (fill none) (layer "Dwgs.User") (tstamp {uid()}))')
    out.append(f'  (gr_text "{PROJECT}: floorplan generated from hw/board.yaml" (at {fmt(kx(150))} {fmt(ky(CARD_HEIGHT + 6))}) '
               f'(layer "Cmts.User") (tstamp {uid()}) (effects (font (size 3 3) (thickness 0.4))))')
    out.append("")
    # Footprints.
    for part in design.parts:
        px, py = kx(part.x), ky(part.y)
        out.append(f'  (footprint "appliance:{part.footprint}" (layer "F.Cu") (tstamp {uid()}) (at {fmt(px)} {fmt(py)})')
        out.append(f'    (property "Sheetfile" "{PROJECT}.kicad_sch")')
        out.append(f'    (attr smd)')
        # Reference and value inside the body so nothing is clipped at the card edge.
        out.append(f'    (fp_text reference "{part.ref}" (at 0 0) (layer "F.SilkS") (tstamp {uid()})'
                   f' (effects (font (size 1.5 1.5) (thickness 0.2))))')
        out.append(f'    (fp_text value "{part.value}" (at 0 2) (layer "F.Fab") (tstamp {uid()})'
                   f' (effects (font (size 1 1) (thickness 0.15))))')
        w, h = part.body_w, part.body_h
        if part.rotation in (90, 270):
            w, h = h, w
        if w > 0:
            for layer, grow in (("F.SilkS", 0.0), ("F.CrtYd", 0.5), ("F.Fab", 0.0)):
                out.append(f'    (fp_rect (start {fmt(-w / 2 - grow)} {fmt(-h / 2 - grow)}) (end {fmt(w / 2 + grow)} {fmt(h / 2 + grow)}) '
                           f'(stroke (width 0.1) (type default)) (fill none) (layer "{layer}") (tstamp {uid()}))')
        if part.package:
            # Pin-1 mark at the rotated position of ball A1.
            ax, ay = part.local_to_board(*part.package.ball_xy(0, 0))
            out.append(f'    (fp_circle (center {fmt(ax - part.x - 0.8)} {fmt(-(ay - part.y) - 0.8)}) (end {fmt(ax - part.x - 0.4)} '
                       f'{fmt(-(ay - part.y) - 0.8)}) (stroke (width 0.15) (type default)) (fill none) (layer "F.SilkS") (tstamp {uid()}))')
        for pad in part.pads:
            bx, by = part.local_to_board(pad.x, pad.y)
            lx, ly = bx - part.x, -(by - part.y)
            net = f' (net {net_id[pad.net]} "{pad.net}")' if pad.net else ""
            out.append(f'    (pad "{pad.name}" smd {pad.shape} (at {fmt(lx)} {fmt(ly)}) (size {fmt(pad.size[0])} {fmt(pad.size[1])}) '
                       f'(layers {pad.layers}){net} (tstamp {uid()}))')
        out.append("  )")
    out.append("")
    for track in design.tracks:
        for a, b in zip(track.points, track.points[1:]):
            out.append(f'  (segment (start {fmt(kx(a[0]))} {fmt(ky(a[1]))}) (end {fmt(kx(b[0]))} {fmt(ky(b[1]))}) '
                       f'(width {fmt(track.width)}) (layer "{track.layer}") (net {net_id[track.net]}) (tstamp {uid()}))')
    for via in design.vias:
        out.append(f'  (via (at {fmt(kx(via.x))} {fmt(ky(via.y))}) (size {fmt(via.size)}) (drill {fmt(via.drill)}) '
                   f'(layers "F.Cu" "B.Cu") (net {net_id[via.net]}) (tstamp {uid()}))')
    out.append("")
    for zone in design.zones:
        pts = " ".join(f"(xy {fmt(kx(x))} {fmt(ky(y))})" for x, y in zone.polygon)
        out.append(f'  (zone (net {net_id[zone.net]}) (net_name "{zone.net}") (layer "{zone.layer}") (tstamp {uid()}) '
                   f'(hatch edge 0.5) (priority {zone.priority}) (connect_pads (clearance 0.2)) (min_thickness 0.25) '
                   f'(filled_areas_thickness no) (fill yes (thermal_gap 0.3) (thermal_bridge_width 0.3)) '
                   f'(polygon (pts {pts})))')
    out.append(")")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_project(path: Path) -> None:
    project = {
        "board": {"design_settings": {"defaults": {}, "rules": {
            "min_clearance": 0.1, "min_copper_edge_clearance": 0.5, "min_hole_clearance": 0.25,
            "min_hole_to_hole": 0.25, "min_microvia_diameter": 0.2, "min_microvia_drill": 0.1,
            "min_resolved_spokes": 1, "min_silk_clearance": 0.0, "min_text_height": 0.8, "min_text_thickness": 0.08,
            "min_through_hole_diameter": 0.15, "min_track_width": 0.1, "min_via_annular_width": 0.075,
            "min_via_diameter": 0.3, "solder_mask_clearance": 0.0, "solder_mask_min_width": 0.0},
            "rule_severities": {"unconnected_items": "warning", "zones_intersect": "warning",
                                "lib_footprint_issues": "ignore", "lib_footprint_mismatch": "ignore",
                                "footprint_type_mismatch": "ignore", "missing_courtyard": "ignore",
                                "silk_overlap": "ignore", "silk_over_copper": "ignore"}},
                  "layer_presets": [], "viewports": []},
        "boards": [], "cvpcb": {"equivalence_files": []}, "libraries": {"pinned_footprint_libs": [], "pinned_symbol_libs": []},
        "meta": {"filename": f"{PROJECT}.kicad_pro", "version": 1},
        "net_settings": {"classes": [{"bus_width": 12, "clearance": 0.1, "diff_pair_gap": 0.15, "diff_pair_via_gap": 0.25,
                                      "diff_pair_width": 0.1, "line_style": 0, "microvia_diameter": 0.2,
                                      "microvia_drill": 0.1, "name": "Default", "pcb_color": "rgba(0, 0, 0, 0.000)",
                                      "schematic_color": "rgba(0, 0, 0, 0.000)", "track_width": 0.1,
                                      "via_diameter": 0.4, "via_drill": 0.2, "wire_width": 6}],
                         "meta": {"version": 3}, "net_colors": None, "netclass_assignments": None,
                         "netclass_patterns": []},
        "pcbnew": {"last_paths": {}, "page_layout_descr_file": ""},
        "schematic": {"legacy_lib_dir": "", "legacy_lib_list": []},
        "sheets": [], "text_variables": {},
    }
    path.write_text(json.dumps(project, indent=2) + "\n", encoding="utf-8")


def symbol_definition(part_class: str, parts: list[Part]) -> tuple[str, list[str]]:
    """One library symbol per part class; pins on four sides in pad order."""
    sample = parts[0]
    pins = sample.pads
    per_side = math.ceil(len(pins) / 4)
    width = max(30.0, per_side * 2.54 + 10.0) if len(pins) > 40 else 30.0
    height = max(20.0, per_side * 2.54 + 10.0)
    name = f"appliance:{part_class}"
    lines = [f'    (symbol "{name}" (pin_names (offset 1.016)) (in_bom yes) (on_board yes)',
             f'      (property "Reference" "U" (at 0 {fmt(height / 2 + 3)} 0) (effects (font (size 1.27 1.27))))',
             f'      (property "Value" "{part_class}" (at 0 {fmt(-height / 2 - 3)} 0) (effects (font (size 1.27 1.27))))',
             f'      (property "Footprint" "appliance:{sample.footprint}" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))',
             f'      (symbol "{part_class}_0_1"',
             f'        (rectangle (start {fmt(-width / 2)} {fmt(height / 2)}) (end {fmt(width / 2)} {fmt(-height / 2)}) '
             f'(stroke (width 0.254) (type default)) (fill (type background)))',
             "      )", f'      (symbol "{part_class}_1_1"']
    positions = []
    for k, pad in enumerate(pins):
        side, slot = k // per_side, k % per_side
        offset = (slot - (per_side - 1) / 2) * 2.54
        if side == 0:
            x, y, rot = -width / 2 - 2.54, offset, 0
        elif side == 1:
            x, y, rot = width / 2 + 2.54, offset, 180
        elif side == 2:
            x, y, rot = offset, height / 2 + 2.54, 270
        else:
            x, y, rot = offset, -height / 2 - 2.54, 90
        positions.append((x, y, rot))
        lines.append(f'        (pin bidirectional line (at {fmt(x)} {fmt(y)} {rot}) (length 2.54) '
                     f'(name "{pad.name}" (effects (font (size 0.8 0.8)))) (number "{pad.name}" (effects (font (size 0.8 0.8)))))')
    lines += ["      )", "    )"]
    return "\n".join(lines), [f"{fmt(x)} {fmt(y)} {rot}" for x, y, rot in positions]


def write_schematic(design: Design, path: Path) -> None:
    """One A0 sheet per part group; a global label on every pin carries the net."""
    root_uuid = uid()
    groups = {"asics": lambda p: p.part_class in ("layer_asic", "head_asic"),
              "memory": lambda p: p.part_class in ("lpddr5x", "ddr4"),
              "fpga": lambda p: p.part_class in ("fpga", "clock_gen", "sfp_cage", "pcie_edge"),
              "power": lambda p: p.part_class in ("vrm_core", "regulator", "aux_power")}
    sheet_uuids = {name: uid() for name in groups}
    root = [f"(kicad_sch (version 20230121) (generator asic_kicad_gen)", f"  (uuid {root_uuid})", '  (paper "A3")',
            "  (lib_symbols)"]
    for index, name in enumerate(groups):
        x, y = 30 + (index % 2) * 120, 30 + (index // 2) * 80
        root.append(f'  (sheet (at {x} {y}) (size 80 50) (fields_autoplaced) (stroke (width 0.15) (type solid)) '
                    f'(fill (color 0 0 0 0.0)) (uuid {sheet_uuids[name]}) (property "Sheetname" "{name}" (at {x} {y - 1} 0) '
                    f'(effects (font (size 1.27 1.27)) (justify left bottom))) (property "Sheetfile" "{name}.kicad_sch" '
                    f'(at {x} {y + 51} 0) (effects (font (size 1.27 1.27)) (justify left top))) '
                    f'(instances (project "{PROJECT}" (path "/{root_uuid}" (page "{index + 2}")))))')
    root.append(f'  (sheet_instances (path "/" (page "1")))')
    root.append(")")
    path.write_text("\n".join(root) + "\n", encoding="utf-8")
    for name, select in groups.items():
        sheet_path = f"/{root_uuid}/{sheet_uuids[name]}"
        parts = [p for p in design.parts if select(p)]
        by_class: dict[str, list[Part]] = {}
        for part in parts:
            by_class.setdefault(part.part_class if part.package else part.footprint, []).append(part)
        lines = [f"(kicad_sch (version 20230121) (generator asic_kicad_gen)", f"  (uuid {uid()})", '  (paper "A0")',
                 "  (lib_symbols"]
        pin_positions: dict[str, list[str]] = {}
        for key, group in by_class.items():
            text, positions = symbol_definition(key, group)
            lines.append(text)
            pin_positions[key] = positions
        lines.append("  )")
        cursor_x, cursor_y, row_height = 60.0, 60.0, 0.0
        for key, group in by_class.items():
            per_side = math.ceil(len(group[0].pads) / 4)
            width = (max(30.0, per_side * 2.54 + 10.0) if len(group[0].pads) > 40 else 30.0) + 60.0
            height = max(20.0, per_side * 2.54 + 10.0) + 60.0
            for part in group:
                if cursor_x + width > 1150:
                    cursor_x, cursor_y = 60.0, cursor_y + row_height
                    row_height = 0.0
                sx, sy = cursor_x + width / 2, cursor_y + height / 2
                lines.append(f'  (symbol (lib_id "appliance:{key}") (at {fmt(sx)} {fmt(sy)} 0) (unit 1) (in_bom yes) '
                             f'(on_board yes) (dnp no) (uuid {uid()})')
                lines.append(f'    (property "Reference" "{part.ref}" (at {fmt(sx)} {fmt(sy - height / 2 + 27)} 0) '
                             f'(effects (font (size 2 2))))')
                lines.append(f'    (property "Value" "{part.value or key}" (at {fmt(sx)} {fmt(sy + height / 2 - 27)} 0) '
                             f'(effects (font (size 1.5 1.5))))')
                lines.append(f'    (property "Footprint" "appliance:{part.footprint}" (at {fmt(sx)} {fmt(sy)} 0) '
                             f'(effects (font (size 1.27 1.27)) hide))')
                for pad in part.pads:
                    lines.append(f'    (pin "{pad.name}" (uuid {uid()}))')
                lines.append(f'    (instances (project "{PROJECT}" (path "{sheet_path}" (reference "{part.ref}") (unit 1))))')
                lines.append("  )")
                for pad, position in zip(part.pads, pin_positions[key]):
                    if not pad.net:
                        continue
                    px, py, rot = position.split()
                    gx, gy = sx + float(px), sy - float(py)     # schematic y is down
                    ang = {"0": 180, "180": 0, "270": 90, "90": 270}[rot]
                    lines.append(f'  (global_label "{pad.net}" (shape input) (at {fmt(gx)} {fmt(gy)} {ang}) '
                                 f'(effects (font (size 0.8 0.8)) (justify {"right" if ang == 180 else "left"})) (uuid {uid()}) '
                                 f'(property "Intersheetrefs" "${{INTERSHEET_REFS}}" (at 0 0 0) (effects (font (size 0.8 0.8)) hide)))')
                cursor_x += width
                row_height = max(row_height, height)
        lines.append(")")
        (path.parent / f"{name}.kicad_sch").write_text("\n".join(lines) + "\n", encoding="utf-8")


def floorplan_svg(design: Design) -> str:
    """A light rendering of the placement and the ring ribbons (the KiCad SVG
    export of the same board is tens of megabytes because of the zone fills)."""
    scale = 4.0
    w, h = CARD_LENGTH * scale + 40, CARD_HEIGHT * scale + 60
    fills = {"layer_asic": "#dbe8f7", "head_asic": "#f7dbdb", "lpddr5x": "#e8f3e8", "ddr4": "#e8f3e8",
             "fpga": "#fff2cc", "vrm_core": "#f3e6d0", "regulator": "#eeeeee", "clock_gen": "#eeeeee",
             "sfp_cage": "#dddddd", "aux_power": "#dddddd", "pcie_edge": "#dddddd"}
    layer_colors = {LINK_LAYERS[0]: "#1f5fbf", LINK_LAYERS[1]: "#bf5f1f", "F.Cu": "#999999"}

    def sx(x: float) -> float:
        return 20 + x * scale

    def sy(y: float) -> float:
        return 20 + (CARD_HEIGHT - y) * scale

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" font-family="sans-serif" font-size="11">',
             f'<rect width="{w:.0f}" height="{h:.0f}" fill="#fafafa"/>',
             f'<rect x="{sx(0)}" y="{sy(CARD_HEIGHT)}" width="{CARD_LENGTH * scale}" height="{CARD_HEIGHT * scale}" '
             f'fill="#f4f7f4" stroke="#333" stroke-width="1.5"/>',
             f'<rect x="{sx(0)}" y="{sy(FINGER_ZONE_HEIGHT)}" width="{90 * scale}" height="{FINGER_ZONE_HEIGHT * scale}" '
             f'fill="none" stroke="#999" stroke-dasharray="4,3"/>']
    for zone in design.zones:
        if zone.layer == CORE_RAIL_LAYER:
            xs, ys = [p[0] for p in zone.polygon], [p[1] for p in zone.polygon]
            parts.append(f'<rect x="{sx(min(xs))}" y="{sy(max(ys))}" width="{(max(xs) - min(xs)) * scale}" '
                         f'height="{(max(ys) - min(ys)) * scale}" fill="#f9e9e9" stroke="none"/>')
    for part in design.parts:
        if part.body_w == 0:
            continue
        bw, bh = (part.body_h, part.body_w) if part.rotation in (90, 270) else (part.body_w, part.body_h)
        parts.append(f'<rect x="{sx(part.x - bw / 2)}" y="{sy(part.y + bh / 2)}" width="{bw * scale}" height="{bh * scale}" '
                     f'fill="{fills.get(part.part_class, "#eee")}" stroke="#333" stroke-width="1"/>')
        size = 11 if bw > 15 else 7
        parts.append(f'<text x="{sx(part.x)}" y="{sy(part.y) + size / 3}" text-anchor="middle" font-size="{size}">{part.ref}</text>')
    for track in design.tracks:
        if track.layer == "F.Cu":
            continue
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in track.points)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{layer_colors.get(track.layer, "#000")}" stroke-width="0.6"/>')
    parts.append(f'<text x="{sx(2)}" y="{h - 28}" fill="#333">Generated from hw/board.yaml by hw/kicad_gen.py. '
                 f'Blue: ring ribbons on {LINK_LAYERS[0]}; orange: second half of bent hops on {LINK_LAYERS[1]}; '
                 f'pink: core-rail islands on {CORE_RAIL_LAYER}. Memory, PCIe and management nets are unrouted.</text>')
    parts.append(f'<text x="{sx(2)}" y="{h - 12}" fill="#333">' + "; ".join(
        f"{hop.replace('U_', '')} {hi:.0f} mm" for hop, (_, hi) in design.hop_lengths.items()) + "</text>")
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def generate(board: Board, output: Path) -> Design:
    output.mkdir(parents=True, exist_ok=True)
    design = build_design(board)
    write_project(output / f"{PROJECT}.kicad_pro")
    write_pcb(design, output / f"{PROJECT}.kicad_pcb")
    write_schematic(design, output / f"{PROJECT}.kicad_sch")
    (output / "report.md").write_text(report_markdown(design), encoding="utf-8")
    (output / "floorplan.svg").write_text(floorplan_svg(design), encoding="utf-8")
    return design


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "board.yaml")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--check", action="store_true", help="generate into a temporary directory and print the report")
    args = parser.parse_args()
    board = Board.load(args.source)
    if args.check:
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            design = generate(board, Path(directory))
            print(report_markdown(design))
        return
    design = generate(board, args.output)
    print(f"wrote {args.output}/{PROJECT}.kicad_pcb: {len(design.parts)} footprints, {len(design.nets())} nets, "
          f"{len(design.tracks)} tracks, {len(design.vias)} vias")
    for hop, (lo, hi) in design.hop_lengths.items():
        print(f"  {hop}: {lo:.1f} to {hi:.1f} mm{'' if hi <= MAX_LINK_MM else '  OVER LIMIT'}")


if __name__ == "__main__":
    main()
