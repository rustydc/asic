"""Generate a KiCad 7 project for the appliance board from ``hw/board.yaml``.

    python -m hw.kicad_gen            # writes hw/kicad/appliance.kicad_{pro,pcb,sch}, report.md, floorplan.svg
    python -m hw.kicad_gen --check    # generate into a temporary directory and print the report

What it produces is a floorplan-level board, not a finished layout:

* the 1U board outline with its keep-outs (PSU bay, rear I/O strip, fan
  row), the host PCIe cable connector, the BMC and its Ethernet, the PSU and
  fan connectors and the SFP+ cage;
* every part of ``board.yaml`` placed: the activation ring is a regular
  polygon with one ring node per vertex (the FPGA, the eight layer ASICs and
  the two head ASICs, eleven in all), each chip rotated tangentially so its
  link-in edge faces the previous chip and its link-out edge the next, with
  its two LPDDR5X devices on its outer edge and its core regulator on its
  inner edge, rotated with it; the shared regulators and the clock generator
  sit in the middle of the polygon; the ASIC footprint is built from the
  ball map that ``hw/pinout.py`` derives from the power model;
* every net at the signal level, including the memory channels, PCIe and the
  management buses, so the schematic and the PCB agree;
* a twelve-layer stackup, ground and power zones, via-in-pad on the rail
  balls, and the activation ring fully routed as ribbons with escape vias,
  checked against the link length limit;
* a report with link lengths, bend angles, escape density and core-rail
  current density.

The polygon replaces the earlier two-row snake: every hop is the same short
ribbon with two gentle bends, the two long hops of the snake (row change and
closing hop) are gone, and no chip sits directly downstream of more than one
other in the front-to-back airflow.  All eleven ring nodes are on the
polygon because every chip takes the ring in on one edge and out on the
opposite edge, so the loop cannot dip into the middle and come back without
a hairpin.  The memory, PCIe and management nets are present but unrouted:
they need length matching and signal-integrity work that an autorouter would
only imitate.  Decoupling capacitors and the small housekeeping parts are not
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

from hw import pinout
from hw.board import Board, Endpoint

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "kicad"
PROJECT = "appliance"
KICAD_ORIGIN = (20.0, 20.0)     # page offset so the outline sits inside the sheet

LINK_LAYER = "In2.Cu"         # every ring hop is one ribbon on this layer
TRACK_WIDTH = 0.1
VIA_SIZE, VIA_DRILL = 0.4, 0.2
SMALL_VIA_SIZE = 0.3          # via-in-pad on the 0.35 mm memory balls
ESCAPE_OUT = 2.7              # outer-column escape via distance beyond the ball, mm
MAX_LINK_MM = 80.0
MAX_BEND_DEG = 60.0           # a ribbon bend sharper than this is a placement error

# Twelve-layer stackup: signal / ground / link signals / 12 V and I/O rails /
# ground / memory signals / memory signals / ground / core rails / signals / ground / signal.
COPPER_LAYERS = ["F.Cu", "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu", "In5.Cu", "In6.Cu", "In7.Cu",
                 "In8.Cu", "In9.Cu", "In10.Cu", "B.Cu"]
LAYER_ROLES = {"F.Cu": "signal", "In1.Cu": "GND", "In2.Cu": "link signals", "In3.Cu": "12 V and I/O rails",
               "In4.Cu": "GND", "In5.Cu": "memory signals", "In6.Cu": "memory signals", "In7.Cu": "GND",
               "In8.Cu": "core rails", "In9.Cu": "signal", "In10.Cu": "GND", "B.Cu": "signal"}
GND_LAYERS = ["In1.Cu", "In4.Cu", "In7.Cu", "In10.Cu"]
TOP_RAIL_LAYER, CORE_RAIL_LAYER = "In3.Cu", "In8.Cu"

_JEDEC = "ABCDEFGHJKLMNPRTUVWY"
ROW_LETTERS = list(_JEDEC) + [p + c for p in _JEDEC for c in _JEDEC]   # A..Y, AA..AY, BA.., 420 rows


def uid() -> str:
    return str(uuid.uuid4())


def fmt(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


Point = tuple[float, float]


# --------------------------------------------------------------------------
# Form factor: the physical board (y up, the chassis rear at y = 0)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FormFactor:
    key: str
    description: str
    width: float                      # physical x, mm
    depth: float                      # physical y, mm
    ring_centre: Point                # centre of the ring polygon
    keepouts: tuple[tuple[str, tuple[float, float, float, float]], ...] = ()


FORM_FACTORS = {
    "1u": FormFactor("1u", "1U rack chassis, single board, host-attached", 420.0, 360.0, (140.0, 190.0),
                     (("PSU bay (two CRPS)", (273.0, 0.0, 420.0, 190.0)),
                      ("rear I/O", (0.0, 0.0, 273.0, 12.0)),
                      ("front fans", (0.0, 352.0, 420.0, 360.0)))),
}


def select_form_factor(board: Board) -> FormFactor:
    return FORM_FACTORS[board.data["board"].get("form_factor_key", "1u")]


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

    def ball_xy(self, i: int, j: int) -> Point:
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
    rotation: float             # degrees counter-clockwise; local +x is the link-out direction of a ring chip
    body_w: float
    body_h: float
    pads: list[Pad] = field(default_factory=list)
    value: str = ""
    package: Package | None = None

    def local_to_board(self, x: float, y: float) -> Point:
        c, s = math.cos(math.radians(self.rotation)), math.sin(math.radians(self.rotation))
        return self.x + x * c - y * s, self.y + x * s + y * c

    def direction(self, local_angle_deg: float = 0.0) -> Point:
        """Unit vector of a local direction (0 = local +x) on the board."""
        a = math.radians(self.rotation + local_angle_deg)
        return math.cos(a), math.sin(a)

    def pad(self, name: str) -> Pad:
        for pad in self.pads:
            if pad.name == name:
                return pad
        raise KeyError(f"{self.ref} has no pad {name}")

    def corners(self, grow: float = 0.0) -> list[Point]:
        """Body outline on the board, counter-clockwise from the local south-west corner."""
        w, h = self.body_w / 2 + grow, self.body_h / 2 + grow
        return [self.local_to_board(x, y) for x, y in ((-w, -h), (w, -h), (w, h), (-w, h))]

    def extent(self) -> tuple[float, float]:
        """Axis-aligned size of the rotated body."""
        pts = self.corners()
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return max(xs) - min(xs), max(ys) - min(ys)


PACKAGES = {
    "fpga": Package("FFVB676_26x26_P1.0", 26, 26, 1.0, 27.0, 27.0, 0.5),
    "lpddr5x": Package("FBGA315_15x21_P0.8", 15, 21, 0.8, 12.0, 16.8, 0.35),
    "ddr4": Package("FBGA96_8x12_P0.8", 8, 12, 0.8, 7.5, 10.6, 0.35),
}


def asic_package(pin: pinout.Pinout) -> Package:
    """The ASIC package is whatever hw/pinout.py selected from the power model."""
    p = pin.package
    return Package(p.name, p.cols, p.rows, p.pitch, p.body, p.body, p.ball)


def expand_signals(signals: dict[str, int]) -> list[str]:
    return pinout.expand_signals(signals)


# --------------------------------------------------------------------------
# Net naming
# --------------------------------------------------------------------------

def short_ref(ref: str) -> str:
    return ref.replace("U_", "")


def ring_net(hop: int, signal: str) -> str:
    return f"LINK{hop}_{signal}"


def memory_net(asic_ref: str, channel: str, signal: str) -> str:
    return f"{short_ref(asic_ref)}_{channel.upper()}_{signal}"


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


# --------------------------------------------------------------------------
# Ball maps
# --------------------------------------------------------------------------

def link_signals(board: Board) -> list[str]:
    return expand_signals(board.kinds["link"]["signals"])


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


def fpga_ball_map(board: Board) -> list[Pad]:
    """FPGA: link_out on the east edge, link_in on the west edge (both assigned
    by the ribbon router), PCIe on the south edge towards the host connector,
    DDR4 x64 on the north rows, management and small interfaces on the south-west."""
    pkg = PACKAGES["fpga"]
    assigned: dict[tuple[int, int], tuple[str | None, tuple[str, int] | None]] = {}
    for k in range(36):
        assigned[(4 + k // 2, 25 if k % 2 == 0 else 24)] = (None, ("E", k % 2))
        assigned[(4 + k // 2, 0 if k % 2 == 0 else 1)] = (None, ("W", k % 2))
    pcie = expand_signals(board.kinds["pcie_gen4_x8"]["signals"])
    south_edge = iter((i, j) for i in (25, 24) for j in range(2, 24))
    for signal in pcie:
        assigned[next(south_edge)] = (f"PCIE_{signal}", None)
    ddr = expand_signals(board.kinds["ddr4_x64"]["signals"])
    north = iter([(i, j) for i in range(4) for j in range(26)] + [(4, j) for j in range(2, 24)])   # clear of the link columns
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


def memory_ball_map(package: Package, signal_nets: list[str], rails: list[str]) -> list[Pad]:
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


def host_cable_nets(board: Board) -> list[str]:
    """SlimSAS 8i carries the eight lanes plus the sideband."""
    pcie = [f"PCIE_{s}" for s in expand_signals(board.kinds["pcie_gen4_x8"]["signals"])]
    return pcie + ["GND"] * (len(pcie) // 2)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def polygons_overlap(a: list[Point], b: list[Point], margin: float = 0.0) -> bool:
    """Separating-axis test for two convex polygons; ``margin`` is the minimum
    gap that still counts as clear."""
    for poly in (a, b):
        for i in range(len(poly)):
            p, q = poly[i], poly[(i + 1) % len(poly)]
            ax, ay = -(q[1] - p[1]), q[0] - p[0]
            length = math.hypot(ax, ay)
            if length == 0:
                continue
            ax, ay = ax / length, ay / length
            pa = [v[0] * ax + v[1] * ay for v in a]
            pb = [v[0] * ax + v[1] * ay for v in b]
            if max(pa) + margin <= min(pb) + 1e-9 or max(pb) + margin <= min(pa) + 1e-9:
                return False
    return True


def point_in_polygon(x: float, y: float, polygon: list[Point]) -> bool:
    inside = False
    n = len(polygon)
    for i in range(n):
        (x0, y0), (x1, y1) = polygon[i], polygon[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < cross:
                inside = not inside
    return inside


def rect(x0: float, y0: float, x1: float, y1: float) -> list[Point]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def local_rect(part: Part, x0: float, y0: float, x1: float, y1: float) -> list[Point]:
    """A rectangle in a part's local frame, rotated with it onto the board."""
    return [part.local_to_board(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]


def regular_polygon(centre: Point, radius: float, sides: int) -> list[Point]:
    return [(centre[0] + radius * math.cos(2 * math.pi * k / sides), centre[1] + radius * math.sin(2 * math.pi * k / sides))
            for k in range(sides)]


# --------------------------------------------------------------------------
# Design assembly
# --------------------------------------------------------------------------

@dataclass
class Track:
    layer: str
    net: str
    points: list[Point]
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
    polygon: list[Point]
    priority: int = 0


@dataclass
class RingLayout:
    """The ring as a regular polygon: one node per ring chip, in ring order."""
    centre: Point
    radius: float
    side: float                                    # centre-to-centre distance of neighbours
    nodes: dict[str, tuple[float, float, float]]   # ref -> (x, y, rotation)
    angles: dict[str, float]                       # ref -> polar angle of the node, degrees


RING_GAP = 17.0        # neighbour spacing beyond the package body: escapes, ribbon runs and clearance
                       # (15 mm still passes DRC; below that the ports are too close for a ribbon)
FPGA_ANGLE = -90.0     # the FPGA is the ring node nearest the chassis rear


def layout_for(board: Board, package: Package, ff: FormFactor) -> RingLayout:
    refs = [source.component for source, _ in board.ring()]        # ring order, starting at the FPGA
    n = len(refs)
    side = package.body_w + float(board.data["board"].get("ring_gap_mm", RING_GAP))
    radius = side / (2 * math.sin(math.pi / n))
    cx, cy = ff.ring_centre
    nodes, angles = {}, {}
    for k, ref in enumerate(refs):
        # Clockwise around the polygon so each chip's local north (its memory
        # edge) faces outward and its local +x (link out) points at the next chip.
        angle = FPGA_ANGLE - k * 360.0 / n
        a = math.radians(angle)
        nodes[ref] = (cx + radius * math.cos(a), cy + radius * math.sin(a), (angle - 90.0) % 360.0)
        angles[ref] = angle % 360.0
    return RingLayout((cx, cy), radius, side, nodes, angles)


VRM_W, VRM_H = 26.0, 5.5
REG_W, REG_H = 20.0, 6.5


@dataclass
class Design:
    board: Board
    form_factor: FormFactor
    pinout: pinout.Pinout | None = None
    layout: RingLayout | None = None
    parts: list[Part] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    hop_lengths: dict[str, tuple[float, float]] = field(default_factory=dict)   # hop -> (min, max) mm
    hop_bends: dict[str, float] = field(default_factory=dict)                   # hop -> sharpest bend, degrees
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

    def physical_xy(self, part: Part) -> Point:
        return part.x, part.y

    def physical_extent(self, part: Part) -> tuple[float, float]:
        return part.extent()


def ring_hops(board: Board) -> list[tuple[str, str]]:
    return [(a.component, b.component) for a, b in board.ring()]


def place_relative(anchor: Part, dx: float, dy: float, extra_rotation: float = 0.0) -> tuple[float, float, float]:
    """Position and rotation of a part placed at a local offset of ``anchor``, rotating with it."""
    x, y = anchor.local_to_board(dx, dy)
    return x, y, (anchor.rotation + extra_rotation) % 360.0


def build_design(board: Board) -> Design:
    pin = pinout.derive(board)
    asic_pkg = asic_package(pin)
    ff = select_form_factor(board)
    lay = layout_for(board, asic_pkg, ff)
    design = Design(board, ff, pinout=pin, layout=lay)
    hops = ring_hops(board)
    hop_in = {sink: h for h, (_, sink) in enumerate(hops)}
    hop_out = {source: h for h, (source, _) in enumerate(hops)}
    memory_of: dict[str, dict[str, str]] = {}
    for a, b in board.nets["memory"]["channels"]:
        asic, device = Endpoint.parse(a), Endpoint.parse(b)
        memory_of.setdefault(asic.component, {})[asic.interface] = device.component

    # ASICs on their polygon vertices, each with its core regulator on its inner (south) edge.
    for ref in board.instances("layer_asic") + board.instances("head_asic"):
        x, y, rotation = lay.nodes[ref]
        pads = asic_ball_map(board, pin, board.class_of(ref), ref, hop_in[ref], hop_out[ref], memory_of.get(ref, {}))
        asic = Part(ref, board.class_of(ref), asic_pkg.name, x, y, rotation, asic_pkg.body_w, asic_pkg.body_h, pads,
                    value=board.class_of(ref), package=asic_pkg)
        design.parts.append(asic)
        core = f"VDD_CORE_{short_ref(ref)}"
        nets = ["+12V"] * 4 + [core] * 6 + ["GND"] * 6 + ["PMB_SCL", "PMB_SDA", f"VRM_EN_{short_ref(ref)}"]
        vx, vy, vrot = place_relative(asic, 0.0, -(asic_pkg.body_h / 2 + 2.5 + VRM_H / 2))
        design.parts.append(Part(f"VRM_{short_ref(ref)}", "vrm_core", f"VRM_MODULE_{VRM_W:.0f}x{VRM_H:.0f}", vx, vy, vrot,
                                 VRM_W, VRM_H, block_pads(VRM_W, VRM_H, nets, pitch=2.3, pad_size=(1.2, 1.2)),
                                 value=f"core VRM {core}"))

    # LPDDR5X on the outer (north) edge of each layer ASIC, rotated with it
    # (none when the memory is HBM in the package).
    lpddr_signals = pinout.memory_signals(board)
    for asic_ref, channels in memory_of.items():
        asic = design.part(asic_ref)
        pkg = PACKAGES["lpddr5x"]
        for index, (channel, device) in enumerate(sorted(channels.items())):
            dx = -9.5 if index == 0 else 9.5
            x, y, rotation = place_relative(asic, dx, asic.body_h / 2 + 2.0 + pkg.body_w / 2, 90.0)
            nets = [memory_net(asic_ref, channel[-3:], s) for s in lpddr_signals]
            pads = memory_ball_map(pkg, nets, [r for r in board.classes[board.class_of(device)]["rails"]])
            design.parts.append(Part(device, "lpddr5x", pkg.name, x, y, rotation, pkg.body_w, pkg.body_h, pads,
                                     value="LPDDR5X x32", package=pkg))

    # FPGA on its vertex; DDR4 and its regulators on its outer (north) edge, its core regulator inside.
    fpga_ref = board.instances("fpga")[0]
    pkg = PACKAGES["fpga"]
    x, y, rotation = lay.nodes[fpga_ref]
    fpga = Part(fpga_ref, "fpga", pkg.name, x, y, rotation, pkg.body_w, pkg.body_h, fpga_ball_map(board),
                value="FPGA", package=pkg)
    design.parts.append(fpga)
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
        pads = memory_ball_map(pkg, nets, board.classes["ddr4"]["rails"])
        x, y, rotation = place_relative(fpga, -12.75 + 8.5 * n, fpga.body_h / 2 + 2.0 + pkg.body_h / 2)
        design.parts.append(Part(device, "ddr4", pkg.name, x, y, rotation, pkg.body_w, pkg.body_h, pads,
                                 value="DDR4 x16", package=pkg))
    # Regulators follow the power tree: the FPGA's own rails on one block on its
    # inner edge, the DDR4 rails and 3.3 V beside its DDR4 row, every other
    # shared rail on its own block in the middle of the polygon beside the clock
    # generator, equidistant from every chip.
    tree = board.data["power_tree"]["rails"]
    fpga_rails = [r for r, spec in tree.items() if spec.get("per") == ["fpga"]]
    ddr_rails = [r for r in board.classes["ddr4"]["rails"]] if "ddr4" in board.classes else []
    centre_rails = [r for r, spec in tree.items() if spec.get("shared") and r not in ddr_rails and r != "VDD_3V3"]
    blocks = [("REG_FPGA", fpga_rails, place_relative(fpga, 0.0, -(fpga.body_h / 2 + 2.5 + REG_H / 2))),
              ("REG_DDR", ddr_rails, place_relative(fpga, -11.0, fpga.body_h / 2 + 2.0 + PACKAGES["ddr4"].body_h + 2.0 + REG_H / 2)),
              ("REG_3V3", ["VDD_3V3"], place_relative(fpga, 11.0, fpga.body_h / 2 + 2.0 + PACKAGES["ddr4"].body_h + 2.0 + REG_H / 2))]
    cx, cy = lay.centre
    clk_nets = ["VDD_IO_1V8", "GND"] + [f"REFCLK_{short_ref(r)}_{p}" for r in board.nets["refclk"]["sinks"] for p in "PN"]
    design.parts.append(Part("U_CLK", "clock_gen", "QFN64_9x9", cx, cy, 0.0, 9.0, 9.0,
                             block_pads(9.0, 9.0, clk_nets[:24], pitch=0.7, pad_size=(0.3, 0.9)), value="Si5345"))
    for k, rail in enumerate(centre_rails):
        row, col = k // 2, k % 2
        y = cy + (10.0 + 8.0 * (row // 2)) * (1 if row % 2 == 0 else -1)
        blocks.append((f"REG_{rail.split('_', 1)[-1]}" if rail.count("_") else f"REG_{rail}", [rail], (cx + (12.0 if col else -12.0), y, 0.0)))
    for ref, rails, (x, y, rotation) in blocks:
        nets = ["+12V"] * 3 + [rails[0]] * 3 + rails[1:] + ["GND"] * 3
        design.parts.append(Part(ref, "regulator", f"REG_MODULE_{REG_W:.0f}x{REG_H:.0f}", x, y, rotation, REG_W, REG_H,
                                 block_pads(REG_W, REG_H, nets, pitch=1.8, pad_size=(1.2, 1.2)),
                                 value=f"{', '.join(rails)} regulator"))

    add_chassis_parts(design)

    # Daisy chains: JTAG TDO -> next TDI.
    chain = board.nets["jtag"]["chain"]
    for a, b in zip(chain, chain[1:]):
        for pad in design.part(b).pads:
            if pad.net == f"JTAG_TDI_{short_ref(b)}":
                pad.net = f"JTAG_TDO_{short_ref(a)}"

    route_ring(design)
    add_zones(design)
    add_power_vias(design)
    check_fit(design)
    return design


def add_chassis_parts(design: Design) -> None:
    """Connectors and housekeeping that belong to the enclosure: rear I/O
    along y = 0, PSU bay at the rear right, fans along the front."""
    board = design.board
    fpga = design.part(board.instances("fpga")[0])
    sfp_nets = [f"SFP_{s}" for s in expand_signals(board.kinds["sfp_plus"]["signals"])] + ["VDD_3V3"] * 2 + ["GND"] * 4
    # Beside the FPGA on the rear edge, clear of its DDR4 row.
    design.parts.append(Part("J_HOST", "host_cable", "SLIMSAS_8I", fpga.x - 34.0, 6.0, 0.0, 26.0, 9.0,
                             block_pads(26.0, 9.0, host_cable_nets(board), pitch=1.0, pad_size=(0.6, 1.5)),
                             value="SlimSAS 8i host cable"))
    design.parts.append(Part("J_SFP", "sfp_cage", "SFP_CAGE", 60.0, 27.0, 0.0, 14.0, 50.0,
                             block_pads(50.0, 14.0, sfp_nets, pitch=2.0), value="SFP+ cage"))
    eth = [f"BMC_ETH{k}_{p}" for k in range(4) for p in "PN"]
    design.parts.append(Part("J_BMC_NET", "rj45", "RJ45_MAGJACK", 30.0, 13.0, 0.0, 16.0, 21.0,
                             block_pads(16.0, 21.0, eth + ["VDD_3V3", "GND", "GND", "GND"], pitch=1.6),
                             value="BMC Ethernet"))
    fans = [f"FAN_{k}_{s}" for k in range(6) for s in ("PWM", "TACH")]
    bmc_nets = eth + fans + ["UART_TX", "UART_RX", "PMB_SCL", "PMB_SDA", "PSU_ON", "VDD_3V3", "VDD_3V3", "GND", "GND"]
    design.parts.append(Part("U_BMC", "bmc", "BGA_BMC_21x21", 30.0, 60.0, 0.0, 21.0, 21.0,
                             block_pads(21.0, 21.0, bmc_nets, pitch=1.2, pad_size=(0.6, 1.2)), value="BMC SoC"))
    psu_nets = ["+12V"] * 6 + ["GND"] * 6 + ["PMB_SCL", "PMB_SDA", "PSU_ON", "GND"]
    for k, x in enumerate((300.0, 375.0)):
        design.parts.append(Part(f"J_PSU{k}", "psu_input", "CRPS_BLADES", x, 180.0, 0.0, 40.0, 10.0,
                                 block_pads(40.0, 10.0, psu_nets, pitch=2.8, pad_size=(1.6, 2.0)),
                                 value="CRPS 12 V output"))
    for k in range(6):
        nets = ["+12V", "GND", f"FAN_{k}_PWM", f"FAN_{k}_TACH"]
        design.parts.append(Part(f"J_FAN{k}", "fan_header", "FAN_4PIN", 35.0 + 70.0 * k, 348.0, 0.0, 10.0, 6.0,
                                 block_pads(10.0, 6.0, nets, pitch=2.54, pad_size=(1.2, 1.5)), value="fan header"))


KEEPOUT_RESIDENTS = ("psu_input", "fan_header", "host_cable", "rj45", "sfp_cage")   # live in their keep-outs by design
BODY_GAP = 0.5      # minimum gap between any two bodies (matches the courtyard growth)


def check_fit(design: Design) -> None:
    """Every part inside the board, outside the keep-outs, and clear of every other part."""
    ff = design.form_factor
    boxed = [p for p in design.parts if p.body_w > 0]
    for part in boxed:
        for x, y in part.corners():
            if x < 0 or x > ff.width or y < 0 or y > ff.depth:
                raise ValueError(f"{part.ref} does not fit the {ff.description} ({part.x:.1f}, {part.y:.1f})")
        if part.part_class in KEEPOUT_RESIDENTS:
            continue
        for name, (kx0, ky0, kx1, ky1) in ff.keepouts:
            if polygons_overlap(part.corners(), rect(kx0, ky0, kx1, ky1)):
                raise ValueError(f"{part.ref} sits in the {name} keep-out")
    for i, a in enumerate(boxed):
        for b in boxed[i + 1:]:
            if polygons_overlap(a.corners(), b.corners(), BODY_GAP):
                raise ValueError(f"{a.ref} overlaps {b.ref}")


# --------------------------------------------------------------------------
# Via-in-pad on the rails
# --------------------------------------------------------------------------

def is_rail(net: str | None, board: Board) -> bool:
    if not net:
        return False
    return net == "GND" or net == "+12V" or net.startswith("VDD_CORE_") or net in board.data["power_tree"]["rails"]


def add_power_vias(design: Design) -> None:
    """Via-in-pad on every ground and rail pad (as an FCBGA on an HDI board would
    have) so the planes connect to the packages, regulators and connectors."""
    for part in design.parts:
        for pad in part.pads:
            if not is_rail(pad.net, design.board):
                continue
            x, y = part.local_to_board(pad.x, pad.y)
            # Only where a zone of that net will actually pick the via up.
            if any(zone.net == pad.net and point_in_polygon(x, y, zone.polygon) for zone in design.zones):
                size = VIA_SIZE if min(pad.size) >= VIA_SIZE else SMALL_VIA_SIZE
                design.vias.append(Via(x, y, pad.net, size=size, drill=size / 2))


# --------------------------------------------------------------------------
# Ring routing: escape vias plus ribbons of 36 lanes with gentle bends
# --------------------------------------------------------------------------

PATH_RUN = 5.0           # a ribbon leaves and enters a port straight for this long before it may bend
PATH_MARGIN = 0.6        # kept for the escape reach used by the report


def escape_reach(package: Package) -> float:
    """Distance from a package centre to a point just past the outer escape vias, where a ribbon path starts or ends."""
    return (package.cols - 1) / 2 * package.pitch + ESCAPE_OUT + PATH_MARGIN


def escape_via_local(pad: Pad, pitch: float) -> Point:
    """Escape via of a link ball: outer column straight out past the package
    edge, inner column a dogbone half a pitch diagonally, so the lanes of the
    two columns interleave at half the ball pitch.  The dogbone shift is the
    same local direction on both edges, so lane k leaves on the east at the
    offset it arrives at on the west of the next chip."""
    edge, depth = pad.escape
    dog = pitch / 2
    if edge == "E":
        return (pad.x + ESCAPE_OUT, pad.y) if depth == 0 else (pad.x + dog, pad.y + dog)
    if edge == "W":
        return (pad.x - ESCAPE_OUT, pad.y) if depth == 0 else (pad.x - dog, pad.y + dog)
    raise ValueError(f"link balls escape east or west, not {edge}")


def escape_via(part: Part, pad: Pad) -> Point:
    return part.local_to_board(*escape_via_local(pad, part.package.pitch))


def escape_pads(part: Part, edge_local: str) -> list[Pad]:
    return [pad for pad in part.pads if pad.escape and pad.escape[0] == edge_local]


def lane_centre(part: Part, edge_local: str) -> Point:
    """Board centre of a link port's escape vias; a ribbon path starts or ends here."""
    vias = [escape_via(part, pad) for pad in escape_pads(part, edge_local)]
    return sum(v[0] for v in vias) / len(vias), sum(v[1] for v in vias) / len(vias)


def unit(a: Point, b: Point) -> Point:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    return dx / length, dy / length


def turn_deg(ua: Point, ub: Point) -> float:
    """Unsigned angle between two unit vectors, degrees."""
    dot = max(-1.0, min(1.0, ua[0] * ub[0] + ua[1] * ub[1]))
    return math.degrees(math.acos(dot))


def corner_at(b0: Point, ua: Point, ub: Point, da: float, db: float) -> Point:
    """Corner of two offset lines: lateral ``da`` on the segment with direction
    ``ua`` and ``db`` on the one with ``ub``, the centrelines meeting at ``b0``."""
    na, nb = (-ua[1], ua[0]), (-ub[1], ub[0])
    pa = (b0[0] + da * na[0], b0[1] + da * na[1])
    pb = (b0[0] + db * nb[0], b0[1] + db * nb[1])
    det = ub[0] * ua[1] - ua[0] * ub[1]
    if abs(det) < 1e-9:                       # collinear segments: the offset line is the same
        return pa
    rx, ry = pb[0] - pa[0], pb[1] - pa[1]
    t = (rx * (-ub[1]) - (-ub[0]) * ry) / det
    return pa[0] + t * ua[0], pa[1] + t * ua[1]


def ribbon(path: list[Point], starts: dict[str, Point], end_vias: list[tuple[str | None, Point]],
           end_scale: float = 1.0) -> tuple[dict[str, list[Point]], dict[str, str]]:
    """Route each start via along the polyline ``path`` at its own lateral
    offset and finish on the end via lying on the same lane.  ``end_scale`` is
    the ratio of the end part's lane pitch to the start part's, for hops
    between packages of different ball pitch; the lanes converge over the
    last segment.  Returns the polylines per net and the mapping end-via-pad -> net.
    """
    segments = list(zip(path, path[1:]))
    tracks: dict[str, list[Point]] = {}
    assignment: dict[str, str] = {}
    used: set[int] = set()
    u0 = unit(*segments[0])
    n0 = (-u0[1], u0[0])                              # left normal
    ul = unit(*segments[-1])
    nl = (-ul[1], ul[0])
    for net, start in starts.items():
        d = (start[0] - path[0][0]) * n0[0] + (start[1] - path[0][1]) * n0[1]
        d_end = d * end_scale
        points = [start]
        for (a0, b0), (a1, b1) in zip(segments, segments[1:]):
            points.append(corner_at(b0, unit(a0, b0), unit(a1, b1), d, d))
        if end_scale != 1.0:
            # Converge over the last segment: leave its corner at d, arrive at d_end.
            if len(segments) == 1:
                points.append((path[0][0] + d * n0[0], path[0][1] + d * n0[1]))
            points.append((path[-1][0] + d_end * nl[0], path[-1][1] + d_end * nl[1]))
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


def hop_path(source: Part, sink: Part) -> tuple[list[Point], float]:
    """Centreline of a hop: straight out of the source's link-out port, straight
    into the sink's link-in port, one segment between.  Returns the path and
    the sharpest bend in degrees."""
    u_out, u_in = source.direction(0.0), sink.direction(0.0)
    # The path starts and ends just past the outer escape vias, on the lane
    # centreline, so lanes converging to another pitch are settled before
    # they pass the vias.
    r0, r1 = escape_reach(source.package), escape_reach(sink.package)
    lc0, lc1 = lane_centre(source, "E"), lane_centre(sink, "W")
    n_out, n_in = (-u_out[1], u_out[0]), (-u_in[1], u_in[0])
    d0 = (lc0[0] - source.x) * n_out[0] + (lc0[1] - source.y) * n_out[1]      # lateral offset of the port centre
    d1 = (lc1[0] - sink.x) * n_in[0] + (lc1[1] - sink.y) * n_in[1]
    p0 = (source.x + r0 * u_out[0] + d0 * n_out[0], source.y + r0 * u_out[1] + d0 * n_out[1])
    p1 = (sink.x - r1 * u_in[0] + d1 * n_in[0], sink.y - r1 * u_in[1] + d1 * n_in[1])
    c0 = (p0[0] + PATH_RUN * u_out[0], p0[1] + PATH_RUN * u_out[1])
    c1 = (p1[0] - PATH_RUN * u_in[0], p1[1] - PATH_RUN * u_in[1])
    if math.dist(c0, c1) < 2.0:
        raise ValueError(f"{source.ref} -> {sink.ref}: ports too close for a ribbon ({math.dist(p0, p1):.1f} mm apart)")
    mid = unit(c0, c1)
    bends = (turn_deg(u_out, mid), turn_deg(mid, u_in))
    if max(bends) > MAX_BEND_DEG:
        raise ValueError(f"{source.ref} -> {sink.ref}: ribbon would bend {max(bends):.0f} degrees; the chips do not face each other")
    path = [p0]
    if bends[0] > 0.5:
        path.append(c0)
    if bends[1] > 0.5:
        path.append(c1)
    path.append(p1)
    return path, max(bends)


def route_ring(design: Design) -> None:
    board = design.board
    for hop, (src_ref, dst_ref) in enumerate(ring_hops(board)):
        source, sink = design.part(src_ref), design.part(dst_ref)
        path, bend = hop_path(source, sink)
        src_pads = escape_pads(source, "E")
        dst_pads = escape_pads(sink, "W")
        # Escape stubs and vias on both ends.
        for part, pads in ((source, src_pads), (sink, dst_pads)):
            for pad in pads:
                vx, vy = escape_via(part, pad)
                px, py = part.local_to_board(pad.x, pad.y)
                net = pad.net or f"__{part.ref}_{pad.name}"
                design.vias.append(Via(vx, vy, net))
                design.tracks.append(Track("F.Cu", net, [(px, py), (vx, vy)]))
        # The FPGA's link pins get their nets from the ribbon, so a hop into
        # or out of an unassigned port is routed from the assigned end.
        if not all(pad.net for pad in src_pads):
            path = list(reversed(path))
            src_pads, dst_pads = dst_pads, src_pads
            source, sink = sink, source
        starts = {pad.net: escape_via(source, pad) for pad in src_pads}
        ends = [(pad.name, escape_via(sink, pad)) for pad in dst_pads]
        tracks, assignment = ribbon(path, starts, ends, sink.package.pitch / source.package.pitch)
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
        lengths = []
        for net, points in tracks.items():
            track = Track(LINK_LAYER, net, points)
            design.tracks.append(track)
            lengths.append(track.length())
        design.hop_lengths[f"{src_ref} -> {dst_ref}"] = (min(lengths), max(lengths))
        design.hop_bends[f"{src_ref} -> {dst_ref}"] = bend


# --------------------------------------------------------------------------
# Zones
# --------------------------------------------------------------------------

def add_zones(design: Design) -> None:
    ff, lay = design.form_factor, design.layout
    outline = rect(1.0, 1.0, ff.width - 1.0, ff.depth - 1.0)
    for layer in GND_LAYERS:
        design.zones.append(Zone(layer, "GND", outline))
    # 12 V from the supply inputs across the top rail layer; the I/O rail as a
    # disc under the whole ring; the memory-PHY rail as an island under each
    # chip's two memories, rotated with them.
    design.zones.append(Zone(TOP_RAIL_LAYER, "+12V", outline, priority=0))
    asic_body = design.pinout.package.body
    design.zones.append(Zone(TOP_RAIL_LAYER, "VDD_IO_1V8", regular_polygon(lay.centre, lay.radius + asic_body / 2 + 2.0, 36), priority=1))
    memory_parts = [p for p in design.parts if p.part_class == "lpddr5x"]
    if memory_parts:
        phy_rail = design.board.classes["lpddr5x"]["rails"][1]     # VDD2H: the PHY rail shared with the ASIC
        for part in design.parts:
            if part.part_class == "layer_asic":
                design.zones.append(Zone(TOP_RAIL_LAYER, phy_rail,
                                         local_rect(part, -19.5, part.body_h / 2 + 1.0, 19.5, part.body_h / 2 + 15.5), priority=2))
    # Core rails: one island per ASIC covering the package and its regulator on the inner edge.
    for part in design.parts:
        if part.part_class in ("layer_asic", "head_asic"):
            design.zones.append(Zone(CORE_RAIL_LAYER, f"VDD_CORE_{short_ref(part.ref)}",
                                     local_rect(part, -(part.body_w / 2 + 1.0), -(part.body_h / 2 + 2.5 + VRM_H + 1.0),
                                                part.body_w / 2 + 1.0, part.body_h / 2 + 1.0), 1))
    fpga = design.part(design.board.instances("fpga")[0])
    design.zones.append(Zone(CORE_RAIL_LAYER, "VCCINT_0V85",
                             local_rect(fpga, -15.0, -(fpga.body_h / 2 + 2.5 + REG_H + 1.0), 15.0, 15.0), 1))


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
    board, ff, lay = design.board, design.form_factor, design.layout
    lines = ["# KiCad floorplan report", "",
             "Generated by `python -m hw.kicad_gen` from `hw/board.yaml`. Do not edit.", "",
             f"{ff.description}: {ff.width:.0f} x {ff.depth:.0f} mm, {len(COPPER_LAYERS)} copper layers, "
             f"{len(design.parts)} footprints, {len(design.nets())} nets, {len(design.tracks)} track segments, "
             f"{len(design.vias)} vias, {len(design.zones)} zones. Keep-outs: "
             + "; ".join(f"{name} ({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})" for name, (x0, y0, x1, y1) in ff.keepouts) + ".", "",
             "## Stackup", "", "| Layer | Role |", "| --- | --- |"]
    lines += [f"| {layer} | {role} |" for layer, role in LAYER_ROLES.items()]
    pin = design.pinout
    pitch = pin.package.pitch
    n = len(lay.nodes)
    lines += ["", "## Activation ring", "",
              f"The ring is a regular {n}-gon of {lay.side:.0f} mm side ({lay.radius:.0f} mm circumradius) centred at "
              f"({lay.centre[0]:.0f}, {lay.centre[1]:.0f}), one ring node per vertex, each chip rotated tangentially "
              f"(its link-out edge faces the next chip {360 / n:.1f} degrees round), memories outward, core regulator inward. "
              f"Every hop is one 36-lane ribbon on {LINK_LAYER} at {pitch / 2} mm lane pitch (outer-column and dogbone vias "
              f"interleaved), {TRACK_WIDTH} mm tracks, leaving and entering the ports straight for {PATH_RUN:.0f} mm with "
              f"two bends between. Limit {MAX_LINK_MM:.0f} mm per `board.yaml`.", "",
              "| Hop | Shortest lane (mm) | Longest lane (mm) | Sharpest bend | Within limit |", "| --- | ---: | ---: | ---: | --- |"]
    for hop, (lo, hi) in design.hop_lengths.items():
        lines.append(f"| {hop} | {lo:.1f} | {hi:.1f} | {design.hop_bends[hop]:.0f} deg | {'yes' if hi <= MAX_LINK_MM else 'NO'} |")
    pkg = pin.package
    need = pin.requirements
    signal_rows = int(board.data["package_selection"]["signal_rows"])
    deep = sum(1 for b in pin.balls if b.kind == "signal" and
               min(b.row, b.col, pkg.rows - 1 - b.row, pkg.cols - 1 - b.col) >= signal_rows)
    memory_balls = [b for b in pin.balls if b.kind == "signal" and b.interface.startswith("lpddr")]
    if memory_balls:
        memory_rows = 1 + max(b.row for b in memory_balls)
        memory_text = (f"{len(memory_balls)} LPDDR signals on the north {memory_rows} rows, {len(memory_balls) / pkg.body:.1f} per mm "
                       f"of edge, of which {deep} sit deeper than the outer {signal_rows} rows and need microvias or a build-up "
                       "layer pair to escape. The memory nets are not routed here.")
    else:
        memory_text = "No memory balls: the HBM stack sits on the package interposer."
    lines += ["", "## ASIC package and escape density", "",
              f"{pkg.name}: {pkg.cols}x{pkg.rows} balls at {pkg.pitch} mm, {pkg.body:.0f} mm body, selected by "
              f"`hw/pinout.py` for {need.core_amps:.0f} A of core current at {need.rated_tokens_per_second:.0f} tokens/s "
              f"({need.total} balls needed; see the pinout report). "
              f"Per edge: 36 link signals on two columns of 18 rows ({36 / (18 * pkg.pitch):.1f} signals per mm of edge); "
              + memory_text, "",
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
              "* The ASIC ball map is the rule-derived one from hw/pinout.py; the final one comes from the package design.",
              "* The FPGA link pins are assigned by the ribbon router; every other FPGA pin is a placeholder.",
              "* The BMC, PSU and fan connectors are placeholder blocks with their nets; no chassis mechanicals."]
    lines += ["", "## Notes", ""] + [f"* {note}" for note in design.notes]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# KiCad writers
# --------------------------------------------------------------------------

def kicad_xy(design: Design, x: float, y: float) -> Point:
    """Board point to KiCad page coordinates (y down)."""
    return x + KICAD_ORIGIN[0], KICAD_ORIGIN[1] + (design.form_factor.depth - y)


def write_pcb(design: Design, path: Path) -> None:
    ff = design.form_factor
    nets = ["", *design.nets()]
    net_id = {name: index for index, name in enumerate(nets)}

    def K(x: float, y: float) -> str:
        kx, ky = kicad_xy(design, x, y)
        return f"{fmt(kx)} {fmt(ky)}"

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
    out += ["      (copper_finish \"ENIG\")", "    )", "    (pad_to_mask_clearance 0)", "  )", ""]
    for name, index in net_id.items():
        out.append(f'  (net {index} "{name}")')
    out.append("")
    # Outline and keep-outs.
    corners = [(0.0, 0.0), (ff.width, 0.0), (ff.width, ff.depth), (0.0, ff.depth)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        out.append(f'  (gr_line (start {K(*a)}) (end {K(*b)}) '
                   f'(stroke (width 0.1) (type default)) (layer "Edge.Cuts") (tstamp {uid()}))')
    for name, (x0, y0, x1, y1) in ff.keepouts:
        out.append(f'  (gr_rect (start {K(x0, y1)}) (end {K(x1, y0)}) '
                   f'(stroke (width 0.1) (type dash)) (fill none) (layer "Dwgs.User") (tstamp {uid()}))')
        out.append(f'  (gr_text "{name}" (at {K((x0 + x1) / 2, (y0 + y1) / 2)}) (layer "Dwgs.User") (tstamp {uid()}) '
                   f'(effects (font (size 2 2) (thickness 0.3))))')
    out.append(f'  (gr_text "{PROJECT}: {ff.description}, floorplan generated from hw/board.yaml" '
               f'(at {K(ff.width / 2, ff.depth + 6)}) (layer "Cmts.User") (tstamp {uid()}) '
               f'(effects (font (size 3 3) (thickness 0.4))))')
    out.append("")
    # Footprints: written unrotated at the part position, with every pad and
    # outline point already rotated, so any angle works the same way.
    for part in design.parts:
        px, py = kicad_xy(design, part.x, part.y)
        out.append(f'  (footprint "appliance:{part.footprint}" (layer "F.Cu") (tstamp {uid()}) (at {fmt(px)} {fmt(py)})')
        out.append(f'    (property "Sheetfile" "{PROJECT}.kicad_sch")')
        out.append("    (attr smd)")
        # Reference and value inside the body so nothing is clipped at the board edge.
        out.append(f'    (fp_text reference "{part.ref}" (at 0 0) (layer "F.SilkS") (tstamp {uid()})'
                   f' (effects (font (size 1.5 1.5) (thickness 0.2))))')
        out.append(f'    (fp_text value "{part.value}" (at 0 2) (layer "F.Fab") (tstamp {uid()})'
                   f' (effects (font (size 1 1) (thickness 0.15))))')

        def local(bx: float, by: float) -> Point:
            return bx - part.x, -(by - part.y)

        if part.body_w > 0:
            for layer, grow in (("F.SilkS", 0.0), ("F.CrtYd", 0.5), ("F.Fab", 0.0)):
                pts = " ".join(f"(xy {fmt(lx)} {fmt(ly)})" for lx, ly in (local(*c) for c in part.corners(grow)))
                out.append(f'    (fp_poly (pts {pts}) (stroke (width 0.1) (type default)) (fill none) '
                           f'(layer "{layer}") (tstamp {uid()}))')
        if part.package:
            # Pin-1 mark beside ball A1.
            ax, ay = local(*part.local_to_board(*part.package.ball_xy(0, 0)))
            out.append(f'    (fp_circle (center {fmt(ax - 0.8)} {fmt(ay - 0.8)}) (end {fmt(ax - 0.4)} {fmt(ay - 0.8)}) '
                       f'(stroke (width 0.15) (type default)) (fill none) (layer "F.SilkS") (tstamp {uid()}))')
        for pad in part.pads:
            lx, ly = local(*part.local_to_board(pad.x, pad.y))
            angle = f" {fmt(part.rotation)}" if pad.shape == "rect" and part.rotation else ""
            net = f' (net {net_id[pad.net]} "{pad.net}")' if pad.net else ""
            out.append(f'    (pad "{pad.name}" smd {pad.shape} (at {fmt(lx)} {fmt(ly)}{angle}) (size {fmt(pad.size[0])} {fmt(pad.size[1])}) '
                       f'(layers {pad.layers}){net} (tstamp {uid()}))')
        out.append("  )")
    out.append("")
    for track in design.tracks:
        for a, b in zip(track.points, track.points[1:]):
            out.append(f'  (segment (start {K(*a)}) (end {K(*b)}) '
                       f'(width {fmt(track.width)}) (layer "{track.layer}") (net {net_id[track.net]}) (tstamp {uid()}))')
    for via in design.vias:
        out.append(f'  (via (at {K(via.x, via.y)}) (size {fmt(via.size)}) (drill {fmt(via.drill)}) '
                   f'(layers "F.Cu" "B.Cu") (net {net_id[via.net]}) (tstamp {uid()}))')
    out.append("")
    for zone in design.zones:
        pts = " ".join(f"(xy {K(x, y)})" for x, y in zone.polygon)
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
              "fpga": lambda p: p.part_class in ("fpga", "clock_gen", "sfp_cage", "host_cable", "bmc", "rj45"),
              "power": lambda p: p.part_class in ("vrm_core", "regulator", "psu_input", "fan_header")}
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
    root.append('  (sheet_instances (path "/" (page "1")))')
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
    """A light rendering of the placement and the ring ribbons on the physical
    board (the KiCad SVG export of the same board is tens of megabytes because
    of the zone fills)."""
    ff = design.form_factor
    scale = 3.0
    w, h = ff.width * scale + 40, ff.depth * scale + 70
    fills = {"layer_asic": "#dbe8f7", "head_asic": "#f7dbdb", "lpddr5x": "#e8f3e8", "ddr4": "#e8f3e8",
             "fpga": "#fff2cc", "vrm_core": "#f3e6d0", "regulator": "#eeeeee", "clock_gen": "#eeeeee",
             "sfp_cage": "#dddddd", "host_cable": "#dddddd", "psu_input": "#dddddd", "fan_header": "#dddddd",
             "bmc": "#fff2cc", "rj45": "#dddddd"}

    def sx(x: float) -> float:
        return 20 + x * scale

    def sy(y: float) -> float:
        return 20 + (ff.depth - y) * scale

    def poly(points: list[Point]) -> str:
        return " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" font-family="sans-serif" font-size="11">',
             f'<rect width="{w:.0f}" height="{h:.0f}" fill="#fafafa"/>',
             f'<rect x="{sx(0)}" y="{sy(ff.depth)}" width="{ff.width * scale}" height="{ff.depth * scale}" '
             f'fill="#f4f7f4" stroke="#333" stroke-width="1.5"/>']
    for name, (x0, y0, x1, y1) in ff.keepouts:
        parts.append(f'<rect x="{sx(x0)}" y="{sy(y1)}" width="{(x1 - x0) * scale}" height="{(y1 - y0) * scale}" '
                     f'fill="none" stroke="#999" stroke-dasharray="4,3"/>')
        parts.append(f'<text x="{sx(x0) + 4}" y="{sy(y1) + 12}" fill="#777">{name}</text>')
    for zone in design.zones:
        if zone.layer == CORE_RAIL_LAYER:
            parts.append(f'<polygon points="{poly(zone.polygon)}" fill="#f9e9e9" stroke="none"/>')
    for part in design.parts:
        if part.body_w == 0:
            continue
        parts.append(f'<polygon points="{poly(part.corners())}" fill="{fills.get(part.part_class, "#eee")}" '
                     f'stroke="#333" stroke-width="1"/>')
        size = 10 if min(part.body_w, part.body_h) > 15 else 6
        parts.append(f'<text x="{sx(part.x)}" y="{sy(part.y) + size / 3}" text-anchor="middle" font-size="{size}">{part.ref}</text>')
    for track in design.tracks:
        if track.layer == "F.Cu":
            continue
        parts.append(f'<polyline points="{poly(track.points)}" fill="none" stroke="#1f5fbf" stroke-width="0.6"/>')
    parts.append(f'<text x="{sx(2)}" y="{h - 40}" fill="#333">{ff.description}. Generated from hw/board.yaml by '
                 f'hw/kicad_gen.py. Blue: ring ribbons on {LINK_LAYER}; pink: core-rail islands on {CORE_RAIL_LAYER}. '
                 f'Rear of the chassis at the bottom.</text>')
    parts.append(f'<text x="{sx(2)}" y="{h - 24}" fill="#333">Memory, PCIe and management nets are unrouted. Longest lane per hop: '
                 + "; ".join(f"{hop.replace('U_', '')} {hi:.0f} mm" for hop, (_, hi) in design.hop_lengths.items()) + "</text>")
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
    parser.add_argument("--output", type=Path, default=None, help="defaults to hw/kicad for board.yaml, hw/kicad_<suffix> otherwise")
    parser.add_argument("--check", action="store_true", help="generate into a temporary directory and print the report")
    args = parser.parse_args()
    if args.output is None:
        args.output = OUTPUT_DIR if args.source.stem == "board" else HERE / f"kicad_{args.source.stem.replace('board_', '')}"
    board = Board.load(args.source)
    if args.check:
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            design = generate(board, Path(directory))
            print(report_markdown(design))
        return
    design = generate(board, args.output)
    lay = design.layout
    print(f"wrote {args.output}/{PROJECT}.kicad_pcb ({design.form_factor.description}): {len(design.parts)} footprints, "
          f"{len(design.nets())} nets, {len(design.tracks)} tracks, {len(design.vias)} vias, "
          f"ASIC package {design.pinout.package.name}, ring {len(lay.nodes)}-gon of {lay.side:.0f} mm side")
    for hop, (lo, hi) in design.hop_lengths.items():
        print(f"  {hop}: {lo:.1f} to {hi:.1f} mm, bend {design.hop_bends[hop]:.0f} deg"
              f"{'' if hi <= MAX_LINK_MM else '  OVER LIMIT'}")


if __name__ == "__main__":
    main()
