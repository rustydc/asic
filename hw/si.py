"""Signal integrity of the PSRAM clock nets on the module card.

The first cut of the card had four APS512XXN devices sharing one 250 MHz
single-ended 1.8 V clock from the layer ASIC.  Whether that survives is a
question of the net's shape and the driver's strength (it does not: see
the report), so this is a small time-domain field solver for
lossless transmission-line networks: every trace is an LC ladder of 1 mm
segments (0.31 nH and 0.125 pF for a 50 ohm line at 160 mm/ns), solved by
the leapfrog scheme at 2 ps steps, with the driver a Thevenin source
behind a resistor and each receiver the datasheet's 5 pF package input
behind a 1 nH ball inductance plus half a picofarad of pad and via.  The
result at each receiver's die is judged against the datasheet: the clock
edge (tKHKL, at most 0.6 ns at 250 MHz), the absolute maximum ratings
(-0.4 V to VDD + 0.4 V), a single clean crossing of the VIL/VIH band
(0.4 V and VDD - 0.4 V) per edge, the duty cycle (45 to 55 percent), and
the skew between the four devices, which is not a datasheet limit but eats
the controller's quarter-period margin.

``python -m hw.si`` writes ``hw/si_psram_clock.md``.  The physical
constants are placeholders for the card's stack-up; they are named here so
the study can be rerun when the fabricator's numbers arrive.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Physical constants (placeholders for the card's stack-up)
# --------------------------------------------------------------------------

Z0 = 50.0                    # ohm, single-ended trace impedance
V_MM_NS = 160.0              # mm/ns, propagation velocity (er about 3.5)
SEG_MM = 1.0                 # segment length of the LC ladder
DT_PS = 2.0                  # time step; stable below sqrt(LC) = 6.25 ps per segment
L_MM = Z0 / V_MM_NS * 1e-9   # H/mm  (0.3125 nH)
C_MM = 1.0 / (Z0 * V_MM_NS) * 1e-9  # F/mm (0.125 pF)

VDD = 1.8
F_CLK_MHZ = 250.0
T_NS = 1000.0 / F_CLK_MHZ
T_RAMP_NS = 0.35             # the driver's own 0 to 100 percent edge into no load (0.21 ns 20 to 80)

# The receiver: APS512XXN datasheet tables 24 and 27 and section 8.5.
C_IN_PKG = 5e-12             # package input pin capacitance
L_PKG = 1e-9                 # ball and bond inductance (assumed for the 24-ball BGA)
C_PAD = 0.5e-12              # the board pad and its via
VIL = 0.4
VIH = VDD - 0.4
T_KHKL_MAX_NS = 0.6          # CLK rise or fall time at 250 MHz
OVERSHOOT_V = 0.4            # absolute maximum: -0.4 V to VDD + 0.4 V
DUTY_MIN, DUTY_MAX = 0.45, 0.55
SKEW_BUDGET_NS = 0.4         # our own: what the quarter-period launch margin can spare across a group


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------

class Network:
    """Nodes with capacitance to ground, inductive branches between nodes,
    resistive shunts from a node to a fixed voltage, and one Thevenin driver."""

    def __init__(self) -> None:
        self.cap: list[float] = []
        self.branches: list[tuple[int, int, float]] = []
        self.shunts: list[tuple[int, float, float]] = []     # node, conductance, reference voltage
        self.loads: dict[str, int] = {}                     # name -> die node
        self.driver: tuple[int, float] | None = None        # node, source resistance

    def node(self, c: float = 0.0) -> int:
        self.cap.append(c)
        return len(self.cap) - 1

    def line(self, start: int, length_mm: float) -> int:
        """A trace of `length_mm` from `start`; returns its far node."""
        n = max(1, int(round(length_mm / SEG_MM)))
        seg = length_mm / n
        node = start
        for _ in range(n):
            self.cap[node] += C_MM * seg / 2
            end = self.node(C_MM * seg / 2)
            self.branches.append((node, end, L_MM * seg))
            node = end
        return node

    def load(self, node: int, name: str) -> int:
        """A receiver at `node`: pad and via, the ball inductance, the package input capacitance."""
        self.cap[node] += C_PAD
        die = self.node(C_IN_PKG)
        self.branches.append((node, die, L_PKG))
        self.loads[name] = die
        return die

    def shunt(self, node: int, ohms: float, vref: float) -> None:
        self.shunts.append((node, 1.0 / ohms, vref))

    def drive(self, node: int, ohms: float) -> None:
        self.driver = (node, ohms)


def clock_source(t_ns: np.ndarray) -> np.ndarray:
    """The driver's open-circuit voltage: a 50 percent trapezoid at F_CLK, low at t = 0."""
    phase = np.mod(t_ns, T_NS)
    v = np.where(phase < T_RAMP_NS, phase / T_RAMP_NS,
                 np.where(phase < T_NS / 2, 1.0,
                          np.where(phase < T_NS / 2 + T_RAMP_NS, 1.0 - (phase - T_NS / 2) / T_RAMP_NS, 0.0)))
    return VDD * v


def simulate(net: Network, t_end_ns: float, dt_ps: float = DT_PS) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Leapfrog integration; returns the time axis, each load's die voltage and the driver node's voltage."""
    assert net.driver is not None
    dt = dt_ps * 1e-12
    n = len(net.cap)
    cap = np.array(net.cap)
    a = np.array([b[0] for b in net.branches], dtype=int)
    b = np.array([b[1] for b in net.branches], dtype=int)
    ind = np.array([br[2] for br in net.branches])
    g = np.zeros(n)
    gv = np.zeros(n)
    for node, cond, vref in net.shunts:
        g[node] += cond
        gv[node] += cond * vref
    src_node, src_r = net.driver
    g_src = 1.0 / src_r
    steps = int(round(t_end_ns * 1e-9 / dt))
    t = np.arange(steps) * dt_ps * 1e-3
    vs = clock_source(t + dt_ps * 0.5e-3)
    v = np.zeros(n)
    i = np.zeros(len(ind))
    watch = {name: node for name, node in net.loads.items()}
    hist = {name: np.zeros(steps) for name in watch}
    src_hist = np.zeros(steps)
    dt_c = dt / cap
    for k in range(steps):
        i += dt / ind * (v[a] - v[b])
        j = np.bincount(b, i, n) - np.bincount(a, i, n)
        inject = gv.copy()
        inject[src_node] += g_src * vs[k]
        gtot = g.copy()
        gtot[src_node] += g_src
        v = (v + dt_c * (j + inject)) / (1.0 + dt_c * gtot)
        for name, node in watch.items():
            hist[name][k] = v[node]
        src_hist[k] = v[src_node]
    return t, hist, src_hist


# --------------------------------------------------------------------------
# Topologies
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Topology:
    name: str
    description: str
    build: object                # (net) -> None, from the driver node 0

    def network(self, r_drive: float) -> Network:
        net = Network()
        root = net.node()
        net.drive(root, r_drive)
        self.build(net, root)
        return net


def point_to_point(trunk_mm: float) -> Topology:
    def build(net: Network, root: int) -> None:
        net.load(net.line(root, trunk_mm), "d0")
    return Topology("point-to-point", f"one clock per device, {trunk_mm:.0f} mm", build)


def star(trunk_mm: float, stubs_mm: tuple[float, ...]) -> Topology:
    def build(net: Network, root: int) -> None:
        junction = net.line(root, trunk_mm)
        for k, stub in enumerate(stubs_mm):
            net.load(net.line(junction, stub), f"d{k}")
    stubs = "/".join(f"{s:.0f}" for s in stubs_mm)
    return Topology(f"star of {len(stubs_mm)}", f"{trunk_mm:.0f} mm trunk to a junction, stubs of {stubs} mm", build)


def fly_by(trunk_mm: float, pitch_mm: float, n: int, stub_mm: float = 1.5, end_ohms: float | None = None) -> Topology:
    def build(net: Network, root: int) -> None:
        node = net.line(root, trunk_mm)
        for k in range(n):
            net.load(net.line(node, stub_mm), f"d{k}")
            if k < n - 1:
                node = net.line(node, pitch_mm)
        if end_ohms:
            net.shunt(node, end_ohms, VDD / 2)
    term = f", {end_ohms:.0f} ohm AC termination at the end" if end_ohms else ""
    return Topology(f"fly-by of {n}" + (", terminated" if end_ohms else ""),
                    f"{trunk_mm:.0f} mm trunk, devices {pitch_mm:.0f} mm apart on {stub_mm:.1f} mm stubs{term}", build)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def crossings(t: np.ndarray, v: np.ndarray, level: float, rising: bool) -> np.ndarray:
    """Times at which v crosses `level` in the given direction, by linear interpolation."""
    lo, hi = v[:-1], v[1:]
    mask = (lo < level) & (hi >= level) if rising else (lo > level) & (hi <= level)
    idx = np.nonzero(mask)[0]
    frac = (level - lo[idx]) / (hi[idx] - lo[idx])
    return t[idx] + frac * (t[idx + 1] - t[idx])


@dataclasses.dataclass
class LoadMetrics:
    rise_ns: float               # 20 to 80 percent
    fall_ns: float               # 80 to 20 percent
    v_max: float
    v_min: float
    clean: bool                  # one crossing of each threshold per edge in the window
    duty: float                  # fraction of the period above VDD/2
    delay_ns: float              # VDD/2 rising crossing, launch to die

    @property
    def ok(self) -> bool:
        return (max(self.rise_ns, self.fall_ns) <= T_KHKL_MAX_NS and self.v_max <= VDD + OVERSHOOT_V
                and self.v_min >= -OVERSHOOT_V and self.clean and DUTY_MIN <= self.duty <= DUTY_MAX)


def analyse(t: np.ndarray, v: np.ndarray, cycle: int = 4) -> LoadMetrics:
    """Metrics over one period starting half a nanosecond before the driver's rising edge of `cycle`;
    the delay is from the launch (the open-circuit source's midpoint) to the die's midpoint."""
    t0 = cycle * T_NS - 0.5
    win = (t >= t0) & (t < t0 + T_NS)
    tw, vw = t[win], v[win]
    r20 = crossings(tw, vw, 0.2 * VDD, True)
    r80 = crossings(tw, vw, 0.8 * VDD, True)
    f80 = crossings(tw, vw, 0.8 * VDD, False)
    f20 = crossings(tw, vw, 0.2 * VDD, False)
    rise = r80[0] - r20[0] if len(r20) and len(r80) else math.inf
    fall = f20[-1] - f80[-1] if len(f20) and len(f80) else math.inf
    clean = all(len(c) == 1 for c in (crossings(tw, vw, VIL, True), crossings(tw, vw, VIH, True),
                                       crossings(tw, vw, VIH, False), crossings(tw, vw, VIL, False)))
    duty = float(np.mean(vw > VDD / 2))
    src_rise = crossings(tw, clock_source(tw), VDD / 2, True)
    mid = crossings(tw, vw, VDD / 2, True)
    delay = mid[0] - src_rise[0] if len(mid) and len(src_rise) else math.inf
    return LoadMetrics(rise, fall, float(vw.max()), float(vw.min()), clean, duty, delay)


@dataclasses.dataclass
class Result:
    topology: Topology
    r_drive: float
    loads: dict[str, LoadMetrics]

    @property
    def skew_ns(self) -> float:
        delays = [m.delay_ns for m in self.loads.values()]
        return max(delays) - min(delays)

    @property
    def worst_edge_ns(self) -> float:
        return max(max(m.rise_ns, m.fall_ns) for m in self.loads.values())

    @property
    def ok(self) -> bool:
        return all(m.ok for m in self.loads.values()) and self.skew_ns <= SKEW_BUDGET_NS


def run(topology: Topology, r_drive: float, cycles: int = 6) -> Result:
    net = topology.network(r_drive)
    t, hist, _ = simulate(net, cycles * T_NS)
    return Result(topology, r_drive, {name: analyse(t, v) for name, v in hist.items()})


# --------------------------------------------------------------------------
# The study
# --------------------------------------------------------------------------

# The card as generated: the clock balls sit on the die's south edge with
# the small interfaces and the PSRAM rows stand north of the chip, so the
# trunk goes round the 23 mm package: about 55 mm.  With the clock balls
# moved to the north edge beside the memory it is about 20 mm.
TRUNK_AS_PLACED_MM = 55.0
TRUNK_NORTH_MM = 20.0
DEVICE_PITCH_MM = 8.0        # 6 mm body plus 2 mm gap along the row
R_DRIVE_SWEEP = (12.0, 18.0, 25.0, 33.0, 50.0)


def topologies(trunk_mm: float) -> list[Topology]:
    return [
        point_to_point(trunk_mm),
        star(trunk_mm, (4.0, 4.0)),
        star(trunk_mm, (12.0, 4.0, 4.0, 12.0)),
        fly_by(trunk_mm, DEVICE_PITCH_MM, 4),
        fly_by(trunk_mm, DEVICE_PITCH_MM, 4, end_ohms=50.0),
    ]


def study(trunks: tuple[float, ...] = (TRUNK_NORTH_MM, TRUNK_AS_PLACED_MM),
          r_sweep: tuple[float, ...] = R_DRIVE_SWEEP) -> list[tuple[float, list[Result]]]:
    out = []
    for trunk in trunks:
        results = [run(topo, r) for topo in topologies(trunk) for r in r_sweep]
        out.append((trunk, results))
    return out


def best(results: list[Result], topology_name: str) -> Result | None:
    """The passing result with the largest drive resistance (the weakest driver that works)."""
    passing = [r for r in results if r.topology.name == topology_name and r.ok]
    return max(passing, key=lambda r: r.r_drive) if passing else None


POINT_TO_POINT_LENGTHS_MM = (20.0, 40.0, 55.0, 70.0, 85.0)   # the nearest to the farthest device from the south-edge balls
POINT_TO_POINT_DRIVES = (33.0, 50.0)


def point_to_point_sweep(lengths: tuple[float, ...] = POINT_TO_POINT_LENGTHS_MM,
                         drives: tuple[float, ...] = POINT_TO_POINT_DRIVES) -> list[Result]:
    return [run(point_to_point(length), r) for length in lengths for r in drives]


def _ns(x: float) -> str:
    return "none" if math.isinf(x) else f"{x:.2f} ns"


def report_markdown(results_by_trunk: list[tuple[float, list[Result]]], p2p: list[Result] | None = None) -> str:
    lines = [
        "# The shared PSRAM clock: signal integrity",
        "",
        "Generated by `python -m hw.si`; the model and its constants are in `hw/si.py`.",
        "",
        f"Lossless LC-ladder time-domain solution, {Z0:.0f} ohm traces at {V_MM_NS:.0f} mm/ns in {SEG_MM:.0f} mm "
        f"segments at {DT_PS:.0f} ps; the driver a {T_RAMP_NS:.2f} ns ramp behind the drive resistance (the pad "
        f"driver plus any series resistor); each receiver {C_PAD * 1e12:.1f} pF of pad, {L_PKG * 1e9:.0f} nH of "
        f"ball and the datasheet's {C_IN_PKG * 1e12:.0f} pF package input. Limits: edges (20 to 80 percent) at most "
        f"{T_KHKL_MAX_NS:.1f} ns (tKHKL at 250 MHz), {-OVERSHOOT_V:.1f} V to VDD + {OVERSHOOT_V:.1f} V (absolute maximum), "
        f"one clean crossing of the {VIL:.1f} / {VIH:.1f} V band per edge, duty {DUTY_MIN * 100:.0f} to {DUTY_MAX * 100:.0f} "
        f"percent, and a group skew of at most {SKEW_BUDGET_NS:.1f} ns (our launch-margin budget, not a datasheet limit).",
        "",
    ]
    for trunk, results in results_by_trunk:
        where = "clock balls on the north edge beside the memory" if trunk <= 30 else "clock balls on the south edge as placed, round the package"
        lines += [f"## Trunk {trunk:.0f} mm ({where})", "",
                  "| Topology | Drive | Worst edge | Peak | Trough | Clean | Duty | Skew | Verdict |",
                  "| --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | :---: |"]
        for r in results:
            duties = [m.duty for m in r.loads.values()]
            lines.append(
                f"| {r.topology.name} | {r.r_drive:.0f} ohm | {_ns(r.worst_edge_ns)} | "
                f"{max(m.v_max for m in r.loads.values()):.2f} V | {min(m.v_min for m in r.loads.values()):.2f} V | "
                f"{'yes' if all(m.clean for m in r.loads.values()) else 'no'} | "
                f"{min(duties) * 100:.0f} to {max(duties) * 100:.0f} % | {r.skew_ns:.2f} ns | {'pass' if r.ok else 'fail'} |")
        lines.append("")
        lines.append("Topologies: " + "; ".join(f"**{t.name}**, {t.description}" for t in topologies(trunk)) + ".")
        lines.append("")
    if p2p:
        lines += ["## One clock per device, by trace length", "",
                  "| Length | Drive | Edge | Peak | Trough | Delay | Verdict |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | :---: |"]
        for r in p2p:
            m = r.loads["d0"]
            lines.append(f"| {r.topology.description.split(', ')[1]} | {r.r_drive:.0f} ohm | {_ns(max(m.rise_ns, m.fall_ns))} | "
                         f"{m.v_max:.2f} V | {m.v_min:.2f} V | {m.delay_ns:.2f} ns | {'pass' if r.ok else 'fail'} |")
        lines.append("")
    lines += ["## Reading", "",
              "Four devices on one clock is 20 pF of package input on the net, and an edge that meets tKHKL into 20 pF "
              "needs a drive under 20 ohm, which then rings through the absolute maximum ratings; the star of four passes "
              "at one drive value on the short trunk and nowhere on the trunk as placed, and the fly-by, whose loaded line "
              "slows to a third of its speed, passes nowhere with or without an end termination. Two per clock passes in a "
              "narrow band around 33 ohm. One clock per device passes from 33 to 50 ohm at every length on the card, with "
              "a 50 ohm total drive (the pad driver plus a series resistor) giving a reflection-free edge of 0.4 ns. So the "
              "die carries sixteen clocks (`psram_clk`, twelve more balls in the small-interface rows) and each is a "
              "series-terminated point-to-point trace.", ""]
    return "\n".join(lines)


def main() -> None:
    results = study()
    text = report_markdown(results, point_to_point_sweep())
    out = Path(__file__).with_name("si_psram_clock.md")
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
