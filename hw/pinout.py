"""Derive the ASIC package and ball map from ``hw/board.yaml``.

    python -m hw.pinout            # write hw/pinout/asic_ballmap.{csv,json} and report.md
    python -m hw.pinout --check    # print the selection and the report only

The package is not designed here; the packaging house designs the substrate
from the bump map.  What this module produces is the thing they need from
us: the ball count the power model demands, the smallest candidate package
that carries it, and a ball map that follows the rules a substrate and a PCB
can both route.

Rules, all from ``package_selection`` in the YAML:

* the core rail needs one ball per ``amps_per_ball`` at the rated
  throughput, and ground matches it plus one return per
  ``signals_per_ground`` signal balls;
* every signal sits in the outer ``signal_rows`` rows, where a through via
  can escape it on the PCB;
* each interface owns one package edge (``edges``), so the die floorplan puts
  its macro on that die edge: the links on the west and east edges as two
  columns of rows, the memory channels on the north rows (or the north and
  south rows, ``memory: [N, S]``) one channel after another with a ground
  after each lane, the small interfaces on the south row or, when the south
  rows hold memory, on the west columns outside the link port;
* ``link_out_mirrored`` wires the link-out lanes in reverse ball order, which
  a module card needs: its ribbon leaves the package on one edge, bends down
  to the card-edge fingers, and the mirror puts each signal at the same
  finger position on both ports so the motherboard hops are straight;
* the remaining balls alternate ground and the core rail, with a few I/O and
  memory-PHY rail balls; the same map serves the head ASIC with the memory
  balls unconnected.

The KiCad generator (``hw/kicad_gen.py``) builds the ASIC footprint from this
map, so the escape geometry and the via-in-pad counts follow it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from hw.board import Board

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "pinout"

_JEDEC = "ABCDEFGHJKLMNPRTUVWY"
ROW_LETTERS = list(_JEDEC) + [p + c for p in _JEDEC for c in _JEDEC]   # A..Y, AA..AY, BA.., 420 rows


@dataclass(frozen=True)
class PackageSpec:
    name: str
    cols: int
    rows: int
    pitch: float
    body: float
    ball: float

    @property
    def balls(self) -> int:
        return self.cols * self.rows

    def outer_balls(self, depth: int) -> int:
        inner = max(0, self.cols - 2 * depth) * max(0, self.rows - 2 * depth)
        return self.balls - inner

    def ball_name(self, i: int, j: int) -> str:
        return f"{ROW_LETTERS[i]}{j + 1}"

    def ball_xy(self, i: int, j: int) -> tuple[float, float]:
        """Ball position relative to the package centre, y up, row 0 north."""
        return (j - (self.cols - 1) / 2) * self.pitch, ((self.rows - 1) / 2 - i) * self.pitch


@dataclass(frozen=True)
class Ball:
    name: str
    row: int
    col: int
    x: float
    y: float
    kind: str                  # "signal", "ground", "rail", "nc"
    interface: str = ""        # link_in, link_out, lpddr_ch0, lpddr_ch1, mgmt, jtag, refclk, strap, or the rail name
    signal: str = ""           # signal name within the interface
    escape: tuple[str, int] | None = None   # (edge, depth) for ring-link balls: depth 0 = outer column


@dataclass
class Requirements:
    rated_tokens_per_second: float
    core_amps: float
    signal_balls: int
    signal_grounds: int
    core_balls: int
    ground_balls: int
    rail_balls: dict[str, int]

    @property
    def total(self) -> int:
        return self.signal_balls + self.signal_grounds + self.core_balls + self.ground_balls + sum(self.rail_balls.values())


@dataclass
class Pinout:
    package: PackageSpec
    requirements: Requirements
    balls: list[Ball]
    rejected: list[tuple[str, str]] = field(default_factory=list)   # (candidate, reason)

    def by_name(self) -> dict[str, Ball]:
        return {ball.name: ball for ball in self.balls}

    def count(self, kind: str, interface: str | None = None) -> int:
        return sum(1 for b in self.balls if b.kind == kind and (interface is None or b.interface == interface))


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

def expand_signals(signals: dict[str, int]) -> list[str]:
    """dq: 32 -> DQ0..DQ31; dqs_pairs: 4 -> DQS0_P, DQS0_N, ...; clock: 1 -> CLK."""
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


def lpddr_lanes(signals: list[str]) -> list[list[str]]:
    """Group an LPDDR x32 channel into four byte lanes (DQ, DQS pair, DMI) and a
    command group (CA, CS, CK pair, RESET), each followed by a ground on the package."""
    lanes = []
    for k in range(4):
        lanes.append([s for s in signals if s.startswith("DQ") and not s.startswith("DQS") and 8 * k <= int(s[2:]) < 8 * k + 8]
                     + [f"DQS{k}_P", f"DQS{k}_N", f"DMI{k}"])
    command = [s for s in signals if s.startswith(("CA", "CS", "CK", "RESET"))]
    lanes.append(command)
    return lanes


def memory_lanes(signals: list[str]) -> list[list[str]]:
    """Lanes of one memory channel, each followed by a ground ball: the LPDDR
    byte lanes above, or the whole port as one lane for a narrow device such
    as an x16 PSRAM."""
    if any(s.startswith("DQS") for s in signals) and any(s.startswith("CA") for s in signals):
        return lpddr_lanes(signals)
    return [list(signals)]


def memory_edges(board: Board) -> list[str]:
    """Package edges that hold the memory channels, ``N`` unless the rules say otherwise."""
    edges = board.data["package_selection"]["edges"]
    value = edges.get("memory", edges.get("lpddr", "N"))
    return list(value) if isinstance(value, list) else [value]


def memory_balls_needed(board: Board) -> int:
    """Memory signal balls plus the ground after every lane."""
    signals = memory_signals(board)
    return len(memory_channels(board)) * (len(signals) + len(memory_lanes(signals))) if signals else 0


MISC_SIGNALS = [("mgmt", "SCLK"), ("mgmt", "MOSI"), ("mgmt", "MISO"), ("mgmt", "CS"), ("mgmt", "IRQ"),
                ("jtag", "TCK"), ("jtag", "TMS"), ("jtag", "TDI"), ("jtag", "TDO"), ("jtag", "TRST"),
                ("refclk", "CLK_P"), ("refclk", "CLK_N"), ("strap", "MODE0"), ("strap", "MODE1")]

LINK_ROWS = 18          # a 36-signal link port is two columns of this many rows


def misc_signals(board: Board) -> list[tuple[str, str]]:
    """The management signals plus any layer-die interface whose kind is
    marked ``misc`` (the shared PSRAM clocks of the modular board)."""
    extra = []
    for name, spec in board.classes["layer_asic"]["interfaces"].items():
        kind = board.kinds.get(spec["kind"], {})
        if kind.get("misc"):
            extra += [(name, signal) for signal in expand_signals(kind["signals"])]
    return MISC_SIGNALS + extra


def link_rows(board: Board) -> int:
    """A link port is a two-deep block of this many positions along its edge."""
    return math.ceil(len(expand_signals(board.kinds["link"]["signals"])) / 2)


# --------------------------------------------------------------------------
# Requirements and package choice
# --------------------------------------------------------------------------

def requirements(board: Board, rated_tokens_per_second: float | None = None) -> Requirements:
    rules = board.data["package_selection"]
    tps = float(rules["rated_tokens_per_second"] if rated_tokens_per_second is None else rated_tokens_per_second)
    amps = board.core_current_a("layer_asic", tps)
    link = len(expand_signals(board.kinds["link"]["signals"]))
    signal_balls = 2 * link + memory_signal_count(board) + len(misc_signals(board))
    signal_grounds = math.ceil(signal_balls / rules["signals_per_ground"])
    core_balls = math.ceil(amps / rules["amps_per_ball"])
    rails = dict(rules.get("rail_balls", {"VDD_IO_1V8": 8, "VDD_PLL_0V9": 2, "VDD2H_1V05": 12, "VDDQ_0V3": 12}))
    return Requirements(tps, amps, signal_balls, signal_grounds, core_balls, core_balls, rails)


def memory_channels(board: Board) -> list[str]:
    """On-board memory channels of a layer ASIC (empty when the memory sits in the package)."""
    return board.memory_interfaces("layer_asic")


def memory_signals(board: Board) -> list[str]:
    channels = memory_channels(board)
    if not channels:
        return []
    kind = board.classes["layer_asic"]["interfaces"][channels[0]]["kind"]
    return expand_signals(board.kinds[kind]["signals"])


def memory_signal_count(board: Board) -> int:
    return len(memory_channels(board)) * len(memory_signals(board))


def candidates(board: Board) -> list[PackageSpec]:
    return [PackageSpec(c["name"], int(c["cols"]), int(c["rows"]), float(c["pitch"]), float(c["body"]), float(c["ball"]))
            for c in board.data["package_selection"]["candidates"]]


def reject_reason(package: PackageSpec, need: Requirements, signal_rows: int) -> str | None:
    if package.balls < need.total:
        return f"{package.balls} balls, {need.total} needed"
    if package.outer_balls(signal_rows) < need.signal_balls + need.signal_grounds:
        return (f"outer {signal_rows} rows hold {package.outer_balls(signal_rows)} balls, "
                f"{need.signal_balls + need.signal_grounds} signal and return balls needed")
    if package.rows < LINK_ROWS + 4 or package.cols < LINK_ROWS + 4:
        return f"too few rows for an {LINK_ROWS}-row link port plus the memory rows"
    return None


def link_block_starts(board: Board, package: PackageSpec) -> dict[str, tuple[str, int]]:
    """(edge, first position along it) of the link_in and link_out blocks.  A
    block on its own edge is centred; two blocks on one edge sit either side
    of the middle, far enough apart that their card-edge finger groups (one
    position per signal at 1.0 mm) do not meet."""
    edges = board.data["package_selection"]["edges"]
    n = link_rows(board)
    signals = 2 * n
    starts = {}
    for port in ("link_in", "link_out"):
        edge = edges[port]
        along = package.rows if edge in ("W", "E") else package.cols
        if edges["link_in"] == edges["link_out"]:
            sep = math.ceil(((signals + 1) / package.pitch - n) / 2)
            mid = along // 2
            starts[port] = (edge, mid - n - sep if port == "link_in" else mid + 1 + sep)
        else:
            starts[port] = (edge, (along - n) // 2)
    return starts


def link_position(package: PackageSpec, edge: str, along: int, depth: int) -> tuple[int, int]:
    """Ball (row, column) of a link ball ``depth`` in from ``edge`` at ``along``."""
    if edge == "W":
        return along, depth
    if edge == "E":
        return along, package.cols - 1 - depth
    if edge == "N":
        return depth, along
    return package.rows - 1 - depth, along


def memory_capacity_reason(board: Board, package: PackageSpec, signal_rows: int) -> str | None:
    """With the memory on the north and south edges every memory ball must sit
    in those edges' outer rows; the single-edge map is allowed to spill."""
    edges = memory_edges(board)
    if edges == ["N"]:
        return None
    capacity = sum(signal_rows * package.cols if edge in ("N", "S") else signal_rows * (package.rows - 2 * signal_rows)
                   for edge in edges)
    needed = memory_balls_needed(board)
    if capacity < needed:
        return f"{len(edges)} memory edges of {signal_rows} rows hold {capacity} balls, {needed} memory balls needed"
    return None


# --------------------------------------------------------------------------
# Ball assignment
# --------------------------------------------------------------------------

def assign(board: Board, package: PackageSpec, need: Requirements) -> list[Ball]:
    rules = board.data["package_selection"]
    edges = rules["edges"]
    signal_rows = int(rules["signal_rows"])
    link = expand_signals(board.kinds["link"]["signals"])
    memory = memory_signals(board)
    mem_edges = memory_edges(board)
    taken: dict[tuple[int, int], Ball] = {}

    def place(i: int, j: int, kind: str, interface: str = "", signal: str = "", escape=None) -> None:
        if (i, j) in taken:
            raise ValueError(f"ball {package.ball_name(i, j)} assigned twice")
        x, y = package.ball_xy(i, j)
        taken[(i, j)] = Ball(package.ball_name(i, j), i, j, x, y, kind, interface, signal, escape)

    # Link ports: two-deep blocks along their edges.  Signal 2k on the outer
    # position, 2k+1 on the inner position of the same row or column, so the
    # PCB ribbon's lanes (outer via, dogbone via) carry consecutive signals.
    # A mirrored link-out port runs the same sequence from the other end.
    n_rows = link_rows(board)
    starts = link_block_starts(board, package)
    r0 = starts["link_in"][1] if starts["link_in"][0] in ("W", "E") else (package.rows - n_rows) // 2
    mirrored = bool(rules.get("link_out_mirrored", False))
    for k, signal in enumerate(link):
        m = len(link) - 1 - k if mirrored else k
        for port, index in (("link_in", k), ("link_out", m)):
            edge, start = starts[port]
            depth = index % 2
            i, j = link_position(package, edge, start + index // 2, depth)
            place(i, j, "signal", port, signal, (edge, depth))
    link_rows_taken = set(range(r0, r0 + n_rows)) if starts["link_in"][0] in ("W", "E") else set()

    # Memory channels.  On the north edge alone: full width above the link
    # rows, then the columns between the link ports, row by row.  On the north
    # and south edges: the outer rows of each edge in five-column strips, north
    # first, so every channel is a compact block and every ball is escapable.
    def north_positions():
        for i in range(package.rows):
            cols = range(package.cols) if i < r0 else range(2, package.cols - 2)
            for j in cols:
                if (i, j) not in taken:
                    yield i, j

    def strip_positions():
        # Four-wide strips through the outer rows of each memory edge in turn.
        for edge in mem_edges:
            if edge in ("N", "S"):
                rows = range(signal_rows) if edge == "N" else range(package.rows - signal_rows, package.rows)
                for j0 in range(0, package.cols, 4):
                    for j in range(j0, min(j0 + 4, package.cols)):
                        for i in rows:
                            if (i, j) not in taken:
                                yield i, j
            else:
                cols = range(signal_rows) if edge == "W" else range(package.cols - signal_rows, package.cols)
                for i0 in range(signal_rows, package.rows - signal_rows, 4):
                    for i in range(i0, min(i0 + 4, package.rows - signal_rows)):
                        for j in cols:
                            if (i, j) not in taken:
                                yield i, j
    positions = north_positions() if mem_edges == ["N"] else strip_positions()
    for channel in memory_channels(board):
        for lane in memory_lanes(memory):
            for signal in lane:
                i, j = next(positions)
                place(i, j, "signal", channel, signal)
            i, j = next(positions)
            place(i, j, "ground", "GND", "")

    # Small interfaces on the south row (between the link blocks when those
    # are on the south edge too), a ground after every four; or, when the
    # south rows hold memory, in the west columns outside the link port.
    if edges.get("misc", "S") == "W":
        memory_rows = set(range(signal_rows)) | set(range(package.rows - signal_rows, package.rows))
        misc = ((i, j) for j0 in (0, 2) for i in range(package.rows) for j in (j0, j0 + 1)
                if i not in memory_rows and i not in link_rows_taken and (i, j) not in taken)
    else:
        misc = ((i, j) for i in (package.rows - 1, package.rows - 2) for j in range(2, package.cols - 2)
                if (i, j) not in taken)
    try:
        for n, (interface, signal) in enumerate(misc_signals(board)):
            i, j = next(misc)
            place(i, j, "signal", interface, signal)
            if n % 4 == 3:
                i, j = next(misc)
                place(i, j, "ground", "GND", "")
    except StopIteration:
        raise ValueError(f"{package.name}: no room for the small interfaces on the {edges.get('misc', 'S')} edge "
                         "beside the link and memory balls") from None

    # Everything else: ground and core in a checkerboard, rails sprinkled in.
    rails = list(need.rail_balls.items())
    rail_index, rail_left = 0, rails[0][1] if rails else 0
    for i in range(package.rows):
        for j in range(package.cols):
            if (i, j) in taken:
                continue
            if (i + j) % 2 == 0:
                place(i, j, "ground", "GND", "")
            elif rails and rail_left > 0 and (i * 7 + j * 3) % 5 == 0:
                place(i, j, "rail", rails[rail_index][0], "")
                rail_left -= 1
                if rail_left == 0 and rail_index + 1 < len(rails):
                    rail_index += 1
                    rail_left = rails[rail_index][1]
            else:
                place(i, j, "rail", "VDD_CORE", "")
    return [taken[(i, j)] for i in range(package.rows) for j in range(package.cols)]


def derive(board: Board, rated_tokens_per_second: float | None = None) -> Pinout:
    """Pick the smallest candidate that meets the requirements and assign its balls."""
    need = requirements(board, rated_tokens_per_second)
    signal_rows = int(board.data["package_selection"]["signal_rows"])
    rejected = []
    for package in sorted(candidates(board), key=lambda p: (p.body, p.balls)):
        reason = reject_reason(package, need, signal_rows) or memory_capacity_reason(board, package, signal_rows)
        if reason is None:
            balls = assign(board, package, need)
            core = sum(1 for b in balls if b.kind == "rail" and b.interface == "VDD_CORE")
            ground = sum(1 for b in balls if b.kind == "ground")
            if core < need.core_balls:
                reason = f"only {core} core balls after signals and rails, {need.core_balls} needed"
            elif ground < need.ground_balls + need.signal_grounds:
                reason = f"only {ground} ground balls, {need.ground_balls + need.signal_grounds} needed"
            else:
                return Pinout(package, need, balls, rejected)
        rejected.append((package.name, reason))
    raise ValueError("no candidate package meets the requirements: " + "; ".join(f"{n}: {r}" for n, r in rejected))


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

def write_csv(pinout: Pinout, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["ball", "row", "col", "x_mm", "y_mm", "kind", "interface", "signal", "escape_edge", "escape_depth"])
        for b in pinout.balls:
            writer.writerow([b.name, b.row, b.col, f"{b.x:.3f}", f"{b.y:.3f}", b.kind, b.interface, b.signal,
                             b.escape[0] if b.escape else "", b.escape[1] if b.escape else ""])


def write_json(pinout: Pinout, path: Path) -> None:
    data = {"package": pinout.package.__dict__, "requirements": pinout.requirements.__dict__,
            "rejected": pinout.rejected,
            "balls": [{"name": b.name, "row": b.row, "col": b.col, "x": b.x, "y": b.y, "kind": b.kind,
                       "interface": b.interface, "signal": b.signal, "escape": b.escape} for b in pinout.balls]}
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def report_markdown(pinout: Pinout, board: Board) -> str:
    p, need = pinout.package, pinout.requirements
    rules = board.data["package_selection"]
    lines = ["# ASIC package and ball map", "",
             f"Derived by `python -m hw.pinout` from `{board.source.as_posix()}`. Do not edit.", "",
             "## Requirements", "",
             f"Rated at {need.rated_tokens_per_second:.0f} tokens/s the core rail draws {need.core_amps:.0f} A "
             f"({board.data['power_model']['mac_energy_pj']} pJ per MAC, {board.data['power_tree']['rails']['VDD_CORE']['volts']} V), "
             f"one ball per {rules['amps_per_ball']} A.", "",
             "| Need | Balls |", "| --- | ---: |",
             f"| Signals ({4 * link_rows(board)} link, {memory_signal_count(board)} memory, {len(misc_signals(board))} management) | {need.signal_balls} |",
             f"| Signal ground returns (1 per {rules['signals_per_ground']}) | {need.signal_grounds} |",
             f"| Core rail | {need.core_balls} |", f"| Ground for the core | {need.ground_balls} |"]
    for rail, count in need.rail_balls.items():
        lines.append(f"| {rail} | {count} |")
    lines += [f"| **Total** | **{need.total}** |", "", "## Candidates", "", "| Package | Balls | Outer rows | Body | Verdict |",
              "| --- | ---: | ---: | ---: | --- |"]
    for c in sorted(candidates(board), key=lambda x: (x.body, x.balls)):
        verdict = "**selected**" if c.name == p.name else next((r for n, r in pinout.rejected if n == c.name), "not needed")
        lines.append(f"| {c.name} | {c.balls} | {c.outer_balls(int(rules['signal_rows']))} | {c.body:.0f} mm | {verdict} |")
    lines += ["", "## Ball map", "", f"{p.name}: {p.cols} x {p.rows} at {p.pitch} mm, {p.body:.0f} mm body.", "",
              "| Use | Balls | Where |", "| --- | ---: | --- |",
              f"| link_in | {pinout.count('signal', 'link_in')} | {rules['edges']['link_in']} edge, a two-deep block of "
              f"{link_rows(board)} positions |",
              f"| link_out | {pinout.count('signal', 'link_out')} | {rules['edges']['link_out']} edge"
              f"{', beside it' if rules['edges']['link_out'] == rules['edges']['link_in'] else ', the same positions'} |",
              *([f"| {len(memory_channels(board))} memory channels | {sum(pinout.count('signal', c) for c in memory_channels(board))} | "
                 f"{' and '.join(memory_edges(board))} rows, lanes with a ground after each |"] if memory_channels(board)
                else ["| memory | 0 | in the package (HBM on the interposer), no balls |"]),
              f"| mgmt, jtag, refclk, strap | {sum(pinout.count('signal', i) for i in ('mgmt', 'jtag', 'refclk', 'strap'))} | "
              f"{rules['edges']['misc']} {'columns outside the link port' if rules['edges']['misc'] == 'W' else 'row'} |",
              f"| VDD_CORE | {pinout.count('rail', 'VDD_CORE')} | interior checkerboard |",
              f"| GND | {pinout.count('ground')} | interior checkerboard and lane returns |"]
    for rail in need.rail_balls:
        lines.append(f"| {rail} | {pinout.count('rail', rail)} | interior |")
    outer = sum(1 for b in pinout.balls if b.kind == "signal" and
                min(b.row, b.col, p.rows - 1 - b.row, p.cols - 1 - b.col) >= int(rules["signal_rows"]))
    lines += ["", f"Signal balls deeper than the outer {rules['signal_rows']} rows: {outer} "
              "(these need a microvia or build-up escape on the PCB).", "",
              "## What the packaging house gets", "",
              "* this map as `asic_ballmap.csv`, with the edge each interface must face;",
              f"* the die-edge assignment it implies: link ports on the {rules['edges']['link_in']} and {rules['edges']['link_out']} die edges, "
              + (f"the memory PHYs on the {' and '.join(memory_edges(board))} edge{'s' if len(memory_edges(board)) > 1 else ''}, "
                 if memory_channels(board) else "the HBM PHY on the north edge towards the stack, ")
              + ("management on the west outside the link port;" if rules["edges"].get("misc") == "W" else "management on the south;"),
              f"* the core current ({need.core_amps:.0f} A at the rating, {board.core_current_a('layer_asic', 50_000):.0f} A "
              "at 50K tokens/s) for the bump map and the substrate power planes.", "",
              "The substrate design, the bump map and the final ball map come back from them; the loop usually runs "
              "two or three times and this file is regenerated from the agreed rules each round."]
    return "\n".join(lines) + "\n"


def write_outputs(pinout: Pinout, board: Board, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_csv(pinout, output / "asic_ballmap.csv")
    write_json(pinout, output / "asic_ballmap.json")
    (output / "report.md").write_text(report_markdown(pinout, board), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "board.yaml")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--tokens-per-second", type=float, default=None, help="rating override")
    parser.add_argument("--check", action="store_true", help="print the report without writing files")
    args = parser.parse_args()
    board = Board.load(args.source)
    pinout = derive(board, args.tokens_per_second)
    if args.check:
        print(report_markdown(pinout, board))
        return
    write_outputs(pinout, board, args.output)
    need = pinout.requirements
    print(f"{pinout.package.name}: {need.total} balls needed of {pinout.package.balls} "
          f"({need.core_amps:.0f} A core at {need.rated_tokens_per_second:.0f} tokens/s); wrote {args.output}/")
    for name, reason in pinout.rejected:
        print(f"  rejected {name}: {reason}")


if __name__ == "__main__":
    main()
