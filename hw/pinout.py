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
  columns of rows, both memory channels on the north rows in byte lanes with
  a ground between lanes, the small interfaces on the south row;
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

ROW_LETTERS = list("ABCDEFGHJKLMNPRTUVWY") + ["A" + c for c in "ABCDEFGHJKLMNPRTUVWY"]


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


MISC_SIGNALS = [("mgmt", "SCLK"), ("mgmt", "MOSI"), ("mgmt", "MISO"), ("mgmt", "CS"), ("mgmt", "IRQ"),
                ("jtag", "TCK"), ("jtag", "TMS"), ("jtag", "TDI"), ("jtag", "TDO"), ("jtag", "TRST"),
                ("refclk", "CLK_P"), ("refclk", "CLK_N"), ("strap", "MODE0"), ("strap", "MODE1")]

LINK_ROWS = 18          # a link port is two columns of this many rows


# --------------------------------------------------------------------------
# Requirements and package choice
# --------------------------------------------------------------------------

def requirements(board: Board, rated_tokens_per_second: float | None = None) -> Requirements:
    rules = board.data["package_selection"]
    tps = float(rules["rated_tokens_per_second"] if rated_tokens_per_second is None else rated_tokens_per_second)
    amps = board.core_current_a("layer_asic", tps)
    link = len(expand_signals(board.kinds["link"]["signals"]))
    lpddr = len(expand_signals(board.kinds["lpddr5x_x32"]["signals"]))
    signal_balls = 2 * link + 2 * lpddr + len(MISC_SIGNALS)
    signal_grounds = math.ceil(signal_balls / rules["signals_per_ground"])
    core_balls = math.ceil(amps / rules["amps_per_ball"])
    rails = {"VDD_IO_1V8": 8, "VDD_PLL_0V9": 2, "VDD2H_1V05": 12, "VDDQ_0V3": 12}
    return Requirements(tps, amps, signal_balls, signal_grounds, core_balls, core_balls, rails)


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


# --------------------------------------------------------------------------
# Ball assignment
# --------------------------------------------------------------------------

def assign(board: Board, package: PackageSpec, need: Requirements) -> list[Ball]:
    rules = board.data["package_selection"]
    edges = rules["edges"]
    link = expand_signals(board.kinds["link"]["signals"])
    lpddr = expand_signals(board.kinds["lpddr5x_x32"]["signals"])
    taken: dict[tuple[int, int], Ball] = {}

    def place(i: int, j: int, kind: str, interface: str = "", signal: str = "", escape=None) -> None:
        if (i, j) in taken:
            raise ValueError(f"ball {package.ball_name(i, j)} assigned twice")
        x, y = package.ball_xy(i, j)
        taken[(i, j)] = Ball(package.ball_name(i, j), i, j, x, y, kind, interface, signal, escape)

    # Link ports: two columns of LINK_ROWS rows, centred on the edge.  Signal
    # 2k on the outer column, 2k+1 on the inner column of the same row, so
    # the PCB ribbon's lanes (outer via, dogbone via) carry consecutive signals.
    r0 = (package.rows - LINK_ROWS) // 2
    for k, signal in enumerate(link):
        i = r0 + k // 2
        depth = k % 2
        if edges["link_in"] == "W":
            place(i, depth, "signal", "link_in", signal, ("W", depth))
            place(i, package.cols - 1 - depth, "signal", "link_out", signal, ("E", depth))
        else:
            place(i, package.cols - 1 - depth, "signal", "link_in", signal, ("E", depth))
            place(i, depth, "signal", "link_out", signal, ("W", depth))

    # Memory channels on the north rows: full width above the link rows, then
    # the columns between the link ports; byte lanes with a ground after each.
    def north_positions():
        for i in range(package.rows):
            cols = range(package.cols) if i < r0 else range(2, package.cols - 2)
            for j in cols:
                if (i, j) not in taken:
                    yield i, j
    north = north_positions()
    for channel in ("lpddr_ch0", "lpddr_ch1"):
        for lane in lpddr_lanes(lpddr):
            for signal in lane:
                i, j = next(north)
                place(i, j, "signal", channel, signal)
            i, j = next(north)
            place(i, j, "ground", "GND", "")

    # Small interfaces on the south row, a ground after every four.
    south = ((package.rows - 1, j) for j in range(2, package.cols - 2))
    for n, (interface, signal) in enumerate(MISC_SIGNALS):
        i, j = next(south)
        place(i, j, "signal", interface, signal)
        if n % 4 == 3:
            i, j = next(south)
            place(i, j, "ground", "GND", "")

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
        reason = reject_reason(package, need, signal_rows)
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
             "Derived by `python -m hw.pinout` from `hw/board.yaml`. Do not edit.", "",
             "## Requirements", "",
             f"Rated at {need.rated_tokens_per_second:.0f} tokens/s the core rail draws {need.core_amps:.0f} A "
             f"({board.data['power_model']['mac_energy_pj']} pJ per MAC, {board.data['power_tree']['rails']['VDD_CORE']['volts']} V), "
             f"one ball per {rules['amps_per_ball']} A.", "",
             "| Need | Balls |", "| --- | ---: |",
             f"| Signals ({2 * 36} link, {2 * 65} memory, {len(MISC_SIGNALS)} management) | {need.signal_balls} |",
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
              f"| link_in | {pinout.count('signal', 'link_in')} | {rules['edges']['link_in']} edge, columns 1-2, rows "
              f"{(p.rows - LINK_ROWS) // 2 + 1}-{(p.rows - LINK_ROWS) // 2 + LINK_ROWS} |",
              f"| link_out | {pinout.count('signal', 'link_out')} | {rules['edges']['link_out']} edge, same rows |",
              f"| lpddr_ch0, lpddr_ch1 | {pinout.count('signal', 'lpddr_ch0') + pinout.count('signal', 'lpddr_ch1')} | "
              f"{rules['edges']['lpddr']} rows, byte lanes with a ground after each |",
              f"| mgmt, jtag, refclk, strap | {sum(pinout.count('signal', i) for i in ('mgmt', 'jtag', 'refclk', 'strap'))} | "
              f"{rules['edges']['misc']} row |",
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
              "* the die-edge assignment it implies: link ports on the west and east die edges, both LPDDR5X PHYs "
              "on the north edge, management on the south;",
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
