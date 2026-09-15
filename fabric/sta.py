"""Static timing of a mapped fabric netlist with OpenSTA.

Takes the netlist written by ``fabric.synth --netlist`` and the liberty it was
mapped to, constrains a clock, and reports the worst setup path and slack.

    python -m fabric.sta --sta /path/to/sta --liberty cells.lib \
        --netlist netlist.v --period-ps 2500

This is pre-layout timing: no wire load, no clock tree, one corner.  Treat
the result as the logic-depth floor of the clock period, and budget 20 to 40
percent on top for wires.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class StaResult:
    period_ps: float
    worst_slack_ps: float
    critical_path_ps: float         # data arrival at the worst endpoint minus launch clock edge
    startpoint: str
    endpoint: str
    max_frequency_mhz: float        # 1 / (period - slack), i.e. what this logic depth allows
    report: str

    def as_dict(self) -> dict:
        return asdict(self)


def run_sta(sta: Path, liberties: Sequence[Path], netlist: Path, *, top: str = "fabric_columns",
            period_ps: float = 1000.0, clock: str = "clk", io_delay_frac: float = 0.1) -> StaResult:
    liberties = [Path(p).resolve() for p in liberties]
    netlist = Path(netlist).resolve()
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        # OpenSTA's Verilog reader does not accept `wire signed`; yosys emits it for signed nets.
        cleaned = work / "netlist.v"
        cleaned.write_text(netlist.read_text(encoding="utf-8").replace("wire signed ", "wire "), encoding="utf-8")
        netlist = cleaned
        # The SDC is read after set_cmd_units, so these values are picoseconds.
        io_delay = period_ps * io_delay_frac
        sdc = "\n".join([
            f"create_clock -name {clock} -period {period_ps:.1f} [get_ports {clock}]",
            f"set_input_delay -clock {clock} {io_delay:.1f} [delete_from_list [all_inputs] [get_ports {clock}]]",
            f"set_output_delay -clock {clock} {io_delay:.1f} [all_outputs]",
        ]) + "\n"
        (work / "design.sdc").write_text(sdc, encoding="utf-8")
        tcl = "\n".join(
            [f"read_liberty {path}" for path in liberties]
            + [f"read_verilog {netlist}",
               f"link_design {top}",
               "set_cmd_units -time ps -capacitance fF",
               "read_sdc design.sdc",
               f"report_checks -path_delay max -path_group {clock} -format full_clock_expanded "
               "-fields {slew cap fanout} -digits 3",
               f"report_checks -path_delay max -path_group {clock} -group_path_count 5 -format end",
               "report_wns",
               "report_tns",
               "exit"]) + "\n"
        (work / "run.tcl").write_text(tcl, encoding="utf-8")
        result = subprocess.run([str(sta), "-no_init", "-exit", "run.tcl"], cwd=work, capture_output=True, text=True)
        report = result.stdout + result.stderr
    if result.returncode != 0 and "slack" not in report:
        raise RuntimeError(f"OpenSTA failed:\n{report[-4000:]}")
    return parse_report(report, period_ps)


def parse_report(report: str, period_ps: float, clock: str = "clk") -> StaResult:
    """Read the worst path of the clock's path group (not the asynchronous recovery group)."""
    group = report.find(f"Path Group: {clock}")
    if group >= 0:
        head = report.rfind("Startpoint:", 0, group)
        report = report[head if head >= 0 else group:]
    slack_match = re.search(r"([-0-9.]+)\s+slack\s+\((MET|VIOLATED)\)", report)
    if not slack_match:
        raise ValueError(f"no slack in OpenSTA report:\n{report[-3000:]}")
    slack_ps = float(slack_match.group(1))       # reports are in ps (set_cmd_units)
    start = re.search(r"Startpoint:\s+(\S+)", report)
    end = re.search(r"Endpoint:\s+(\S+)", report)
    arrival = re.search(r"([-0-9.]+)\s+data arrival time", report)
    critical_ps = float(arrival.group(1)) if arrival else float("nan")
    achievable_ps = period_ps - slack_ps
    return StaResult(period_ps, slack_ps, critical_ps, start.group(1) if start else "?",
                     end.group(1) if end else "?", 1e6 / achievable_ps if achievable_ps > 0 else float("inf"),
                     report[-6000:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sta", type=Path, default=shutil.which("sta"), help="OpenSTA executable")
    parser.add_argument("--liberty", type=Path, required=True, action="append")
    parser.add_argument("--netlist", type=Path, required=True)
    parser.add_argument("--top", default="fabric_columns")
    parser.add_argument("--period-ps", type=float, default=1000.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report", action="store_true", help="print the OpenSTA path report")
    args = parser.parse_args()
    if not args.sta:
        raise SystemExit("OpenSTA not found; pass --sta")
    result = run_sta(args.sta, args.liberty, args.netlist, top=args.top, period_ps=args.period_ps)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(f"period {result.period_ps:.0f} ps: worst slack {result.worst_slack_ps:+.0f} ps, "
              f"critical path {result.critical_path_ps:.0f} ps ({result.startpoint} -> {result.endpoint}), "
              f"logic-depth limit {result.max_frequency_mhz:.0f} MHz")
    if args.report:
        print(result.report)


if __name__ == "__main__":
    main()
