"""Place and route the fabric column datapath with OpenROAD on an open platform.

Drives OpenROAD directly with the platform files from OpenROAD-flow-scripts
(sky130hd or asap7): floorplan, pin placement, timing-driven global
placement, resizing and buffering, clock-tree synthesis, global routing,
parasitic estimation, timing repair, and a final timing report.  Detailed
routing is optional (``--detailed-route``); global-routing parasitics are
what the timing numbers use either way.

    python -m fabric.pnr --openroad /path/openroad --platform sky130hd \
        --platforms-dir /path/OpenROAD-flow-scripts/flow/platforms \
        --netlist net_sky130.v --liberty sky130.lib --period-ps 3000

The netlist comes from ``fabric.synth --netlist``; pass the same liberty (or
the platform's own) so cell names match.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Platform:
    name: str
    tech_lef: str
    cell_lefs: tuple[str, ...]
    lib: str
    site: str
    pin_layer_h: str
    pin_layer_v: str
    min_route_layer: str
    max_route_layer: str
    min_clock_layer: str
    route_adjustment: float
    cts_buffer: str
    tracks_script: str
    rc_script: str
    tie_hi: str            # "cell port"
    tie_lo: str
    core_space_um: float
    time_unit_ps: float    # liberty time unit in ps (sky130: ns -> 1000, asap7: ps -> 1)


PLATFORMS = {
    "sky130hd": Platform(
        name="sky130hd",
        tech_lef="sky130hd/lef/sky130_fd_sc_hd.tlef",
        cell_lefs=("sky130hd/lef/sky130_fd_sc_hd_merged.lef",),
        lib="sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib",
        site="unithd",
        pin_layer_h="met3", pin_layer_v="met2",
        min_route_layer="met1", max_route_layer="met5", min_clock_layer="met3", route_adjustment=0.2,
        cts_buffer="sky130_fd_sc_hd__clkbuf_4",
        tracks_script="sky130hd/make_tracks.tcl",
        rc_script="sky130hd/setRC.tcl",
        tie_hi="sky130_fd_sc_hd__conb_1/HI", tie_lo="sky130_fd_sc_hd__conb_1/LO",
        core_space_um=10.0,
        time_unit_ps=1000.0,
    ),
    "asap7": Platform(
        name="asap7",
        tech_lef="asap7/lef/asap7_tech_1x_201209.lef",
        cell_lefs=("asap7/lef/asap7sc7p5t_28_R_1x_220121a.lef",),
        lib="",   # ASAP7 splits its liberty; pass the merged file with --liberty
        site="asap7sc7p5t",
        pin_layer_h="M4", pin_layer_v="M5",
        min_route_layer="M2", max_route_layer="M7", min_clock_layer="M4", route_adjustment=0.5,
        cts_buffer="BUFx4_ASAP7_75t_R",
        tracks_script="asap7/openRoad/make_tracks.tcl",
        rc_script="asap7/setRC.tcl",
        tie_hi="TIEHIx1_ASAP7_75t_R/H", tie_lo="TIELOx1_ASAP7_75t_R/L",
        core_space_um=2.0,
        time_unit_ps=1.0,
    ),
}


@dataclass
class PnrResult:
    platform: str
    period_ps: float
    worst_slack_ps: float
    tns_ps: float
    critical_path_ps: float
    max_frequency_mhz: float
    design_area_um2: float
    utilization_pct: float
    instances: int
    clock_skew_ps: float | None
    wirelength_um: float        # global-route total wirelength
    overflow: int               # global-route total overflow (0 = routable)
    stage: str                  # which timing report the numbers come from
    report: str

    def as_dict(self) -> dict:
        return asdict(self)


def write_flow(work: Path, platform: Platform, platforms_dir: Path, netlist: Path, liberties: Sequence[Path],
               *, top: str, period_ps: float, utilization: float, detailed_route: bool, threads: int) -> Path:
    p = platforms_dir.resolve()
    lef_reads = [f"read_lef {p / platform.tech_lef}"] + [f"read_lef {p / lef}" for lef in platform.cell_lefs]
    lib_reads = [f"read_liberty {Path(l).resolve()}" for l in liberties]
    period_lib = period_ps / platform.time_unit_ps     # SDC in library time units
    io_delay = period_lib * 0.1
    sdc = "\n".join([
        f"create_clock -name clk -period {period_lib:.4f} [get_ports clk]",
        f"set_input_delay -clock clk {io_delay:.4f} [delete_from_list [all_inputs] [get_ports clk]]",
        f"set_output_delay -clock clk {io_delay:.4f} [all_outputs]",
        "set_max_fanout 20 [current_design]",
    ]) + "\n"
    (work / "design.sdc").write_text(sdc, encoding="utf-8")
    # Equivalent of the flow scripts' fastroute.tcl, without its environment variables.
    fastroute = "\n".join([
        f"set_global_routing_layer_adjustment {platform.min_route_layer}-{platform.max_route_layer} "
        f"{platform.route_adjustment}",
        f"set_routing_layers -clock {platform.min_clock_layer}-{platform.max_route_layer}",
        f"set_routing_layers -signal {platform.min_route_layer}-{platform.max_route_layer}",
    ])
    detailed = "\n".join([
        "detailed_route -output_drc route_drc.rpt -verbose 0",
        "estimate_parasitics -global_routing",
        "report_checks -path_delay max -path_group clk -format full_clock_expanded -fields {slew cap fanout} -digits 3",
        "report_wns", "report_tns",
    ]) if detailed_route else ""
    tcl = "\n".join(lef_reads + lib_reads + [
        f"read_verilog {netlist.resolve()}",
        f"link_design {top}",
        "read_sdc design.sdc",
        f"source {p / platform.rc_script}",
        # Floorplan.
        f"initialize_floorplan -utilization {utilization} -aspect_ratio 1.0 "
        f"-core_space {platform.core_space_um} -site {platform.site}",
        f"source {p / platform.tracks_script}",
        f"place_pins -hor_layers {platform.pin_layer_h} -ver_layers {platform.pin_layer_v}",
        fastroute,
        # Placement with timing-driven global placement, then resize and repair.
        "global_placement -timing_driven -density 0.65",
        "estimate_parasitics -placement",
        "repair_design",
        f"repair_tie_fanout {platform.tie_hi}",
        f"repair_tie_fanout {platform.tie_lo}",
        "detailed_placement",
        "estimate_parasitics -placement",
        "repair_timing -setup",
        "detailed_placement",
        # Clock tree.
        f"clock_tree_synthesis -buf_list {{{platform.cts_buffer}}} -root_buf {platform.cts_buffer} -sink_clustering_enable",
        "set_propagated_clock [all_clocks]",
        "repair_clock_nets",
        "detailed_placement",
        "estimate_parasitics -placement",
        "repair_timing -setup -hold",
        "detailed_placement",
        "check_placement",
        # Global routing and post-route timing.
        # A 16-column slice is pin-heavy for its area; allow overflow so the
        # timing estimate still completes and report the congestion separately.
        # (`report_wire_length` is not used: it crashes the litex-hub build; the
        # router's own "Total wirelength" line carries the same number.)
        "global_route -congestion_iterations 50 -allow_congestion -verbose",
        "estimate_parasitics -global_routing",
        "repair_timing -setup",
        # All reports go to the log; older builds ignore `> file` on some of them.
        "report_checks -path_delay max -path_group clk -format full_clock_expanded -fields {slew cap fanout} -digits 3",
        "report_clock_skew",
        "report_design_area",
        "report_wns",
        "report_tns",
        detailed,
        "write_def design.def",
        "exit",
    ]) + "\n"
    script = work / "flow.tcl"
    script.write_text(tcl, encoding="utf-8")
    return script


def parse_results(work: Path, platform: Platform, period_ps: float, detailed_route: bool) -> PnrResult:
    """Read the final reports.  Older OpenROAD builds ignore `> file` redirects on
    some report commands, so everything is taken from the log, last occurrence
    wins (the post-route reports come last)."""
    unit = platform.time_unit_ps
    log = (work / "openroad.log").read_text(encoding="utf-8") if (work / "openroad.log").exists() else ""
    stage = "detailed_route" if detailed_route and "detailed_route" in log and "route_drc" in log else "global_route"
    wns_all = re.findall(r"wns\s+max\s+([-0-9.]+)", log)
    tns_all = re.findall(r"tns\s+max\s+([-0-9.]+)", log)
    if not wns_all:
        raise ValueError(f"no wns in OpenROAD log {work / 'openroad.log'}")
    wns_val = float(wns_all[-1]) * unit
    tns_val = float(tns_all[-1]) * unit if tns_all else float("nan")
    paths = [m.start() for m in re.finditer(r"Startpoint:", log)]
    timing = log[paths[-1]:] if paths else ""
    timing = timing[: timing.find("slack (") + 20] if "slack (" in timing else timing
    arrival = re.search(r"([-0-9.]+)\s+data arrival time", timing)
    critical = float(arrival.group(1)) * unit if arrival else float("nan")
    area_all = re.findall(r"Design area\s+([0-9.]+)\s+u\^2\s+([0-9.]+)%\s+utilization", log)
    design_area = float(area_all[-1][0]) if area_all else float("nan")
    util = float(area_all[-1][1]) if area_all else float("nan")
    skew_all = re.findall(r"([-0-9.]+)\s+skew", log)
    skew_val = float(skew_all[-1]) * unit if skew_all else None
    inst = re.findall(r"Instance count:\s+(\d+)", log)
    instances = int(inst[-1]) if inst else 0
    wl = re.findall(r"Total wirelength:\s+(\d+)\s+um", log)
    wirelength = float(wl[-1]) if wl else float("nan")
    # Last line of the final congestion table: "Total  resource demand usage% H / V / Total".
    ovf = re.findall(r"^Total\s+\d+\s+\d+\s+[0-9.]+%\s+\d+\s+/\s+\d+\s+/\s+(\d+)", log, re.MULTILINE)
    overflow = int(ovf[-1]) if ovf else -1
    achievable = period_ps - wns_val
    return PnrResult(platform.name, period_ps, wns_val, tns_val, critical, 1e6 / achievable if achievable > 0 else float("inf"),
                     design_area, util, instances, skew_val, wirelength, overflow, stage, timing[-5000:])


def run_pnr(openroad: Path, platform: Platform, platforms_dir: Path, netlist: Path, liberties: Sequence[Path], work: Path,
            *, top: str = "fabric_columns", period_ps: float = 3000.0, utilization: float = 45.0,
            detailed_route: bool = False, threads: int = 4) -> PnrResult:
    work.mkdir(parents=True, exist_ok=True)
    # OpenROAD's Verilog reader rejects `wire signed`; yosys emits it for signed nets.
    cleaned = work / "netlist.v"
    cleaned.write_text(Path(netlist).read_text(encoding="utf-8").replace("wire signed ", "wire "), encoding="utf-8")
    script = write_flow(work, platform, platforms_dir, cleaned, liberties, top=top, period_ps=period_ps,
                        utilization=utilization, detailed_route=detailed_route, threads=threads)
    with (work / "openroad.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([str(openroad), "-exit", "-no_init", "-threads", str(threads), str(script)],
                                cwd=work, stdout=log, stderr=subprocess.STDOUT, text=True)
    log_text = (work / "openroad.log").read_text(encoding="utf-8")
    if result.returncode != 0 or "wns max" not in log_text:
        raise RuntimeError(f"OpenROAD failed (see {work / 'openroad.log'}):\n{log_text[-4000:]}")
    return parse_results(work, platform, period_ps, detailed_route)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openroad", type=Path, default=shutil.which("openroad"))
    parser.add_argument("--platform", choices=sorted(PLATFORMS), required=True)
    parser.add_argument("--platforms-dir", type=Path, required=True, help="OpenROAD-flow-scripts/flow/platforms")
    parser.add_argument("--netlist", type=Path, required=True)
    parser.add_argument("--liberty", type=Path, action="append", default=None,
                        help="liberty file(s); defaults to the platform's typical corner")
    parser.add_argument("--top", default="fabric_columns")
    parser.add_argument("--period-ps", type=float, default=3000.0)
    parser.add_argument("--utilization", type=float, default=45.0)
    parser.add_argument("--detailed-route", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--work", type=Path, required=True, help="working directory for reports and DEF")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if not args.openroad:
        raise SystemExit("openroad not found; pass --openroad")
    platform = PLATFORMS[args.platform]
    liberties = args.liberty or ([args.platforms_dir / platform.lib] if platform.lib else [])
    if not liberties:
        raise SystemExit("this platform needs --liberty (merged ASAP7 liberty)")
    result = run_pnr(args.openroad, platform, args.platforms_dir, args.netlist, liberties, args.work, top=args.top,
                     period_ps=args.period_ps, utilization=args.utilization, detailed_route=args.detailed_route,
                     threads=args.threads)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        skew = f", clock skew {result.clock_skew_ps:.0f} ps" if result.clock_skew_ps is not None else ""
        print(f"{result.platform} after {result.stage}: period {result.period_ps:.0f} ps, worst slack "
              f"{result.worst_slack_ps:+.0f} ps, critical path {result.critical_path_ps:.0f} ps, "
              f"achievable {result.max_frequency_mhz:.0f} MHz, design area {result.design_area_um2:.0f} um2 "
              f"({result.utilization_pct:.0f}% utilization), {result.instances} instances{skew}, "
              f"wirelength {result.wirelength_um:.0f} um, overflow {result.overflow}")
    if args.report:
        print(result.report)


if __name__ == "__main__":
    main()
