"""Place and route the fabric column datapath with OpenROAD on an open platform.

Drives OpenROAD directly with the platform files from OpenROAD-flow-scripts
(sky130hd or asap7): floorplan, pin placement, well taps and the platform
power grid, timing-driven global placement, resizing and buffering,
clock-tree synthesis, global routing, parasitic estimation, timing repair,
and a timing report.  With ``--detailed-route`` it continues through
detailed routing, fill, OpenRCX extraction, timing on the extracted
parasitics, a power report, and static IR-drop analysis of the grid.

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
    pdn_script: str        # platform power-grid strategy (sourced before pdngen)
    tap_cell: str
    tap_distance_um: float
    rcx_rules: str         # OpenRCX extraction rules for post-detailed-route parasitics
    fill_cells: tuple[str, ...]
    vdd_volts: float


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
        pdn_script="sky130hd/pdn.tcl",
        tap_cell="sky130_fd_sc_hd__tapvpwrvgnd_1", tap_distance_um=14.0,
        rcx_rules="sky130hd/rcx_patterns.rules",
        fill_cells=("sky130_fd_sc_hd__fill_1", "sky130_fd_sc_hd__fill_2",
                    "sky130_fd_sc_hd__fill_4", "sky130_fd_sc_hd__fill_8"),
        vdd_volts=1.8,
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
        pdn_script="asap7/openRoad/pdn/grid_strategy-M1-M2-M5-M6.tcl",
        tap_cell="TAPCELL_ASAP7_75t_R", tap_distance_um=25.0,
        rcx_rules="asap7/rcx_patterns.rules",
        fill_cells=("FILLERxp5_ASAP7_75t_R", "FILLER_ASAP7_75t_R", "DECAPx1_ASAP7_75t_R",
                    "DECAPx2_ASAP7_75t_R", "DECAPx4_ASAP7_75t_R", "DECAPx6_ASAP7_75t_R",
                    "DECAPx10_ASAP7_75t_R"),
        vdd_volts=0.70,
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
    drc_violations: int | None  # after detailed route, else None
    power_w: float | None       # report_power total (default activity), after detailed route
    ir_drop_vdd_mv: float | None  # worst static IR drop on VDD from the generated grid
    ir_drop_vss_mv: float | None
    stage: str                  # which timing report the numbers come from
    report: str

    def as_dict(self) -> dict:
        return asdict(self)


def filter_pdn_script(text: str) -> str:
    """Keep the global connections, voltage domain and standard-cell grid of a
    platform PDN strategy; drop the macro grids (there are none here) and the
    `global_connect` call that older builds lack (their pdngen connects the
    supplies itself from the add_global_connection patterns)."""
    kept = []
    for line in text.splitlines():
        if "macro grids" in line:
            break
        if line.strip() == "global_connect":
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


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
    (work / "pdn.tcl").write_text(filter_pdn_script((p / platform.pdn_script).read_text(encoding="utf-8")),
                                  encoding="utf-8")
    # Equivalent of the flow scripts' fastroute.tcl, without its environment variables.
    fastroute = "\n".join([
        f"set_global_routing_layer_adjustment {platform.min_route_layer}-{platform.max_route_layer} "
        f"{platform.route_adjustment}",
        f"set_routing_layers -clock {platform.min_clock_layer}-{platform.max_route_layer}",
        f"set_routing_layers -signal {platform.min_route_layer}-{platform.max_route_layer}",
    ])
    reports = [
        "report_checks -path_delay max -path_group clk -format full_clock_expanded -fields {slew cap fanout} -digits 3",
        "report_clock_skew",
        "report_design_area",
        "report_wns",
        "report_tns",
    ]
    # Detailed route, fill, signoff-style extraction with OpenRCX, timing on the
    # extracted parasitics, power, and static IR drop on the generated grid.
    # The IR-drop and antenna steps are wrapped in catch so a build that lacks
    # them still produces the timing numbers.
    detailed = "\n".join([
        "detailed_route -output_drc route_drc.rpt -output_maze maze.log -droute_end_iter 64 -verbose 1",
        "catch {check_antennas -report_file antennas.rpt} msg; puts $msg",
        f"filler_placement {{{' '.join(platform.fill_cells)}}}",
        "check_placement",
        "define_process_corner -ext_model_index 0 X",
        f"set_extraction_rules_file {p / platform.rcx_rules}",
        "extract_parasitics",
        "write_spef design.spef",
        "read_spef design.spef",
        "puts {--- timing on extracted parasitics ---}",
        *reports,
        "report_power",
        f"set_pdnsim_net_voltage -net VDD -voltage {platform.vdd_volts}",
        "catch {analyze_power_grid -net VDD -outfile ir_vdd.rpt} msg; puts $msg",
        "set_pdnsim_net_voltage -net VSS -voltage 0.0",
        "catch {analyze_power_grid -net VSS -outfile ir_vss.rpt} msg; puts $msg",
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
        # Well taps and the power grid from the platform's strategy, before
        # placement so the stripes are routing blockages from the start.
        f"tapcell -distance {platform.tap_distance_um} -tapcell_master {platform.tap_cell}",
        "source pdn.tcl",
        "pdngen",
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
        "puts {--- timing on global-route parasitics ---}",
        *reports,
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
    drc = re.findall(r"Number of violations\s*=\s*(\d+)", log)
    stage = "detailed_route" if detailed_route and drc else "global_route"
    drc_count = int(drc[-1]) if drc else None
    # report_power's Total row: internal, switching, leakage, total, percent.
    power = re.findall(r"^Total\s+([0-9.e+-]+)\s+([0-9.e+-]+)\s+([0-9.e+-]+)\s+([0-9.e+-]+)\s+100", log, re.MULTILINE)
    power_w = float(power[-1][3]) if power else None

    # pdnsim prints "Worstcase IR drop: X V" once per analyzed net, VDD first then VSS.
    drops = re.findall(r"Worst[- ]?case IR drop\s*:?\s*([0-9.e+-]+)\s*V", log, re.IGNORECASE)
    ir_vdd = float(drops[0]) * 1000.0 if drops else None
    ir_vss = float(drops[1]) * 1000.0 if len(drops) > 1 else None
    # `report_wns` prints "wns max X" in current builds and "wns X" in older ones.
    wns_all = re.findall(r"^wns(?:\s+max)?\s+([-0-9.]+)", log, re.MULTILINE)
    tns_all = re.findall(r"^tns(?:\s+max)?\s+([-0-9.]+)", log, re.MULTILINE)
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
    # `report_clock_skew` prints either "X skew" or a "Latency CRPR Skew" table
    # whose data line is three numbers.
    skew_all = re.findall(r"([-0-9.]+)\s+skew", log)
    if not skew_all:
        skew_all = re.findall(r"^\s*[-0-9.]+\s+[-0-9.]+\s+([-0-9.]+)\s*$", log[log.rfind("Clock clk"):], re.MULTILINE)
    skew_val = float(skew_all[-1]) * unit if skew_all else None
    # Post-route instance count from the written DEF; fall back to the placer's count.
    def_path = work / "design.def"
    inst = re.findall(r"^COMPONENTS\s+(\d+)", def_path.read_text(encoding="utf-8") if def_path.exists() else "", re.MULTILINE)
    inst = inst or re.findall(r"NumInstances:\s+(\d+)", log)
    instances = int(inst[-1]) if inst else 0
    wl = re.findall(r"Total wirelength:\s+(\d+)\s+um", log)
    wirelength = float(wl[-1]) if wl else float("nan")
    # Last line of the final congestion table: "Total  resource demand usage% H / V / Total".
    ovf = re.findall(r"^Total\s+\d+\s+\d+\s+[0-9.]+%\s+\d+\s+/\s+\d+\s+/\s+(\d+)", log, re.MULTILINE)
    overflow = int(ovf[-1]) if ovf else -1
    achievable = period_ps - wns_val
    return PnrResult(platform.name, period_ps, wns_val, tns_val, critical, 1e6 / achievable if achievable > 0 else float("inf"),
                     design_area, util, instances, skew_val, wirelength, overflow, drc_count, power_w, ir_vdd, ir_vss,
                     stage, timing[-5000:])


def run_pnr(openroad: Path, platform: Platform, platforms_dir: Path, netlist: Path, liberties: Sequence[Path], work: Path,
            *, top: str = "fabric_columns", period_ps: float = 3000.0, utilization: float = 45.0,
            detailed_route: bool = False, threads: int = 4) -> PnrResult:
    work.mkdir(parents=True, exist_ok=True)
    # Constants become tie cells (the detailed router refuses constant-driven
    # nets), and OpenROAD's Verilog reader rejects `wire signed`, which yosys
    # emits for signed nets.
    from fabric.synth import map_ties
    tied = work / "netlist_ties.v"
    map_ties(Path(netlist), [Path(l) for l in liberties], tied, tie_hi=platform.tie_hi, tie_lo=platform.tie_lo)
    cleaned = work / "netlist.v"
    cleaned.write_text(tied.read_text(encoding="utf-8").replace("wire signed ", "wire "), encoding="utf-8")
    script = write_flow(work, platform, platforms_dir, cleaned, liberties, top=top, period_ps=period_ps,
                        utilization=utilization, detailed_route=detailed_route, threads=threads)
    with (work / "openroad.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([str(openroad), "-exit", "-no_init", "-threads", str(threads), str(script)],
                                cwd=work, stdout=log, stderr=subprocess.STDOUT, text=True)
    log_text = (work / "openroad.log").read_text(encoding="utf-8")
    if result.returncode != 0 or not re.search(r"^wns", log_text, re.MULTILINE):
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
        if result.stage == "detailed_route":
            fmt = lambda v, u: "n/a" if v is None else f"{v:.3g} {u}"
            print(f"detailed route: {result.drc_violations} DRC violations, power {fmt(result.power_w, 'W')}, "
                  f"IR drop VDD {fmt(result.ir_drop_vdd_mv, 'mV')}, VSS {fmt(result.ir_drop_vss_mv, 'mV')}")
    if args.report:
        print(result.report)


if __name__ == "__main__":
    main()
