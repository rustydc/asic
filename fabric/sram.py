"""What a memory macro costs, and the liberty that says so to the timing tools.

The open libraries this flow maps against have no SRAM in them, so the
memories of ``rtl/fabric_sram.sv`` are blackboxes: synthesis stops at their
boundary, which is right, but a blackbox with no timing is also a hole.  Paths
that end at an address pin and paths that start at a read-data pin are not
checked at all, so a report that says a unit closes at 2 ns is only saying
that its flop-to-flop logic does.

This module fills the hole from both sides.  ``area_um2`` and the access and
setup times are a stand-in for a compiler's datasheet, in the same spirit as
the ROM area placeholder: the shape of the model is the point, and every
constant is one line to replace when a real compiler is in hand.  A macro's
delay grows with the square root of its word count and, more weakly, with its
width, because a taller array is a longer bit line and a wider one a longer
word line; its area is the bit cells plus a periphery that a small array pays
proportionally more of.  ``write_liberty`` turns a design's macros into cells
the timing tools will read, so the address path and the data path are timed
like any other.

The names are the ones yosys gives a parameterised blackbox, which are a hash
of its parameters and not anything to read.  ``macros_from_json`` takes them
from the elaborated design along with the port widths that say what each one
is, so the liberty is generated for the design in hand rather than written by
hand and kept in step.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Macro:
    """One memory instance, as its ports describe it."""
    module: str                  # the name in the netlist
    width: int                   # bits in a word
    depth: int                   # words the address reaches
    read_ports: int
    write_ports: int
    mask_bits: int               # write-mask granularity, in bits

    @property
    def bits(self) -> int:
        return self.width * self.depth


@dataclass(frozen=True)
class Process:
    """A process's memory numbers.  ``bit_um2`` is a 6T cell, ``periphery`` the
    decoders, sense amplifiers and drivers as a multiple of the array, ``base``
    the access time of a 64-word, 64-bit macro."""
    name: str
    bit_um2: float
    periphery: float
    base_access_ps: float
    setup_ps: float
    hold_ps: float
    pin_pf: float
    nand2_um2: float             # to report a macro next to the standard cells


# The bit cells are published numbers rather than guesses, which is worth
# saying because the rest of this module is a stand-in.  Intel's 45 nm test
# chip demonstrated a 0.346 um2 6T cell (153 Mb in 119 mm2, ISSCC 2006), so
# 0.35 is that node's high-density cell; TSMC's N7 high-density 6T cell is
# 0.027 um2, which is what the 7 nm row holds.  For reference at the node the
# density model projects to, TSMC's 28 nm high-density 6T cell is 0.127 um2.
# What stays a stand-in is the periphery multiple and the access time: a small
# single-port macro answers in well under a nanosecond, and the shape of the
# scaling is the point until a compiler's datasheet replaces it.
PROCESSES = {
    "nangate45": Process("nangate45", 0.35, 1.45, 620.0, 160.0, 60.0, 0.004, 0.798),
    "asap7":     Process("asap7", 0.027, 1.55, 210.0, 55.0, 20.0, 0.0008, 0.0787),
}


def area_um2(macro: Macro, process: Process) -> float:
    """Bit cells plus periphery, one array per read port.

    A second read port is a second copy of the array with the writes
    broadcast, which is how a multi-read memory is built from single-port
    macros; a second write port is a dual-port bit cell, about half as dense
    again."""
    cells = macro.bits * process.bit_um2 * (1.0 + 0.55 * (macro.write_ports - 1))
    return cells * process.periphery * macro.read_ports


def access_ps(macro: Macro, process: Process) -> float:
    """Clock to read data: the bit line grows with the square root of the word
    count, the word line with the width."""
    return process.base_access_ps * math.sqrt(max(macro.depth, 1) / 64.0) ** 0.5 * (max(macro.width, 1) / 64.0) ** 0.15


def setup_ps(macro: Macro, process: Process) -> float:
    return process.setup_ps


def clean_name(module: str) -> str:
    """A readable name for a derived blackbox.

    yosys calls it ``$paramod$<hash>\\fabric_sram``, an escaped Verilog
    identifier that the timing tools will not match against a liberty cell, so
    the netlist and the liberty are both put into this form instead."""
    match = re.search(r"\$paramod\$([0-9a-f]+)\\(\w+)", module)
    return f"{match.group(2)}_{match.group(1)[:8]}" if match else module


def clean_netlist(source: Path, target: Path) -> Path:
    """The netlist with the memories' escaped names replaced by clean ones, and
    yosys's ``wire signed``, which OpenSTA's reader does not take, removed."""
    text = Path(source).read_text(encoding="utf-8").replace("wire signed ", "wire ")
    text = re.sub(r"\\(\$paramod\$[0-9a-f]+\\\w+)\s", lambda m: clean_name(m.group(1)) + " ", text)
    Path(target).write_text(text, encoding="utf-8")
    return Path(target)


def macros_from_json(path: Path, name: str = "fabric_sram") -> list[Macro]:
    """The design's memories, from yosys's elaborated modules: a blackbox keeps
    its ports, and the widths say what the parameters were."""
    design = json.loads(Path(path).read_text())
    out = []
    for module, body in design.get("modules", {}).items():
        if name not in module or module.startswith("$abstract"):
            continue                              # yosys's deferred stub, not an instance
        ports = {p: len(v.get("bits", [])) for p, v in body.get("ports", {}).items()}
        reads = max(1, ports.get("rd_en", 1))
        writes = max(1, ports.get("wr_en", 1))
        width = ports.get("rd_data", 0) // reads
        addr = ports.get("rd_addr", 1) // reads
        mask = ports.get("wr_mask", 1) // writes
        out.append(Macro(clean_name(module), width, 1 << addr, reads, writes, max(1, width // max(mask, 1))))
    return sorted(out, key=lambda m: m.module)


def _bus_type(width: int) -> str:
    return f"""    type (bus_{width}) {{
        base_type : array;
        data_type : bit;
        bit_width : {width};
        bit_from : {width - 1};
        bit_to : 0;
        downto : true;
    }}
"""


def _pin(name: str, width: int, direction: str, body: str) -> str:
    if width <= 1:
        return f"        pin ({name}) {{\n            direction : {direction};\n{body}        }}\n"
    return (f"        bus ({name}) {{\n            bus_type : bus_{width};\n            direction : {direction};\n"
            f"{body}        }}\n")


def _cell(macro: Macro, process: Process) -> str:
    access, setup, hold = access_ps(macro, process), setup_ps(macro, process), process.hold_ps
    cap = f"            capacitance : {process.pin_pf};\n"
    ck = (f"            clock : true;\n            capacitance : {process.pin_pf * 4:.4f};\n")

    def sync_in(name: str, width: int) -> str:
        arcs = "".join(
            f"""            timing () {{
                related_pin : "clk";
                timing_type : {kind};
                rise_constraint (mem_constraint) {{ index_1 ("0.0"); index_2 ("0.0"); values ("{value}"); }}
                fall_constraint (mem_constraint) {{ index_1 ("0.0"); index_2 ("0.0"); values ("{value}"); }}
            }}
"""
            for kind, value in (("setup_rising", f"{setup:.1f}"), ("hold_rising", f"{hold:.1f}")))
        return _pin(name, width, "input", cap + arcs)

    out_arc = f"""            timing () {{
                related_pin : "clk";
                timing_type : rising_edge;
                cell_rise (mem_delay) {{ index_1 ("0.0"); index_2 ("0.0"); values ("{access:.1f}"); }}
                cell_fall (mem_delay) {{ index_1 ("0.0"); index_2 ("0.0"); values ("{access:.1f}"); }}
                rise_transition (mem_delay) {{ index_1 ("0.0"); index_2 ("0.0"); values ("20.0"); }}
                fall_transition (mem_delay) {{ index_1 ("0.0"); index_2 ("0.0"); values ("20.0"); }}
            }}
"""
    widths = sorted({macro.width * macro.read_ports, macro.width * macro.write_ports,
                     (macro.depth - 1).bit_length() * macro.read_ports,
                     (macro.depth - 1).bit_length() * macro.write_ports,
                     macro.read_ports, macro.write_ports,
                     (macro.width // macro.mask_bits) * macro.write_ports} - {0, 1})
    aw = max(1, (macro.depth - 1).bit_length())
    return "".join(_bus_type(w) for w in widths) + f"""    cell ("{macro.module}") {{
        area : {area_um2(macro, process):.2f};
        is_macro_cell : true;
{_pin("clk", 1, "input", ck)}{sync_in("rd_en", macro.read_ports)}{sync_in("rd_addr", aw * macro.read_ports)}{_pin("rd_data", macro.width * macro.read_ports, "output", cap + out_arc)}{sync_in("wr_en", macro.write_ports)}{sync_in("wr_addr", aw * macro.write_ports)}{sync_in("wr_data", macro.width * macro.write_ports)}{sync_in("wr_mask", (macro.width // macro.mask_bits) * macro.write_ports)}    }}
"""


THRESHOLDS = ("slew_lower_threshold_pct_rise", "slew_lower_threshold_pct_fall",
              "slew_upper_threshold_pct_rise", "slew_upper_threshold_pct_fall",
              "input_threshold_pct_rise", "input_threshold_pct_fall",
              "output_threshold_pct_rise", "output_threshold_pct_fall",
              "slew_derate_from_library")
DEFAULT_THRESHOLDS = {"slew_lower_threshold_pct_rise": "30.0", "slew_lower_threshold_pct_fall": "30.0",
                      "slew_upper_threshold_pct_rise": "70.0", "slew_upper_threshold_pct_fall": "70.0",
                      "input_threshold_pct_rise": "50.0", "input_threshold_pct_fall": "50.0",
                      "output_threshold_pct_rise": "50.0", "output_threshold_pct_fall": "50.0",
                      "slew_derate_from_library": "1.0"}


def thresholds(reference: Path | None) -> dict[str, str]:
    """Where a library measures a delay from and to.

    A liberty that does not say is rejected outright -- OpenSTA will not link
    it -- and a liberty that says something different from the standard cells
    beside it would have the macros' arcs measured between other points than
    the logic's.  So they are read off the library the macros will be timed
    with: NanGate's slews run 30 to 70 percent, ASAP7's 10 to 90."""
    found = dict(DEFAULT_THRESHOLDS)
    if reference is not None and Path(reference).exists():
        text = Path(reference).read_text(encoding="utf-8", errors="ignore")
        head = text[:text.find("cell (")] if "cell (" in text else text
        for name in THRESHOLDS:
            match = re.search(rf"^\s*{name}\s*:\s*([0-9.]+)\s*;", head, re.M)
            if match:
                found[name] = match.group(1)
    return found


def write_liberty(macros: Sequence[Macro], process: Process, path: Path, reference: Path | None = None) -> Path:
    """A liberty of the design's memories, for the timing tools to read beside
    the standard cells (``reference``, whose thresholds the macros take)."""
    body = "".join(_cell(m, process) for m in macros)
    limits = "".join(f"    {name} : {value};\n" for name, value in thresholds(reference).items())
    Path(path).write_text(f"""library (fabric_sram_{process.name}) {{
    delay_model : table_lookup;
{limits}    time_unit : "1ps";
    voltage_unit : "1V";
    current_unit : "1mA";
    capacitive_load_unit (1, pf);
    pulling_resistance_unit : "1kohm";
    leakage_power_unit : "1nW";
    nom_voltage : 1.0;
    nom_temperature : 25.0;
    nom_process : 1.0;
    default_max_transition : 200.0;
    lu_table_template (mem_delay) {{
        variable_1 : input_net_transition;
        variable_2 : total_output_net_capacitance;
        index_1 ("0.0");
        index_2 ("0.0");
    }}
    lu_table_template (mem_constraint) {{
        variable_1 : related_pin_transition;
        variable_2 : constrained_pin_transition;
        index_1 ("0.0");
        index_2 ("0.0");
    }}
{body}}}
""", encoding="utf-8")
    return Path(path)


def inventory(macros: Sequence[Macro], process: Process) -> dict:
    """What the memories come to: bits, area, and the slowest answer."""
    return {
        "process": process.name,
        "macros": len(macros),
        "bits": sum(m.bits * m.read_ports for m in macros),
        "area_um2": sum(area_um2(m, process) for m in macros),
        "nand2_equiv": sum(area_um2(m, process) for m in macros) / process.nand2_um2,
        "worst_access_ps": max((access_ps(m, process) for m in macros), default=0.0),
    }
