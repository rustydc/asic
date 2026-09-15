"""Load, check, and render the block-level board description in ``board.yaml``.

Usage from the repository root::

    python -m hw.board            # check and write hw/board.svg and hw/board.md
    python -m hw.board --check    # check only

The checks catch the errors that are cheap to make while the schematic is still
a document: dangling interface references, ASICs without their memory, a ring
that does not close at the FPGA, and a power budget that exceeds the input.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).parent
DEFAULT_SOURCE = HERE / "board.yaml"


@dataclass(frozen=True)
class Endpoint:
    component: str
    interface: str

    @classmethod
    def parse(cls, text: str) -> "Endpoint":
        component, _, interface = text.partition(".")
        if not interface:
            raise ValueError(f"endpoint {text!r} must be component.interface")
        return cls(component, interface)


class Board:
    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.classes: dict[str, dict] = data["part_classes"]
        self.components: dict[str, dict] = data["components"]
        self.nets: dict[str, dict] = data["nets"]
        self.kinds: dict[str, dict] = data["interface_kinds"]

    @classmethod
    def load(cls, path: Path = DEFAULT_SOURCE) -> "Board":
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")))

    # Queries ----------------------------------------------------------------

    def class_of(self, refdes: str) -> str:
        return self.components[refdes]["class"]

    def instances(self, part_class: str) -> list[str]:
        return [ref for ref, comp in self.components.items() if comp["class"] == part_class]

    def interface_kind(self, endpoint: Endpoint) -> str:
        part = self.classes[self.class_of(endpoint.component)]
        try:
            return part["interfaces"][endpoint.interface]["kind"]
        except KeyError as error:
            raise KeyError(f"{endpoint.component} ({self.class_of(endpoint.component)}) has no "
                           f"interface {endpoint.interface!r}") from error

    def signal_count(self, kind: str) -> int:
        signals = self.kinds[kind]["signals"]
        total = 0
        for name, count in signals.items():
            total += count * (2 if name.endswith("pair") or name.endswith("pairs") else 1)
        return total

    def pin_budget(self, refdes: str) -> int:
        part = self.classes[self.class_of(refdes)]
        return sum(self.signal_count(spec["kind"]) for spec in part["interfaces"].values())

    # Power model -------------------------------------------------------------

    @property
    def power_model(self) -> dict[str, Any]:
        return self.data.get("power_model", {})

    def design_tokens_per_second(self) -> float:
        return float(self.power_model.get("design_tokens_per_second", 0.0))

    def part_power_w(self, part_class: str, tokens_per_second: float | None = None) -> float:
        """Power of one instance of a part class at a throughput.

        Parts with ``tdp_w`` are fixed budgets.  The ASICs carry ``static_w``
        plus ``macs_per_token``; their dynamic power is throughput times energy
        per token from ``power_model.mac_energy_pj``, plus the LPDDR traffic
        energy for parts marked ``memory_traffic``.
        """
        spec = self.classes[part_class]
        if "tdp_w" in spec:
            return float(spec["tdp_w"])
        tps = self.design_tokens_per_second() if tokens_per_second is None else tokens_per_second
        model = self.power_model
        energy_j = float(spec["macs_per_token"]) * float(model["mac_energy_pj"]) * 1e-12
        if spec.get("memory_traffic"):
            energy_j += float(model["memory_bytes_per_token_per_asic"]) * float(model["memory_energy_pj_per_byte"]) * 1e-12
        return float(spec.get("static_w", 0.0)) + energy_j * tps

    def power_budget_w(self, tokens_per_second: float | None = None) -> tuple[float, float]:
        """Return (load watts, input watts after regulator efficiency) at a throughput
        (default: the model's design point)."""
        load = sum(self.part_power_w(self.class_of(ref), tokens_per_second) for ref in self.components)
        efficiency = self.data["power_tree"]["regulator_efficiency"]
        return load, load / efficiency

    def core_current_a(self, part_class: str, tokens_per_second: float | None = None) -> float:
        """Core-rail current of one ASIC, taking all of its power on VDD_CORE (conservative)."""
        volts = self.data["power_tree"]["rails"]["VDD_CORE"]["volts"]
        return self.part_power_w(part_class, tokens_per_second) / volts

    def power_available_w(self) -> float:
        return sum(entry.get("watts", 0) for entry in self.data["board"]["power_input"])

    def ring(self) -> list[tuple[Endpoint, Endpoint]]:
        return [(Endpoint.parse(a), Endpoint.parse(b)) for a, b in self.nets["ring"]["hops"]]

    # Checks -----------------------------------------------------------------

    def check(self) -> list[str]:
        problems: list[str] = []
        problems += self._check_references()
        problems += self._check_ring()
        problems += self._check_memory()
        problems += self._check_power()
        return problems

    def _check_references(self) -> list[str]:
        problems = []
        for ref, comp in self.components.items():
            if comp["class"] not in self.classes:
                problems.append(f"{ref}: unknown part class {comp['class']!r}")
        for part, spec in self.classes.items():
            for name, iface in spec.get("interfaces", {}).items():
                if iface["kind"] not in self.kinds:
                    problems.append(f"{part}.{name}: unknown interface kind {iface['kind']!r}")
            for rail in spec.get("rails", []):
                if rail not in self.data["power_tree"]["rails"]:
                    problems.append(f"{part}: rail {rail} is not in the power tree")
        for net_name, net in self.nets.items():
            pairs = net.get("hops", []) + net.get("channels", [])
            for a, b in pairs:
                for text in (a, b):
                    endpoint = Endpoint.parse(text)
                    if endpoint.component not in self.components:
                        problems.append(f"{net_name}: unknown component {endpoint.component}")
                        continue
                    try:
                        kind = self.interface_kind(endpoint)
                    except KeyError as error:
                        problems.append(f"{net_name}: {error}")
                        continue
                    if kind != net["kind"] and not (net["kind"] == "ddr4_x16" and kind == "ddr4_x64"):
                        problems.append(f"{net_name}: {text} is {kind}, net is {net['kind']}")
            for ref in net.get("slaves", []) + net.get("sinks", []) + net.get("chain", []):
                if ref not in self.components:
                    problems.append(f"{net_name}: unknown component {ref}")
        return problems

    def _check_ring(self) -> list[str]:
        problems = []
        hops = self.ring()
        if hops[0][0].component != "U_FPGA" or hops[-1][1].component != "U_FPGA":
            problems.append("ring must start and end at U_FPGA")
        for (_, sink), (source, _) in zip(hops, hops[1:]):
            if sink.component != source.component:
                problems.append(f"ring breaks between {sink.component} and {source.component}")
        visited = [hop[1].component for hop in hops[:-1]]
        if len(set(visited)) != len(visited):
            problems.append("ring visits a component twice")
        order = [self.class_of(ref) for ref in visited]
        layers = [ref for ref in visited if self.class_of(ref) == "layer_asic"]
        heads = [ref for ref in visited if self.class_of(ref) == "head_asic"]
        if order != ["layer_asic"] * len(layers) + ["head_asic"] * len(heads):
            problems.append("ring order must be all layer ASICs then all head ASICs")
        if set(layers) != set(self.instances("layer_asic")):
            problems.append("every layer ASIC must be on the ring exactly once")
        if set(heads) != set(self.instances("head_asic")):
            problems.append("every head ASIC must be on the ring exactly once")
        for ref in ("mgmt", "jtag", "refclk"):
            members = self.nets[ref].get("slaves") or self.nets[ref].get("chain") or self.nets[ref].get("sinks")
            missing = set(layers + heads) - set(members)
            if missing:
                problems.append(f"{ref} net is missing {sorted(missing)}")
        return problems

    def _check_memory(self) -> list[str]:
        problems = []
        attached: dict[str, list[str]] = {ref: [] for ref in self.components}
        used_devices: set[str] = set()
        for a, b in self.nets["memory"]["channels"]:
            asic, device = Endpoint.parse(a), Endpoint.parse(b)
            attached[asic.component].append(asic.interface)
            if device.component in used_devices:
                problems.append(f"memory device {device.component} attached twice")
            used_devices.add(device.component)
        for ref in self.instances("layer_asic"):
            channels = sorted(attached[ref])
            if channels != ["lpddr_ch0", "lpddr_ch1"]:
                problems.append(f"{ref} needs two LPDDR channels, has {channels}")
        for ref in self.instances("head_asic"):
            if attached[ref]:
                problems.append(f"{ref} is a head ASIC and must not have memory attached")
        unused = set(self.instances("lpddr5x")) - used_devices
        if unused:
            problems.append(f"unattached memory devices: {sorted(unused)}")
        return problems

    def _check_power(self) -> list[str]:
        _, input_w = self.power_budget_w()
        available = self.power_available_w()
        if input_w > available:
            return [f"power budget {input_w:.0f} W exceeds available input {available:.0f} W"]
        return []

    # Reports ----------------------------------------------------------------

    def summary_markdown(self) -> str:
        load, input_w = self.power_budget_w()
        lines = [f"# {self.data['board']['name']} board summary", "",
                 "Generated by `python -m hw.board` from `hw/board.yaml`. Do not edit.", "",
                 "## Bill of materials (major parts)", "",
                 f"| Class | Qty | Description | Power each at {self.design_tokens_per_second():.0f} tok/s (W) |",
                 "| --- | ---: | --- | ---: |"]
        for part, spec in self.classes.items():
            qty = len(self.instances(part))
            if qty:
                lines.append(f"| {part} | {qty} | {spec['description']} | {self.part_power_w(part):.1f} |")
        model = self.power_model
        lines += ["", "## Power", "",
                  f"At the design point of {self.design_tokens_per_second():.0f} tokens/s and "
                  f"{model.get('mac_energy_pj', 0):.1f} pJ per MAC: load {load:.0f} W, input {input_w:.0f} W at "
                  f"{self.data['power_tree']['regulator_efficiency']:.0%} regulator efficiency, "
                  f"available {self.power_available_w():.0f} W.", "",
                  "| Tokens/s | Layer ASIC (W) | Core current at "
                  f"{self.data['power_tree']['rails']['VDD_CORE']['volts']:.2f} V (A) | Load (W) | Input (W) |",
                  "| ---: | ---: | ---: | ---: | ---: |"]
        for tps in model.get("report_tokens_per_second", []):
            load_t, input_t = self.power_budget_w(tps)
            lines.append(f"| {tps} | {self.part_power_w('layer_asic', tps):.0f} | "
                         f"{self.core_current_a('layer_asic', tps):.0f} | {load_t:.0f} | {input_t:.0f} |")
        lines += ["", "## Activation ring", "",
                  " -> ".join([self.ring()[0][0].component] + [hop[1].component for hop in self.ring()]), "",
                  "## Signal pin budget per instance (excluding power and ground)", "",
                  "| Class | Signal pins |", "| --- | ---: |"]
        for part in self.classes:
            if self.classes[part].get("interfaces") and self.instances(part):
                lines.append(f"| {part} | {self.pin_budget(self.instances(part)[0])} |")
        link = self.kinds["link"]
        lines += ["", "## Link", "", f"{link['description']}; {self.signal_count('link')} signals per hop, "
                  f"{len(self.ring())} hops, max {link['max_length_mm']} mm."]
        return "\n".join(lines) + "\n"

    def svg(self) -> str:
        """Block diagram: the eleven ring nodes on a regular polygon, clockwise
        from the FPGA at the bottom (the chassis rear), memories outside."""
        import math
        W, H = 1180, 900
        box_w, box_h = 96, 60
        mem_w, mem_h = 40, 22
        cx, cy, radius = 590, 420, 300
        refs = [source.component for source, _ in self.ring()]
        n = len(refs)
        angle = {ref: -90.0 - k * 360.0 / n for k, ref in enumerate(refs)}       # clockwise, FPGA at the bottom
        pos: dict[str, tuple[float, float]] = {}
        for ref, a in angle.items():
            h = box_h + 20 if ref == "U_FPGA" else box_h
            pos[ref] = (cx + radius * math.cos(math.radians(a)) - box_w / 2, cy - radius * math.sin(math.radians(a)) - h / 2)
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="sans-serif" font-size="12">' % (W, H),
                 f'<rect width="{W}" height="{H}" fill="#fafafa"/>',
                 f'<text x="20" y="28" font-size="16" font-weight="bold">{self.data["board"]["name"]}: block diagram</text>',
                 f'<text x="20" y="46" fill="#555">{self.data["board"]["form_factor"]}</text>']

        def box(ref: str, x: float, y: float, w: float, h: float, fill: str, label: str, sub: str = "") -> None:
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="#333"/>')
            parts.append(f'<text x="{x + w / 2:.1f}" y="{y + 20:.1f}" text-anchor="middle" font-weight="bold">{label}</text>')
            if sub:
                parts.append(f'<text x="{x + w / 2:.1f}" y="{y + 38:.1f}" text-anchor="middle" fill="#333">{sub}</text>')

        def rect(ref: str) -> tuple[float, float, float, float]:
            x, y = pos[ref]
            h = box_h + 20 if ref == "U_FPGA" else box_h
            return x, y, x + box_w, y + h

        def centre(ref: str) -> tuple[float, float]:
            x1, y1, x2, y2 = rect(ref)
            return (x1 + x2) / 2, (y1 + y2) / 2

        def clip_to_edge(ref: str, x_from: float, y_from: float) -> tuple[float, float]:
            """Point where the segment from (x_from, y_from) to ref's centre meets ref's box edge."""
            x1, y1, x2, y2 = rect(ref)
            cx_, cy_ = centre(ref)
            dx, dy = cx_ - x_from, cy_ - y_from
            candidates = []
            for edge, delta in ((x1, dx), (x2, dx)):
                if delta:
                    candidates.append((edge - x_from) / delta)
            for edge, delta in ((y1, dy), (y2, dy)):
                if delta:
                    candidates.append((edge - y_from) / delta)
            eps = 1e-6
            valid = []
            for t in candidates:
                px, py = x_from + dx * t, y_from + dy * t
                if 0 < t <= 1 and x1 - eps <= px <= x2 + eps and y1 - eps <= py <= y2 + eps:
                    valid.append(t)
            t = min(valid)
            return x_from + dx * t, y_from + dy * t

        # Ring hops.
        parts.append('<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
                     '<path d="M0,0 L8,4 L0,8 z" fill="#1f5fbf"/></marker></defs>')
        line_style = 'stroke="#1f5fbf" stroke-width="2.5" marker-end="url(#arrow)" fill="none"'
        for source, sink in self.ring():
            (cx1, cy1), (cx2, cy2) = centre(source.component), centre(sink.component)
            x1, y1 = clip_to_edge(source.component, cx2, cy2)
            x2, y2 = clip_to_edge(sink.component, cx1, cy1)
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {line_style}/>')
        # Memory devices outside each layer ASIC, along the radius.
        memory_of: dict[str, list[str]] = {}
        for a, b in self.nets["memory"]["channels"]:
            memory_of.setdefault(Endpoint.parse(a).component, []).append(Endpoint.parse(b).component)
        for ref in self.instances("layer_asic"):
            a = math.radians(angle[ref])
            ux, uy = math.cos(a), -math.sin(a)                     # outward, SVG y down
            tx, ty = -uy, ux                                       # tangent
            ex, ey = centre(ref)
            for index, device in enumerate(memory_of.get(ref, [])):
                side = -1 if index == 0 else 1
                mx = ex + ux * 62 + tx * side * 26 - mem_w / 2
                my = ey + uy * 62 + ty * side * 26 - mem_h / 2
                parts.append(f'<rect x="{mx:.1f}" y="{my:.1f}" width="{mem_w}" height="{mem_h}" fill="#e8f3e8" stroke="#333"/>')
                parts.append(f'<text x="{mx + mem_w / 2:.1f}" y="{my + 15:.1f}" text-anchor="middle" font-size="10">{device[2:]}</text>')
                sx, sy = clip_to_edge(ref, mx + mem_w / 2, my + mem_h / 2)
                parts.append(f'<line x1="{mx + mem_w / 2:.1f}" y1="{my + mem_h / 2:.1f}" x2="{sx:.1f}" y2="{sy:.1f}" stroke="#2a7a2a" stroke-width="2"/>')
        # Boxes.
        fx, fy = pos["U_FPGA"]
        box("U_FPGA", fx, fy, box_w, box_h + 20, "#fff2cc", "U_FPGA", "PCIe Gen4 x8 (cable)")
        parts.append(f'<rect x="{fx + 18:.1f}" y="{fy + box_h + 32:.1f}" width="60" height="18" fill="#ddd" stroke="#333"/>')
        parts.append(f'<text x="{fx + 48:.1f}" y="{fy + box_h + 45:.1f}" text-anchor="middle" font-size="9">host cable</text>')
        parts.append(f'<line x1="{fx + 48:.1f}" y1="{fy + box_h + 20:.1f}" x2="{fx + 48:.1f}" y2="{fy + box_h + 32:.1f}" stroke="#333"/>')
        parts.append(f'<rect x="{fx + box_w + 12:.1f}" y="{fy + 52:.1f}" width="56" height="18" fill="#eee" stroke="#333"/>')
        parts.append(f'<text x="{fx + box_w + 40:.1f}" y="{fy + 65:.1f}" text-anchor="middle" font-size="9">DDR4 x64</text>')
        parts.append(f'<line x1="{fx + box_w:.1f}" y1="{fy + 61:.1f}" x2="{fx + box_w + 12:.1f}" y2="{fy + 61:.1f}" stroke="#333"/>')
        parts.append(f'<rect x="{fx - 72:.1f}" y="{fy + 52:.1f}" width="60" height="18" fill="#eee" stroke="#333" stroke-dasharray="3,2"/>')
        parts.append(f'<text x="{fx - 42:.1f}" y="{fy + 65:.1f}" text-anchor="middle" font-size="9">SFP+ (opt)</text>')
        for ref in self.instances("layer_asic"):
            x, y = pos[ref]
            box(ref, x, y, box_w, box_h, "#dbe8f7", ref, f"L{self.components[ref]['layers']}")
        for ref in self.instances("head_asic"):
            x, y = pos[ref]
            box(ref, x, y, box_w, box_h, "#f7dbdb", ref, "head mode")
        parts.append(f'<rect x="{cx - 48}" y="{cy - 30}" width="96" height="60" rx="6" fill="#eee" stroke="#333"/>')
        parts.append(f'<text x="{cx}" y="{cy - 6}" text-anchor="middle" font-weight="bold">U_CLK</text>')
        parts.append(f'<text x="{cx}" y="{cy + 12}" text-anchor="middle" font-size="10">shared regulators</text>')
        # Legend and budgets.
        load, input_w = self.power_budget_w()
        parts.append(f'<text x="20" y="{H - 40}" fill="#333">Blue: activation ring, {self.signal_count("link")}-signal '
                     f'source-synchronous link per hop, clockwise on a regular {n}-gon from the FPGA at the rear. '
                     f'Green: LPDDR5X x32 channels. Power budget {load:.0f} W load / {input_w:.0f} W input.</text>')
        parts.append(f'<text x="20" y="{H - 20}" fill="#333">Management SPI, JTAG chain, and reference clock fan out '
                     f'from U_FPGA / U_CLK to all ten ASICs (not drawn); the BMC, PSUs and fans are off the ring.</text>')
        parts.append("</svg>")
        return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--check", action="store_true", help="check only; do not write outputs")
    args = parser.parse_args()
    board = Board.load(args.source)
    problems = board.check()
    for problem in problems:
        print(f"ERROR: {problem}")
    if problems:
        raise SystemExit(1)
    load, input_w = board.power_budget_w()
    print(f"board OK: {len(board.components)} components, {len(board.ring())} ring hops, "
          f"{load:.0f} W load / {input_w:.0f} W input of {board.power_available_w():.0f} W available")
    if not args.check:
        (HERE / "board.svg").write_text(board.svg(), encoding="utf-8")
        (HERE / "board.md").write_text(board.summary_markdown(), encoding="utf-8")
        print("wrote hw/board.svg and hw/board.md")


if __name__ == "__main__":
    main()
