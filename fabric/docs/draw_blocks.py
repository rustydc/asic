"""Draws the block diagrams in docs/ (appliance.svg, layer_die.svg, tile.svg) from a box list.

    python -m fabric.docs.draw_blocks

The SVGs are styled by class so the same drawing sits inline in a page whose
stylesheet recolours it; the standalone files carry a light stylesheet.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent

STYLE = """
.box{fill:#ffffff;stroke:#2a3a4a;stroke-width:1.2}
.unit{fill:#e8f1f6;stroke:#1f5f7a;stroke-width:1.2}
.mem{fill:#fbf1de;stroke:#9a6a12;stroke-width:1.2}
.ctl{fill:#eaf3e6;stroke:#3d7a2e;stroke-width:1.2}
.ext{fill:#f1eef7;stroke:#5a4a8a;stroke-width:1.2}
.bus{fill:#dfe6ec;stroke:#2a3a4a;stroke-width:1.2}
.group{fill:none;stroke:#8a96a3;stroke-width:1;stroke-dasharray:4 3}
.t{font:12px 'IBM Plex Sans',system-ui,sans-serif;fill:#17202a}
.h{font:600 13px 'IBM Plex Sans',system-ui,sans-serif;fill:#17202a}
.s{font:10px 'IBM Plex Mono',ui-monospace,monospace;fill:#3e4c5a}
.g{font:600 11px 'IBM Plex Sans',system-ui,sans-serif;fill:#5a6773;letter-spacing:.06em}
.e{stroke:#2a3a4a;stroke-width:1.2;fill:none;marker-end:url(#ah)}
.e2{stroke:#2a3a4a;stroke-width:1.2;fill:none;marker-end:url(#ah);marker-start:url(#at)}
.ed{stroke:#9a6a12;stroke-width:1.4;fill:none;marker-end:url(#ahm);marker-start:url(#atm)}
.ec{stroke:#3d7a2e;stroke-width:1.2;fill:none;marker-end:url(#ahc)}
.lbl{font:10px 'IBM Plex Mono',ui-monospace,monospace;fill:#3e4c5a}
"""

DEFS = """<defs>
<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#2a3a4a"/></marker>
<marker id="at" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M10 0L0 5L10 10z" fill="#2a3a4a"/></marker>
<marker id="ahm" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#9a6a12"/></marker>
<marker id="atm" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M10 0L0 5L10 10z" fill="#9a6a12"/></marker>
<marker id="ahc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#3d7a2e"/></marker>
</defs>"""


class Drawing:
    def __init__(self, width: int, height: int, title: str) -> None:
        self.w, self.h, self.title = width, height, title
        self.parts: list[str] = []

    def box(self, x, y, w, h, title, lines=(), cls="box", rx=4):
        self.parts.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/>')
        ty = y + 17
        self.parts.append(f'<text class="h" x="{x + w / 2}" y="{ty}" text-anchor="middle">{esc(title)}</text>')
        for line in lines:
            ty += 14
            self.parts.append(f'<text class="s" x="{x + w / 2}" y="{ty}" text-anchor="middle">{esc(line)}</text>')
        return (x, y, w, h)

    def group(self, x, y, w, h, label):
        self.parts.append(f'<rect class="group" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>')
        self.parts.append(f'<text class="g" x="{x + 10}" y="{y + 15}">{esc(label.upper())}</text>')

    def text(self, x, y, s, cls="t", anchor="start"):
        self.parts.append(f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{esc(s)}</text>')

    def edge(self, points, cls="e", label=None, lx=None, ly=None, anchor="middle"):
        d = "M" + " L".join(f"{x} {y}" for x, y in points)
        self.parts.append(f'<path class="{cls}" d="{d}"/>')
        if label:
            (x0, y0), (x1, y1) = points[0], points[-1]
            self.text(lx if lx is not None else (x0 + x1) / 2, ly if ly is not None else (y0 + y1) / 2 - 4, label, "lbl", anchor)

    def svg(self, inline: bool = False) -> str:
        style = "" if inline else f"<style>{STYLE}</style>"
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
                f'role="img" aria-label="{esc(self.title)}">{style}{DEFS}' + "".join(self.parts) + "</svg>")


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def appliance() -> Drawing:
    d = Drawing(1260, 330, "The appliance: host, FPGA, the ring of layer dies and head dies, each layer die's PSRAM")
    d.box(20, 120, 110, 60, "Host", ["PCIe Gen4 x4-x8", "SlimSAS cable"], "ext")
    d.box(170, 100, 150, 100, "Controller FPGA", ["host protocol, scheduler", "sampling, context alloc", "embedding table (DDR4)", "bring-up, telemetry"], "ctl")
    d.edge([(130, 150), (170, 150)], "e2", "PCIe", 150, 140)
    # ring of dies
    d.group(345, 30, 895, 270, "ring: source-synchronous ready/valid packet link, 32 bits at 250 MHz DDR, framing + CRC")
    x = 360
    for i in range(8):
        d.box(x, 60, 88, 78, f"Layer die {i}", [f"layers {4 * i}-{4 * i + 3}", "R R R G", "3306 tiles"], "unit")
        d.box(x, 210, 88, 58, "16 x PSRAM", ["APS512XXN", "HPI x16"], "mem")
        d.edge([(x + 44, 138), (x + 44, 210)], "ed", "1 GB, 16 GB/s" if i == 0 else None, x + 50, 178, "start")
        if i < 7:
            d.edge([(x + 88, 99), (x + 96, 99)], "e")
        x += 96
    d.box(x + 4, 60, 76, 78, "Head die A", ["vocab rows 0-124K", "top-k + LSE", "no memory"], "unit")
    d.edge([(x - 8, 99), (x + 4, 99)], "e")
    d.box(x + 4, 160, 76, 78, "Head die B", ["vocab rows 124K-248K", "top-k + LSE", "no memory"], "unit")
    d.edge([(x + 42, 138), (x + 42, 160)], "e")
    # link in and out
    d.edge([(320, 130), (360, 99)], "e")
    d.text(20, 228, "work item in: hidden vector, context, position", "lbl")
    d.edge([(x + 4, 199), (x + 4, 285), (300, 285), (300, 200)], "e", "work item out + two partial top-k lists", 700, 281)
    return d


def layer_die() -> Drawing:
    d = Drawing(1360, 720, "A layer die: sequencer, command bus, the units behind their adapters, the vector buffer, the memory path")
    # sequencer
    d.box(20, 30, 200, 118, "Token sequencer", ["program memory 256b x DEPTH", "per-buffer scoreboards", "wr_cnt / rd_cnt per id", "in-order issue, tag return"], "ctl")
    d.box(240, 30, 150, 60, "Constants SRAM", ["requant mult/shift, eps", "gains; ~80 KB over SPI"], "box")
    d.box(240, 100, 150, 48, "Link + SPI", ["ring packets in/out", "mode strap, boot"], "ext")
    d.edge([(140, 148), (140, 190)], "ec", "command bus: cmd_valid[unit], engine, len, src, dst, a2, a3, arg, tag / cmd_ready, done_valid, done_tag", 150, 175, "start")
    # command bus bar
    d.parts.append('<rect class="bus" x="20" y="190" width="1320" height="16" rx="3"/>')
    d.text(680, 202, "command bus (one unit addressed per cycle; each engine returns its tag on its done port)", "s", "middle")
    # adapters + units
    units = [
        ("Pass adapter", ["tiles: NT x (ROM + columns)", "pass table: pass, rb,", "chain, last, raw, nbytes,", "dst_off; T tokens a pass"], 200, "unit"),
        ("Norm x2", ["fabric_rmsnorm", "RMS / L2 / gated", "n_beats at run time"], 130, "unit"),
        ("Conv", ["fabric_conv_silu", "4-tap causal + SiLU", "hist -> hist_next"], 130, "unit"),
        ("Gates", ["fabric_head_gates", "a, b accs ->", "decay, beta"], 120, "unit"),
        ("State x4", ["fabric_delta_state8", "header beat + K rows", "q,k,v,gates -> y"], 160, "unit"),
        ("SwiGLU", ["fabric_swiglu", "silu(g) * u"], 110, "unit"),
        ("Residual", ["fabric_residual", "h + y*mult >> sh"], 125, "unit"),
        ("Rotary x2", ["table: pos->sin/cos", "head: norm, rotate"], 150, "unit"),
        ("Attention x4", ["fabric_attention", "online softmax, gate", "records from rows"], 150, "unit"),
    ]
    x = 45
    tops = []
    for name, lines, w, cls in units:
        d.box(x, 226, w - 10, 84, name, lines, cls)
        d.edge([(x + (w - 10) / 2, 206), (x + (w - 10) / 2, 226)], "ec")
        d.edge([(x + (w - 10) / 2, 310), (x + (w - 10) / 2, 340)], "e2")
        tops.append(x + (w - 10) / 2)
        x += w
    # vector buffer bar
    d.parts.append('<rect class="bus" x="20" y="340" width="1320" height="40" rx="3"/>')
    d.text(680, 357, "Vector buffer (fabric_vb): byte-addressed SRAM, 24-bit addresses, 16-byte beats", "h", "middle")
    d.text(680, 373, "NR read ports (norm 4, tiles TMAX, conv 2, gates 2, state 4, swiglu 2, residual 2, rotary 2, attn 4, mem 1), NW=19 write ports with byte enables, read data one cycle later", "s", "middle")
    # memory unit
    d.group(20, 400, 1320, 300, "memory path: one port per die")
    d.box(40, 430, 220, 120, "Memory unit", ["mover: state/hist DMA", "append: window, block,", "index, block sums", "index_scan + topk", "rows: mover, as stored"], "mem")
    d.edge([(145, 380), (145, 430)], "e2")
    d.edge([(30, 206), (30, 490), (40, 490)], "ec")
    d.box(290, 450, 120, 80, "Arbiter", ["3 requesters", "one req in flight", "per requester"], "mem")
    d.edge([(260, 490), (290, 490)], "ed")
    d.box(440, 450, 140, 80, "Bridge (CDC)", ["async FIFOs: req 4,", "wdata 16, rdata 8", "core clk <-> 250 MHz"], "mem")
    d.edge([(410, 490), (440, 490)], "ed", "memory port: req / wdata / rdata", 425, 440)
    d.box(610, 450, 135, 80, "Stripe unit", ["beat address ->", "device, 25b addr,", "8b beats; x_done"], "mem")
    d.edge([(580, 490), (610, 490)], "ed")
    d.box(775, 430, 160, 120, "Channel ctrl x16", ["HPI x16 protocol", "MR0/4/8 init, refresh", "tCEM, tCPH, latency", "128b beats <-> 16b DQ"], "mem")
    d.edge([(745, 490), (775, 490)], "ed", "x_valid[dev]", 760, 440)
    d.box(965, 450, 120, 80, "PHY", ["delay lines", "DLL period/quarter", "DQS capture"], "mem")
    d.edge([(935, 490), (965, 490)], "ed")
    d.box(1115, 440, 110, 100, "PSRAM x16", ["APS512XXN", "64 MB each", "ce_n, dq[15:0],", "dm, dqs"], "ext")
    d.edge([(1085, 490), (1115, 490)], "ed")
    d.text(40, 590, "per-context memory map (2 KB pages): for each recurrent layer the state (v_heads slots: header beat + K rows int8) and the conv history;", "s")
    d.text(40, 606, "for the global layer the window (local_window x KV heads x key+value), the block store (one record per closed block), the index (4-bit codes + scale per block), the block sums.", "s")
    d.text(40, 640, "the command bus operands are byte addresses into the vector buffer (src, dst, a2, a3) or beat addresses into memory (memory unit); len is a beat count or a token count.", "s")
    d.text(40, 656, "programs are written by fabric/sequencer.py, placed by fabric/engine.py (Layout) and encoded as 256-bit words: 250 bits used.", "s")
    return d


def tile() -> Drawing:
    d = Drawing(1180, 250, "A tile: the via ROM, the column datapath, the shared requantizer")
    d.box(20, 60, 160, 100, "Via ROM", ["4096 rows x 64 cols x 4b", "signed-digit pairs", "read P=2 rows per cycle", "by cycle counter"], "box")
    d.box(230, 40, 310, 140, "Columns (fabric_columns)", ["stage A: taps select (no arithmetic)", "stage B: carry-save term reduction", "stage C: carry-save accumulate (sum, carry)", "T tokens: T accumulator sets", "psum_in chain from the previous row block"], "unit")
    d.edge([(180, 110), (230, 110)], "e", "rom_words", 205, 100)
    d.box(620, 40, 240, 140, "Requantizer walk", ["one shared unit per tile", "walks the 64 columns after 2048 cycles", "resolve carries (2 x 12b), x mult,", "3 add stages, shift, saturate", "9 cycles after the walk"], "unit")
    d.edge([(540, 110), (620, 110)], "e", "acc pair", 580, 100)
    d.box(900, 40, 260, 140, "Outputs", ["q_out: T x 64 int8 (q_valid)", "psum_out: T x 64 x ACC, for the", "chained next tile (raw)", "done"], "box")
    d.edge([(860, 110), (900, 110)], "e")
    d.text(20, 215, "inputs: start (loads accumulators from psum_in), x_valid / x_data (P activations per cycle per token, in row order), x_ready, mult/shift per column from the constants SRAM.", "s")
    return d


def main() -> None:
    for name, fn in (("appliance", appliance), ("layer_die", layer_die), ("tile", tile)):
        (HERE / f"{name}.svg").write_text(fn().svg(), encoding="utf-8")
        (HERE / f"{name}.inline.svg").write_text(fn().svg(inline=True), encoding="utf-8")
    print("wrote", ", ".join(f"{n}.svg" for n in ("appliance", "layer_die", "tile")))


if __name__ == "__main__":
    main()
