# Appliance board (revision A0)

Block-level schematic capture for the first appliance board. `board.yaml` is
the source of truth; `python -m hw.board` checks it and regenerates
`board.svg` (block diagram) and `board.md` (bill of materials, power, pin
budget, ring order). Everything here is a budget to validate, not a vendor-
confirmed number, until the ASIC pinout and power are characterized.

```bash
python -m hw.board            # check, then write board.svg and board.md
python -m unittest discover -s hw/tests -t .
```

## Population

| Part | Qty | Role |
| --- | ---: | --- |
| Layer ASIC | 8 | four decoder layers each, personalized coefficient masks, layer-mode strap |
| Head ASIC | 2 | same die, head-mode strap, each holds half of the 248K-row LM head; no memory |
| LPDDR5X 16 Gb x32 | 16 | two per layer ASIC: 4 GB and 77 GB/s raw per ASIC at 9600 MT/s |
| Controller FPGA | 1 | PCIe Gen4 x8 endpoint, ring source and sink, sampling, scheduling, management |
| DDR4 x16 | 4 | 4 GB on the FPGA for the embedding table and per-context metadata |
| Clock generator | 1 | reference clock to all eleven devices |
| SFP+ cage | 1 (optional) | standalone 10GbE operation, bracket mounted |

## Topology

One unidirectional activation ring:

```text
FPGA -> A0 -> A1 -> A2 -> A3 -> A4 -> A5 -> A6 -> A7 -> H0 -> H1 -> FPGA
```

Each hop is the 32-bit source-synchronous DDR link from the architecture doc
(36 signals, 2 GB/s raw, under 80 mm). Head chips forward the incoming hidden
vector unchanged and append their top-k logits and partial softmax sum, so no
broadcast path is needed. The FPGA merges the two lists and samples.

Layer ASICs sit in two rows of four so the ring snakes out along row A and
back along row B, ending next to the FPGA. Both head chips sit at the bracket
end beside the FPGA. Each layer ASIC has one LPDDR5X device on either side.

Management is a SPI bus from the FPGA with one chip select per ASIC, a JTAG
chain through all eleven devices, and a fanned-out reference clock. Each ASIC
has a two-pin mode strap (layer or head).

## Power

ASIC power is no longer a placeholder: it is throughput times energy per
token, from the `power_model` block in `board.yaml`. The one measured input
is the energy of a fixed-weight multiply-accumulate in the column datapath,
0.39 pJ on ASAP7 at the tool's default activity
(`fabric/results/pnr_asap7_signoff.json`), which derates to about 1 pJ for
real activity and projects to 2 to 4 pJ on a 28 nm-class node; the model
carries 3 pJ. Each layer ASIC does 866M MACs per token (9B geometry), each
head ASIC 508M, and the layer ASICs also move about 4.2 MB per token of
index and KV traffic at roughly 40 pJ per byte. The other parts keep fixed
budgets (1 W per LPDDR5X, 25 W FPGA).

At the simulator's 14.5K tokens/s ceiling that is 43 W per layer ASIC and
439 W of load, 517 W of 12 V input at 85 percent regulator efficiency.
`board.md` tabulates the same at 5K, 25K and 50K tokens/s: the board runs
from about 235 W to 1.6 kW of input across the throughput range, and a
layer ASIC's core rail goes from 20 A to 180 A at 0.8 V. The first draft's
slot plus one 8-pin connector (216 W) is therefore replaced by a 12V-2x6
(12VHPWR) connector, 600 W plus the slot's 66 W, and each ASIC needs a
multiphase core regulator. A 7 nm-class die at about 1 pJ per MAC would
put the whole range back under 600 W and the design point under one 8-pin
connector; the process choice is the biggest lever on the power supply.

The power tree is a 12 V intermediate bus into point-of-load regulators. Each
ASIC gets its own core buck with PMBus telemetry so per-chip current is
observable during bring-up. The LPDDR5X rails, 1.8 V I/O, FPGA rails, DDR4
rails, and 3.3 V housekeeping are shared.

## Host interface: PCIe, and why not Ethernet or PoE

**PoE is out.** The highest PoE class (802.3bt Type 4) delivers 71 W to the
device. This board needs hundreds of watts at any useful throughput. Even a
4B-geometry build at the lowest rate in the table sits well above the PoE
ceiling once the FPGA and memory are counted.

**Ethernet as the data path is viable but not the first build.** The token
traffic is tiny. At 50K tokens/s with a few hundred bytes per token, both
directions together are under 20 MB/s, and a 128K prefill moves 512 KB of
token ids. Even 1GbE would carry it. The reasons to still start with PCIe:

* the FPGA's hard PCIe block is the cheapest working host link, with no
  network stack to write in fabric;
* PCIe round trips are a few microseconds, below the 250 to 500 µs token
  latency target, while a network hop adds tens of microseconds and jitter;
* a PCIe card form factor gives power, cooling, and mechanicals for free.

PCIe Gen4 x4 is enough bandwidth. The board wires x8 because the FPGA's hard
block is x8 and the lanes cost nothing.

The SFP+ cage on the bracket keeps the standalone-appliance option open. The
FPGA transceivers are there anyway, so a later firmware can serve a UDP or
gRPC-style token protocol over 10GbE with no board change. A standalone box
would then need its own 12 V supply rather than a host slot.

## ASIC package and ball map

`python -m hw.pinout` derives the ASIC package and its ball map from the
board description and writes `hw/pinout/asic_ballmap.csv`, `asic_ballmap.json`
and `report.md`. The pinout is co-designed with the die floorplan and the
substrate, and the substrate is the packaging house's work; what we owe them
is the ball count the power model demands, the package that carries it, and
a ball map with the constraints a substrate and a PCB can both route. The
rules live in the `package_selection` block of `board.yaml`:

* the core rail needs one ball per 0.5 A at the rated throughput, and ground
  matches it plus one return per four signal balls;
* signals sit in the outer three rows, where a through via escapes them;
* each interface owns a package edge, which is also the die-edge assignment
  for the I/O ring: link in west, link out east as two columns of 18 rows,
  both LPDDR5X channels on the north rows in byte lanes with a ground after
  each lane, management, JTAG, reference clock and mode strap on the south
  row, ground and the core rail in a checkerboard elsewhere;
* the smallest candidate package that satisfies all of that is selected.

At the design point of 14.5K tokens/s the core rail draws 54 A, which is 108
core balls, 108 ground balls, 54 signal returns, 216 signals and 34 rail
balls: 520 in all. The 400-ball package in the first draft cannot carry that,
so the selection is a 784-ball 28 × 28 array at 0.8 mm in the same 23 mm
body. At 50K tokens/s the core needs 354 balls each for power and ground and
the selection grows to a 35 × 35 array in a 29 mm body, which the current
floorplan does not fit (the generator says so rather than placing it). The
package is therefore a rating decision before it is a mechanical one.

The KiCad generator builds every ASIC footprint from this map, so the ball
pitch, the escape geometry, the via-in-pad counts and the ring's lane pitch
follow it. `hw/pinout/report.md` lists the candidates with the reason each
was rejected or chosen, the ball counts by use, and the 41 memory signals
that sit deeper than the outer three rows and need a build-up escape.

## KiCad project

`python -m hw.kicad_gen` turns `board.yaml` into a KiCad 7 project under
`hw/kicad/` (`appliance.kicad_pro`, `.kicad_pcb`, `.kicad_sch` with four
sub-sheets, `report.md`, `floorplan.svg`). It regenerates from the YAML, so
the description stays the source of truth. The files are written directly in
KiCad's S-expression formats; the generator itself needs only PyYAML, and
KiCad is used to open and check the result. The tests run KiCad's own
design-rule check when its `pcbnew` Python module is installed (`apt install
kicad` on Ubuntu 24.04) and pass with no violation other than the unrouted
nets.

![Floorplan](kicad/floorplan.svg)

What the project contains:

* the 312 × 111 mm full-length card with the x8 edge fingers, the bracket
  keep-out, the SFP+ cage and the 12V-2x6 auxiliary connector;
* all 33 parts of `board.yaml` on generated footprints, plus one core
  regulator block per ASIC and seven shared-rail regulator blocks. Row A
  runs left to right and row B is rotated 180 degrees so the ring snakes
  back; the memories sit outside the rows, the regulators between them,
  and the two head ASICs under the FPGA so the closing hop stays short;
* the ASIC ball map derived by `hw/pinout.py` (see above), shared by all ten
  ASICs, with the head ASICs' memory balls unconnected. The FPGA's link pins
  are assigned by the ribbon router; its other pins are placeholders;
* every net at the signal level (1,744 nets), so schematic and board agree:
  ring links, sixteen LPDDR5X channels, the DDR4 x64, PCIe, management SPI,
  JTAG chain, reference clocks, mode straps and rails;
* a twelve-layer stackup with four ground planes, a 12 V and I/O rail layer
  and a core-rail layer, via-in-pad on every ground and rail ball a zone can
  reach, and ground, 12 V, memory-PHY and core-rail zones;
* the activation ring fully routed: each straight hop is one 36-lane ribbon
  on In2.Cu, and each bent hop is split by escape depth into two 18-lane
  ribbons on In2.Cu and In5.Cu that taper from the 1 mm via pitch to 0.25 mm
  in their body so the corners stay cheap. Every hop is within the 80 mm
  link limit; the longest are the closing hop to the FPGA at 77 mm and the
  row change at 75 mm.

What it does not contain, on purpose: routed memory, PCIe or management nets
(length matching and signal integrity are interactive work), decoupling,
regulator internals, thermal vias, mounting holes, or a real ASIC ball map.
`report.md` carries the link lengths, the escape density (the 130 memory
signals per ASIC north edge need a build-up layer pair to escape), and the
core-rail current density: at 50K tokens/s a layer ASIC draws about 180 A,
which is over 100 A/mm² on a single 2 oz plane across the package width, so
the real board needs the regulator against the package with several plane
layers between them.

Two floorplan findings came out of routing the ring. The first draft put the
head ASICs at the bracket end and the closing hop came to 85 mm, over the
link limit; moving the heads under the FPGA fixed it. The row-change hop
between A3 and A4 only meets the limit with the rows 31 mm apart, the lanes
compressed to 0.25 mm and the ribbon split over two layers; a single-layer
ribbon at the via pitch was 105 mm. If the single-ended 1.8 V link keeps its
80 mm budget, those constraints are real; the LVDS fallback in `board.yaml`
would relax them.

## Open items before schematic entry in an EDA tool

1. ASIC package and ball map: the derived map in `hw/pinout/` goes to the
   packaging house with the bump map once the die floorplan exists; the
   substrate design and the final map come back from them.
2. Measured energy per MAC on the target process, which sets the core
   regulator sizing and the auxiliary connector; the 3 pJ in `board.yaml` is
   a projection from the ASAP7 signoff run.
3. Link I/O standard: 1.8 V single-ended at 500 Mb/s per pin needs a signal
   integrity check at 80 mm; fall back to eight LVDS pairs at 2 Gb/s if it
   fails.
4. LPDDR5X versus GDDR6. The simulator wants 75 to 100 GB/s per ASIC with
   int4 KV; two LPDDR5X-9600 x32 devices give 77 GB/s raw. GDDR6 doubles that
   at higher PHY effort and power.
5. FPGA part selection against the pin budget in `board.md`: two links, DDR4
   x64, PCIe x8, SPI with ten chip selects, JTAG, optional SFP+.
