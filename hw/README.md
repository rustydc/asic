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

## Form factor: a host-attached 1U

The board is a single 420 × 360 mm board in a 1U rack chassis, attached to
a host over a PCIe cable. The first draft was a full-length PCIe card; the
energy model below moved it out of the slot. At 3 pJ per MAC the board
draws 0.5 to 1.6 kW of input across the throughput range, which is a rack
power supply and a row of fans, not a 600 W auxiliary connector on a
passively cooled card. The 1U also has the room the 50K tokens/s package
needs (29 mm bodies on a 230 mm ring) and a management controller. The
chassis layout:

* rear panel: the SlimSAS 8i host connector next to the FPGA, the BMC's
  RJ45, the optional SFP+ cage, the status LEDs;
* rear right: the PSU bay with two CRPS modules in a 1+1 redundant pair;
* the activation ring is a regular polygon in the left two thirds of the
  board, the FPGA at its rear vertex, the shared regulators and the clock
  generator in its middle;
* front: six 40 mm fans blowing front to back across the ring.

## Population

| Part | Qty | Role |
| --- | ---: | --- |
| Layer ASIC | 8 | four decoder layers each, personalized coefficient masks, layer-mode strap |
| Head ASIC | 2 | same die, head-mode strap, each holds half of the 248K-row LM head; no memory |
| LPDDR5X 16 Gb x32 | 16 | two per layer ASIC: 4 GB and 77 GB/s raw per ASIC at 9600 MT/s |
| Controller FPGA | 1 | PCIe Gen4 x8 endpoint over the host cable, ring source and sink, sampling, scheduling |
| DDR4 x16 | 4 | 4 GB on the FPGA for the embedding table and per-context metadata |
| Clock generator | 1 | reference clock to all eleven devices |
| BMC | 1 | AST2600-class: PSU PMBus, fan control, thermal, FPGA console, its own Ethernet |
| Host cable connector | 1 | SlimSAS 8i (SFF-8654): PCIe Gen4 x8 plus sideband to a host adapter card |
| CRPS receptacle | 2 | 12 V from each supply module, 1+1 redundant |
| Fan header | 6 | 4-pin PWM, one per front fan |
| SFP+ cage | 1 (optional) | standalone 10GbE operation, rear panel |

## Topology

One unidirectional activation ring:

```text
FPGA -> A0 -> A1 -> A2 -> A3 -> A4 -> A5 -> A6 -> A7 -> H0 -> H1 -> FPGA
```

Each hop is the 32-bit source-synchronous DDR link from the architecture doc
(36 signals, 2 GB/s raw, under 80 mm). Head chips forward the incoming hidden
vector unchanged and append their top-k logits and partial softmax sum, so no
broadcast path is needed. The FPGA merges the two lists and samples.

On the board the eleven ring nodes sit on the vertices of a regular 11-gon,
clockwise from the FPGA at the rear, each chip rotated so its link-in edge
faces the previous chip and its link-out edge the next. Every hop is then
the same short ribbon. Each layer ASIC's two LPDDR5X devices sit on its
outer edge and its core regulator on its inner edge, rotated with it. All
eleven nodes are on the polygon, heads and FPGA included, because a chip
takes the ring in on one edge and out on the opposite edge: a loop that
dipped into the middle for the heads would have to hairpin back out.

Management is a SPI bus from the FPGA with one chip select per ASIC, a JTAG
chain through all eleven devices, and a fanned-out reference clock. Each ASIC
has a two-pin mode strap (layer or head). The BMC owns the chassis: PSU
PMBus, fan PWM and tach, the core regulators' PMBus, the FPGA's UART
console, and its own Ethernet port.

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
budgets (1 W per LPDDR5X, 25 W FPGA, 5 W BMC, 12 W per fan).

At the simulator's 14.5K tokens/s ceiling that is 43 W per layer ASIC and
517 W of load, 608 W of 12 V input at 85 percent regulator efficiency.
`board.md` tabulates the same at 5K, 25K and 50K tokens/s: the board runs
from about 330 W to 1.66 kW of input across the throughput range, and a
layer ASIC's core rail goes from 21 A to 177 A at 0.8 V. The supply is
therefore two 2000 W CRPS modules in a 1+1 redundant pair, one of which
carries the whole load, and each ASIC needs a multiphase core regulator.
A 7 nm-class die at about 1 pJ per MAC would put the whole range back under
600 W and make the card viable again; the process choice is the biggest
lever on the power supply and the enclosure.

The power tree is a 12 V intermediate bus from the CRPS modules into
point-of-load regulators. Each ASIC gets its own core buck with PMBus
telemetry so per-chip current is observable during bring-up. The LPDDR5X
rails, 1.8 V I/O, FPGA rails, DDR4 rails, and 3.3 V housekeeping are shared.

## Host interface: PCIe over a cable, and why not Ethernet or PoE

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
  latency target, while a network hop adds tens of microseconds and jitter.

The 1U is host-attached: a SlimSAS 8i cable from the rear panel to a
retimer or host adapter card in the server next to it, the same arrangement
as an external PCIe expansion chassis. PCIe Gen4 x4 is enough bandwidth; the
board wires x8 because the FPGA's hard block is x8 and the cable carries it.

The SFP+ cage on the rear panel keeps the standalone-appliance option open.
The FPGA transceivers are there anyway, so a later firmware can serve a UDP
or gRPC-style token protocol over 10GbE with no board change, and the
chassis already has its own supply and management.

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

The package is rated at the 50K tokens/s target, where the core rail draws
177 A: 354 core balls, 354 ground balls, 54 signal returns, 216 signals and
34 rail balls, 1012 in all. That selects a 1225-ball 35 × 35 array at
0.8 mm in a 29 mm body. The 784-ball 23 mm package of the card draft is
enough at the 14.5K design point (520 balls) and is what the card form
factor still uses; 400 balls never are. The rating is a decision: it fixes
the package before the die exists, and the 1U was retargeted partly so the
larger package fits.

The KiCad generator builds every ASIC footprint from this map, so the ball
pitch, the escape geometry, the via-in-pad counts and the ring's lane pitch
follow it. `hw/pinout/report.md` lists the candidates with the reason each
was rejected or chosen, the ball counts by use, and the memory signals that
sit deeper than the outer three rows and need a build-up escape.

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

* the 420 × 360 mm 1U board with its keep-outs (PSU bay, rear I/O strip,
  fan row), the SlimSAS host connector, the BMC and its RJ45, the two CRPS
  receptacles, six fan headers and the SFP+ cage, all placed physically and
  checked against the keep-outs;
* all 44 parts of `board.yaml` on generated footprints, plus one core
  regulator block per ASIC and seven shared-rail regulator blocks. The
  eleven ring chips sit on a regular 11-gon of 46 mm side (82 mm
  circumradius), each rotated tangentially, with its memories on the
  outer edge and its core regulator on the inner edge rotated with it;
  the FPGA's DDR4 and their regulators sit on its outer edge at the rear,
  the shared regulators and the clock generator in the middle of the
  polygon. Footprints are written with every pad and outline point
  already rotated, so any angle works;
* the ASIC ball map derived by `hw/pinout.py` (see above), shared by all ten
  ASICs, with the head ASICs' memory balls unconnected. The FPGA's link pins
  are assigned by the ribbon router; its other pins are placeholders;
* every net at the signal level (1,754 nets), so schematic and board agree:
  ring links, sixteen LPDDR5X channels, the DDR4 x64, PCIe to the host
  connector, management SPI, JTAG chain, reference clocks, mode straps,
  the BMC's fan, PMBus and console nets, and the rails;
* a twelve-layer stackup with four ground planes, a 12 V and I/O rail layer
  and a core-rail layer, via-in-pad on every ground and rail ball a zone can
  reach, and ground, 12 V, memory-PHY and core-rail zones;
* the activation ring fully routed: every hop is one 36-lane ribbon on
  In2.Cu at half the ball pitch that leaves and enters the ports straight
  and bends twice, by up to 23 degrees, in between. The longest lane of
  any hop is 27 mm against the 80 mm link limit, and all eleven hops are
  alike. The side is the package body plus 17 mm; 15 mm still passes
  DRC, and below that the ports are too close for a ribbon to bend.

What it does not contain, on purpose: routed memory, PCIe or management nets
(length matching and signal integrity are interactive work), decoupling,
regulator internals, thermal vias, mounting holes, chassis mechanicals, or
a real ASIC ball map. `report.md` carries the link lengths, the escape
density (the 130 memory signals per ASIC north edge need a build-up layer
pair to escape), and the core-rail current density: at 50K tokens/s a layer
ASIC draws about 177 A, which is over 80 A/mm² on a single 2 oz plane across
the package width, so the real board needs the regulator against the package
with several plane layers between them.

Why a polygon and not the two-row snake of the earlier drafts:

* The snake had two hops that carried all the risk, the row change and the
  closing hop into the FPGA, at 76 and 74 mm against the 80 mm limit with
  the 29 mm package, and only with the lanes compressed to 0.25 mm and
  split over two layers. On the polygon the longest lane anywhere is
  27 mm on one layer at the natural lane pitch, so the single-ended 1.8 V
  link has its margin back and the LVDS fallback is no longer needed for
  length.
* Airflow: in the snake each row of four sat in series in the front-to-back
  air, so the rear chip of a row breathed air heated by three 140 W
  packages. On the polygon the vertices are staggered and no chip sits
  directly behind more than one other.
* The chips are rotated by multiples of 32.7 degrees, so the memory escape
  routing and the regulators sit at angles to the board axes. KiCad and
  the fab do not mind; the memory length matching is no harder than on
  axis.
* The middle of the polygon (110 mm across) holds the clock generator,
  equidistant from every chip, and the shared regulators. The right third
  of the board is free apart from the PSU bay, so a shorter chassis is
  possible.

## The high-end variant: four layer ASICs and one head, HBM in the package

`board_27b.yaml` describes the 27B-class appliance on a 2 nm-class die
(`sim/config/qwen35_27b_2nm_hbm.json`): four layer ASICs of sixteen layers
each (5.71B coefficients, about 152 mm² of fabric), one head ASIC holding
the whole LM head at 42 percent of the same die, and an HBM4 stack on each
layer ASIC's interposer, so there is no memory on the board. The same
tools take it as a source (`--source hw/board_27b.yaml`) and write
`board_27b.md`, `board_27b.svg`, `pinout_27b/` and `kicad_27b/`:

* five via personalisations per model instead of ten, and one product
  family with the 9B if that moves to the same eight-layer die;
* at the rating of 234K tokens/s (the fabric ceiling) the board draws
  2.3 kW of load, 2.7 kW of input from a 1+1 pair of 3000 W CRPS modules,
  and 525 W per layer die; at the 50K target it is a quarter of that;
* the package is a 55 × 55 array at 1.0 mm (3,025 balls, 2,778 needed):
  656 A on the core rail at the rating, 1,312 core balls, no memory balls,
  the north edge free for the HBM PHY towards the stack;
* the ring is a regular hexagon of 85 mm side with 10 mm between the
  packages for the cold plates; every hop is under 65 mm with bends of
  30 to 42 degrees, and pcbnew DRC reports only the unrouted nets;
* the regulators follow the power tree: one block per shared rail (I/O,
  HBM core, HBM I/O) in the middle of the hexagon, the FPGA's rails on one
  block, DDR4 rails and 3.3 V beside the DDR4 row.

Cooling is direct-to-chip liquid. 525 W over a 55 mm package is about
175 W/cm² at the lid and 250 to 300 W/cm² at the die, which a lidless
microchannel cold plate handles with about 25 K of rise at 1 to 2 l/min
per plate; through a lid and a conventional cold plate the junction lands
near 95 °C, which is marginal. The board therefore assumes bare-die cold
plates, a rear quick-disconnect pair, coolant flow and leak sensing on the
BMC, and fans only for the regulators, DDR4 and FPGA. The 12 V input at
2.7 kW is 225 A across the board, and the per-die core rail at 656 A wants
the multiphase regulator on the package substrate or backside power
delivery on the die; the 26 mm regulator block on the board is a
placeholder for that decision, not a design.

## Open items before schematic entry in an EDA tool

1. ASIC package and ball map: the derived map in `hw/pinout/` goes to the
   packaging house with the bump map once the die floorplan exists; the
   substrate design and the final map come back from them.
2. Measured energy per MAC on the target process, which sets the core
   regulator sizing and the supply rating; the 3 pJ in `board.yaml` is a
   projection from the ASAP7 signoff run.
3. Link I/O standard: 1.8 V single-ended at 500 Mb/s per pin needs a signal
   integrity check at 80 mm; fall back to eight LVDS pairs at 2 Gb/s if it
   fails.
4. LPDDR5X versus GDDR6. The simulator wants 75 to 100 GB/s per ASIC with
   int4 KV; two LPDDR5X-9600 x32 devices give 77 GB/s raw. GDDR6 doubles that
   at higher PHY effort and power.
5. FPGA part selection against the pin budget in `board.md`: two links, DDR4
   x64, PCIe x8, SPI with ten chip selects, JTAG, optional SFP+.
6. Host side: the retimer or adapter card and the cable length the Gen4
   link tolerates; the fallback is a Gen3 cable or a redriver at the
   rear panel.
7. Thermal: 1.4 kW of load in 1U at 50K tokens/s is at the limit of
   front-to-back air over 29 mm packages even with the staggered polygon;
   the 25K row (0.8 kW) is the comfortable air-cooled point.
