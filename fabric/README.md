# Fixed-weight fabric tile

The fabric is the part of the die that holds the model. This directory defines
its unit, the **tile**: a via-programmed coefficient ROM feeding a row of
multiply-accumulate columns. The reference model, the coefficient compiler,
the synthesizable RTL, and the model-to-fabric mapping all live here, and the
RTL is checked bit-exact against the model with Icarus Verilog.

```bash
python -m fabric.tile                        # tile budget for the 9B geometry
python -m fabric.tile --geometry qwen3_5_4b  # and the 4B
python -m unittest discover -s fabric/tests -t .   # model, mapping, and RTL tests
```

## Why a ROM and not pure wiring

A coefficient that is literally a wire between an input line and an
accumulator can never be time-multiplexed, because a static wire cannot select
a different multiple on a different cycle. Pure wiring therefore means one
adder leaf per coefficient, all 866 million of them used once per token. That
is not buildable. The only way a mask-programmed coefficient can share
arithmetic is for the mask to encode a *bit* that a wordline reads out, which
is a ROM.

So the tile is a mask ROM with a very wide read port and the arithmetic sitting
against its bitlines. The architecture doc's "no addressed weight store" is
kept in the sense that matters: the coefficient never leaves the tile, the
read is a few tens of micrometres of bitline, and there is no weight bus,
cache, or DRAM traffic. What is given up is the idea that the weights are
wiring. They are a via pattern in a ROM array, which is denser anyway.

## Tile microarchitecture

```text
                 activations x[r], x[r+1]   (P = 2 rows per cycle, int8)
                        |
          +-------------v--------------+
          | shared multiples per bank  |   0, x, 2x, 3x, 4x, 5x, 6x, 7x
          | (shifts + three adders)    |   11-bit signed
          +----+----+----+---- ... ----+
               |    |    |               one 8-way bus per bank
   ROM row r   v    v    v
   +----+   +--------------------------------------------------+
   |via |-->| column c: select multiple by |w[r][c]|, add/sub  |--> acc[c]  (24-bit)
   |ROM |   | column c+1 ...                                   |
   |    |   +--------------------------------------------------+
   4096 rows x 64 columns x 4 bits = 1 Mbit, 2 banks (even / odd rows)
```

| Parameter | Value | Note |
| --- | ---: | --- |
| Rows (input dims per tile) | 4096 (9B), 2560 (4B) | equals the hidden width; one pass covers a full input vector |
| Columns (outputs per tile) | 64 | |
| Coefficient format | symmetric int4, −7..7 | never −8, so the magnitude selects one of 8 multiples |
| Activation format | int8 | |
| Rows per cycle | 2 | two ROM banks read in parallel; the main area-versus-latency knob |
| Accumulator | 24-bit | worst case 4096 × 127 × 7 fits with margin |
| Requantization | `sat8((acc × mult + 2^(shift−1)) >> shift)` | per-column 16-bit multiplier and 5-bit shift from SRAM, not from the fabric |
| Pass length | rows / rows-per-cycle = 2048 cycles | |

A pass starts by loading the accumulators with chained partial sums (zero for a
fresh pass), consumes two activations per cycle in row order, and after 2048
cycles the shared requantizer walks the columns and publishes the resolved
accumulators and the requantized int8 outputs together. Matrices
wider than 64 outputs use more tiles side by side. Matrices deeper than the
tile, such as the 12288-wide FFN down-projection, use three tiles over
disjoint row ranges in parallel and a partial-sum reduction after them, so
every pass takes the same 2048 cycles.

The per-column requantization constants are the only per-model numbers that
are not in the via pattern. They live in SRAM loaded at boot over the
management SPI, about 80 KB per die.

## Personalization

Every coefficient bit is the presence or absence of one via in the ROM array.
Ten chips, ten via masks, one base mask set. The base wafers can be processed
up to the via layer and banked, then personalized. `Tile.via_count` and
`via_coordinates` produce the pattern the GDS writer places; the row and bit
addressing is fixed here, the bit-cell geometry comes from the foundry ROM
compiler.

Verification per variant is: DRC on the via layer, `decompile` of the via maps
against the quantized weights, and `tile_forward` against `reference_matmul`
on random vectors. No timing closure per variant, because the ROM read and
the column datapath do not depend on which vias are present.

## Mapping the models

The mapping counts tiles per matrix for one `R,R,R,G` shard and for a head
die holding half the vocabulary. All numbers use the placeholder density model
below.

| | 9B layer die | 9B head die | 4B layer die | 4B head die |
| --- | ---: | ---: | ---: | ---: |
| Tiles | 3306 | 1940 | 2858 | 1940 |
| Coefficients | 866M | 509M | 447M | 318M |
| Utilization | 99.9% | 100% | 97.7% | 100% |
| Area at 2 rows/cycle | 242 mm² | 142 mm² | 176 mm² | 119 mm² |
| Latency per layer, 800 MHz | 10.2 µs | 2.6 µs (die) | 6.4 µs | 1.6 µs (die) |
| Fabric energy per token | 62 µJ | 37 µJ | 32 µJ | 19 µJ |

The head slice fits inside a layer die's tile count in both geometries, which
is what makes the head-mode personalization possible. At 50K tokens/s the
fabric energy is about 3 W per 9B layer die before clocks, memory, and I/O.

The rows-per-cycle knob trades column area for latency. ROM area is fixed by
the coefficient count; the MAC columns scale with rows per cycle.

| Rows/cycle | Clock | 9B layer die area | ROM / MAC | Latency per layer | Pass |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 800 MHz | 218 mm² | 104 / 106 mm² | 20.5 µs | 5.1 µs |
| 2 | 800 MHz | 242 mm² | 104 / 130 mm² | 10.2 µs | 2.6 µs |
| 4 | 800 MHz | 291 mm² | 104 / 179 mm² | 5.1 µs | 1.3 µs |
| 2 | 500 MHz | 242 mm² | 104 / 130 mm² | 16.4 µs | 4.1 µs |

Two rows per cycle is the baseline. The simulator's stage times are derived
from it: 102 ticks (10.2 µs) per 9B layer, 26 ticks per head die.

Because a layer is four sequential passes, a layer stage could accept a new
token every pass (2.6 µs) while a token takes four passes to get through. The
simulator currently treats a layer as one stage with initiation interval equal
to latency, so it understates fabric throughput by up to 4×. It does not
matter yet because the global stage's memory traffic is the bottleneck at
about 66 µs.

## Density model

`DensityModel` holds the numbers behind the tables. The MAC-column entries
are calibrated from open-tooling synthesis (next section); the rest are
placeholders the MPW tile is meant to replace.

| Entry | Value | Source |
| --- | ---: | --- |
| ROM area | 0.03 µm² per bit | placeholder: via-ROM compiler cell at the target node |
| MAC column | 115 µm² per bank | 380 NAND2 equivalents from 64-column sky130 synthesis, at a 0.30 µm² 28 nm NAND2 |
| Accumulator, requantizer share, registers | 385 µm² per column | 1280 NAND2 equivalents, same source |
| Tile overhead | 2500 µm² | placeholder: multiples generator, ROM periphery, control |
| Clock | 800 MHz | placeholder: ROM read plus column add in one cycle |
| ROM read | 3 fJ per bit | placeholder |
| MAC | 60 fJ per coefficient | placeholder |

At these numbers a 28 nm-class 9B layer die is 242 mm², about 43 percent
ROM and 54 percent MAC columns. The timing-clean column (next sections)
costs about 45 percent more than the first ripple-carry version; that is the
price of a pipeline with no carry chain per cycle. At a 16 nm-class node expect roughly
half. If the ROM cell comes in denser than 0.03 µm² per bit, the MAC columns
dominate and one row per cycle becomes the better trade.

## Open tooling and open PDKs

Yes, the column datapath synthesizes with open tooling today, and the numbers
above already come from it. `fabric/synth.py` runs yosys (native or the
`yowasp-yosys` PyPI build) on `fabric_columns` against any liberty file:

```bash
pip install yowasp-yosys
curl -LO https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/master/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
python -m fabric.synth --liberty sky130_fd_sc_hd__tt_025C_1v80.lib --rows 256 --cols 16 --rows-per-cycle 2
FABRIC_LIBERTY=$PWD/sky130_fd_sc_hd__tt_025C_1v80.lib python -m unittest fabric.tests.test_synth
```

Measured first on the original ripple-carry column (16 columns, 256 rows,
two rows per cycle) on two manufacturable 130 nm libraries and two predictive
advanced-node kits:

| Library | Kind | NAND2 | Cells | Per column | NAND2-eq per column |
| --- | --- | ---: | ---: | ---: | ---: |
| SkyWater sky130 HD (130 nm) | manufacturable | 3.75 µm² | 12,087 | 5221 µm² | 1392 |
| IHP SG13G2 (130 nm) | manufacturable | 7.26 µm² | 12,027 | 9349 µm² | 1288 |
| NanGate 45 (FreePDK45) | predictive | 0.80 µm² | 12,999 | 1043 µm² | 1307 |
| ASAP7 (7 nm FinFET) | predictive | 0.058 µm² | 11,974 | 74 µm² | 1271 |

Four libraries spanning 130 nm to 7 nm agree within ten percent once
normalized to their own NAND2, so NAND2 equivalents are the number to carry.
The first synthesis run also caught a design error: a per-column requantizer
multiplier that tripled the column area, now a single unit shared across the
64 columns.

The timing work below reshaped the column (carry-save accumulate,
signed-digit taps, a deeper shared requantizer). A full 64-column tile on
sky130 now costs 1663, 2050 and 2808 NAND2 equivalents per column at 1, 2 and
4 rows per cycle, which is the 1280 + 380 per bank the density model uses.

ASAP7 ships its cells in several liberty files; merge them first:

```bash
python -m fabric.synth --merge asap7_tt.lib --liberty SIMPLE.lib --liberty INVBUF.lib \
    --liberty AO.lib --liberty OA.lib --liberty SEQ.lib
python -m fabric.synth --liberty asap7_tt.lib --rows 256 --cols 16 --rows-per-cycle 2
```

On the predictive kits: FreePDK15 (NC State, 15 nm FinFET) is the same kind
of thing as ASAP7 and NanGate 45, an academic model of a node with no fab
behind it. Its standard cells are the NanGate 15 nm library distributed by
Silvaco behind a registration, so it is not fetchable in a script the way the
OpenROAD platforms are, and ASAP7 already gives the below-28 nm bracket. Any
of them is fine for relative area, useless for a tapeout.

What open PDKs can and cannot do for this project:

* **Relative column cost: yes, done above.** Predictive kits (NanGate 45,
  ASAP7, FreePDK15) bracket the target node from both sides.
* **The MPW tile: yes.** A 1024 × 64 test tile on sky130 or IHP SG13G2 is
  about a square millimetre of columns plus a hand-drawn via-ROM array, and
  both processes run open shuttles. That measures the column datapath, the
  ROM cell, and the personalization flow end to end, at 130 nm.
* **Timing: yes, pre-layout and placed-and-routed.** OpenSTA builds from
  source in a few minutes and times the yosys netlist against the same
  liberty files; OpenROAD then places, buffers, clocks and routes it on
  the sky130 and ASAP7 platform files. See the next two sections.
* **The production die: no.** At 130 nm the ROM cell is 30 to 50 times larger
  than at 28 nm, so the 9B layer die would be several thousand square
  millimetres. The production node needs a foundry PDK under NDA.
* **OpenRPDK28 (RIOS Lab): not yet.** It is an academic 28 nm template
  under construction with device models and rule decks but, as far as its
  repository shows, no standard-cell library or liberty timing, and it is
  not tied to a foundry, so nothing built on it can be fabricated. It may
  become useful for checking a via-ROM cell against 28 nm-class design rules
  once it matures.

A further density lever, not in the baseline, is current-mode readout: drive
wordlines with the activation as a pulse width, sum bitline currents, and
digitize once per column. That removes the column datapath entirely and is
the only path well below 0.15 µm² per coefficient, at the cost of analog
precision risk. It is a second-generation experiment.

## Timing on 130 nm and 7 nm

`fabric/sta.py` runs OpenSTA on a netlist written by `fabric/synth.py`
(`--target-ps` for timing-driven mapping, `--netlist` to keep it), with a
clock constraint and ten percent input and output delays:

```bash
python -m fabric.synth --liberty sky130.lib --rows 256 --cols 16 --rows-per-cycle 2 \
    --target-ps 2500 --netlist net_sky130.v
python -m fabric.sta --sta /path/to/OpenSTA/build/sta --liberty sky130.lib \
    --netlist net_sky130.v --period-ps 2500 --report
```

Results for the column datapath at two rows per cycle, typical corner:

| Library | Critical path | Of which unbuffered fanout | Logic-depth limit |
| --- | ---: | ---: | ---: |
| sky130 HD, 130 nm | 4.25 ns | 1.4 ns (80-load flop) | 235 MHz, ~310 MHz buffered |
| ASAP7, 7 nm predictive | 0.73 ns | 0.14 ns (52-load flop) | 1.37 GHz, ~1.7 GHz buffered |

What these numbers are: post-synthesis, pre-layout, no wire load, no clock
tree, ideal clock, one corner, and no cell sizing or buffering beyond what
ABC does inside one combinational block. Flop-driven nets are not buffered,
so the first line of every critical path is a fat unbuffered fanout that
place-and-route would fix; the "buffered" column subtracts it and charges a
buffer tree instead. ASAP7 maps to its smallest cells throughout, so it is
pessimistic in the other direction. Budget 20 to 40 percent on top for wires.

Read across the nodes: 130 nm at roughly 300 MHz and predictive 7 nm at
roughly 1.7 GHz bracket a 28 nm-class part somewhere around 0.6 to 1.2 GHz
for this logic depth. The 800 MHz placeholder is inside that range, not
proven by it. The MPW tile and a foundry library settle it.

Getting here changed the design, and every change was driven by a reported
critical path:

1. **Requantizer multiplier per column** (3000 cells per column): made one
   shared unit that walks the columns after the pass.
2. **41-bit ripple-carry adds** in the requantizer at 6 to 9 ns on sky130:
   yosys emits ripple adders and ABC cannot restructure carry chains, so the
   multiply became a carry-save tree with the rounding constant folded in and
   the final add runs in three 14-bit chunks over three stages.
3. **Column select fanout** (`q_col` on 600 accumulator mux bits): replaced by
   a one-hot walking select, 69 loads per bit regardless of column count.
4. **Control strobes** merged back into one 600-load net by synthesis even
   with `keep`: the per-column copies are now their own module with
   `keep_hierarchy`.
5. **Pre-adders for 3x, 5x, 7x** on the activation path: every coefficient
   magnitude is now a signed pair of power-of-two taps, so stage A has no
   arithmetic at all.
6. **24-bit accumulate and the term sum**, ripple again: both are carry-save.
   The accumulator is a (sum, carry) pair with a constant-depth 3:2 reduction
   per cycle; the carries resolve once per pass in the requantizer walk, in
   two 12-bit chunks. The term sum reduces the four taps and the negation
   count to a carry-save pair at accumulator width, so no width change and
   no wrap can occur.
7. **Datapath resets** removed; `start` loads the accumulators, everything
   else free-runs, and only control keeps an asynchronous reset.

The column pipeline is now A (inputs), B (term reduction), C (accumulate);
the requantizer is select, two resolve stages, multiply, three add stages,
and shift/saturate, nine cycles of latency after the 64-cycle walk. All of
it stays bit-exact against the Python model, which never changed.

## Place and route

`fabric/pnr.py` drives OpenROAD through floorplan, timing-driven global
placement, resize and buffering, clock-tree synthesis, hold repair, global
routing, and post-route timing on wire parasitics extracted from the global
route. It uses the platform files from OpenROAD-flow-scripts (LEF, RC
tables, track scripts) and the same netlists OpenSTA timed above:

```bash
micromamba create -p eda -c litex-hub -c conda-forge openroad   # or build from source
git clone --depth 1 https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts orfs
python -m fabric.pnr --openroad eda/bin/openroad --platform sky130hd \
    --platforms-dir orfs/flow/platforms --netlist net_sky130.v \
    --period-ps 3000 --utilization 35 --work pnr_sky130
python -m fabric.pnr --openroad eda/bin/openroad --platform asap7 \
    --platforms-dir orfs/flow/platforms --netlist net_asap7.v \
    --liberty asap7_tt.lib --period-ps 700 --utilization 45 --work pnr_asap7
```

Results for the same 256-row, 16-column, two-rows-per-cycle slice, after
global routing, typical corner (`fabric/results/pnr_*.json` holds the parsed
numbers and the final critical path):

| | sky130 HD, 130 nm | ASAP7, 7 nm predictive |
| --- | ---: | ---: |
| Target period | 3.00 ns | 700 ps |
| Worst slack | −1.15 ns | −2.7 ps |
| Achievable clock | **241 MHz** | **1.42 GHz** |
| Pre-layout estimate | 235 MHz (310 buffered) | 1.37 GHz (1.7 buffered) |
| Clock insertion / skew | 1.9 ns, 8 levels / 470 ps | 280 ps, 4 levels / 73 ps |
| Hold buffers | 3051 | 1584 |
| Design area | 215,071 µm² (47 %) | 2,623 µm² (45 %) |
| Placed area before repair | 159,971 µm² | 2,646 µm² |
| Instances | 28,034 | 23,621 |
| Wirelength | 1.48 m | 131 mm |
| Routing overflow (global) | 297 edges, ≤ 6 per edge | 0 |

The pre-layout numbers hold up. On ASAP7 the routed design is a little
faster than the unbuffered estimate: the resizer fixes the fanout the STA
run flagged, and wires, the clock tree and 73 ps of skew cost less than
that fix gains. On sky130 the routed flop-to-flop path is 4.2 ns, the same
as the unbuffered pre-layout path, so none of the expected buffering gain
arrived (see below); the 2345-sink clock tree built from `clkbuf_4` alone
is eight levels and 1.9 ns deep, but its 470 ps of skew happens to favour
the worst path. The 28 nm-class bracket stays where the previous section
put it, roughly 0.6 to 1.2 GHz, with the 800 MHz placeholder inside it.

What limits each node now:

* **ASAP7:** the critical path is twenty-seven XOR and NAND levels in the
  requantizer, the ripple form ABC leaves a 14-bit chunk add in, plus the
  skew between two clock branches. Both are design choices: narrower
  chunks or a conditional-sum adder should be worth 100 to 150 ps, and a
  two-level clock mesh removes most of the skew. Neither is worth doing
  before the target library is known.
* **sky130:** the critical path runs through the walking select and the
  requantizer input mux, and one `o41ai_4` cell on it drives a 64 fF net
  at a 0.9 ns slew that the resizer gave up on (`RSZ-0062`, three times).
  That one cell is 0.85 ns of a 4.2 ns path. It is a buffering failure
  rather than a logic-depth limit; a hand-placed buffer tree on the select
  would recover most of the missing 300 MHz. It only matters for the MPW
  tile.

Area moves in the expected direction. Sky130 grows 34 percent in place and
route, almost all of it the 3051 hold buffers that propagated-clock skew
forces on a design whose datapath registers have no logic between them in
the walk, plus the clock tree. ASAP7 does not grow at all because the
resizer downsizes 10,390 cells that yosys had mapped larger than needed,
which pays for its 1584 hold buffers. Per column slice the routed area is
3585 NAND2 equivalents on sky130 and 2827 on ASAP7, against 2050 for a
column of the 64-wide tile pre-layout; the 16-column slice amortizes the
shared requantizer over four times fewer columns, which accounts for most
of the difference. The density model still carries the pre-layout
64-column figure. The sky130 run says place and route can add a third on
top of it, mostly hold buffers a better clock tree would reduce, and the
ASAP7 run says it can add nothing; the die-size table should be read with
that spread in mind until the target library settles it.

What this run is not: detailed routing (`--detailed-route` runs it, at
several times the wall time), a signoff extraction, multiple corners, or a
power grid. The global router reports no overflow on ASAP7 and a few
hundred lightly overfull edges on sky130 at 35 percent target utilization,
so both would detail-route; the sky130 slice is pin-limited, with 1390
ports on a 0.68 mm square, which is an artifact of routing a slice rather
than the whole tile.

Two things the flow had to work around in the litex-hub OpenROAD build of
February 2024: `report_wire_length` crashes on a typo in its error path,
so wirelength comes from the router's own log line, and `> file`
redirection on several report commands writes empty files, so every report
goes to the log and `parse_results` reads that.

## Detailed routing, power grid and IR drop

`--detailed-route` extends the flow: well taps and the platform's power
grid go in at floorplan (`pdngen` with the OpenROAD-flow-scripts strategy,
M1/M2 rails with M5/M6 stripes on ASAP7, met1 rails with met4/met5 stripes
on sky130), and after global routing it runs TritonRoute, fill cells,
OpenRCX extraction, timing on the extracted parasitics, `report_power`,
and static IR-drop analysis of both supply nets.

```bash
python -m fabric.pnr --openroad eda/bin/openroad --platform asap7 \
    --platforms-dir orfs/flow/platforms --netlist net_asap7.v \
    --liberty asap7_tt.lib --period-ps 700 --utilization 25 \
    --detailed-route --work pnr_asap7_drt --json
```

ASAP7, same 256-row 16-column slice, after detailed routing and
extraction (`fabric/results/pnr_asap7_signoff.json`):

| | ASAP7, 7 nm predictive |
| --- | ---: |
| Detailed-route DRC violations | 0 (8 iterations); 0 antenna violations |
| Timing on extracted parasitics | 700 ps met exactly, **1.43 GHz** |
| Same path on global-route estimates | −8 ps |
| Clock skew | 75 ps |
| Cells (before 38,086 fillers) | 26,433 |
| Design area | 3,311 µm² at 31 % of a 107 µm die |
| Wirelength | 157 mm |
| Power (default 10 % activity, 1.43 GHz) | 17.9 mW: 61 % internal, 39 % switching |
| Static IR drop, VDD | 364 mV worst at the core edge, 12 mV average |

The timing story does not change from global routing to signoff: the
extracted parasitics come in slightly under the global-route estimate, and
the slice closes at exactly the 700 ps it was asked for. The pre-layout
bracket for 28 nm stands.

The power grid is the new information. The slice draws 1.7 W/mm² at the
default activity, which is high, and the platform's default grid was not
made for that: 18 nm M1/M2 rails fed from M5/M6 stripes every 5.4 µm lose
100 mV or more on half of the cells, and 364 mV at the right core edge,
where the last stripe sits 3 µm from the boundary. That result is with a
supply point on every top-layer node (this build turns `-dx 20 -dy 20`
into 4.2 million sources), so it is the distribution loss alone; a real
bump grid adds to it. For the tile this says two things: stripe pitch and
rail width are first-order design parameters at this power density, and
the density model's 800 MHz at 28 nm will need a grid sized from the
measured current, not the platform default. It does not change the area
model.

Getting a pin-dense slice through this build's TritonRoute took some
care, and each item is a flag or an experiment in `fabric/pnr.py`:

* constants must be tie cells (`hilomap` on the mapped netlist, with
  `opt_clean -purge` and `setundef -zero` first) or the router refuses
  the supply-typed nets they become;
* buffers inserted by post-route timing repair must be legalized and the
  design re-routed before detailed routing;
* on ASAP7 the router could not reach a handful of M4/M5 I/O pins on the
  left and bottom edges whatever the stub length or spacing (experiments
  A and B); with 1 µm stubs on the top and right edges only (experiment
  C) it routes cleanly;
* `pin_access` before global routing puts guides on M9 on ASAP7 and is
  therefore off there;
* the routing derating follows the flow scripts (0.25 on ASAP7, 0.2 on
  sky130): at 0.5 the slice overflowed by a few edges and the guides of
  those nets were unusable;
* on sky130 the router rejected every net on a pin whose li1 and met1
  shapes are separate LEF `PORT`s (xor2 B, dfrtp RESET_B, mux, fa) when
  the ports straddle a gcell boundary; the driver merges those ports in
  a copy of the cell LEF, and the resizer is told not to use the probe
  cells it had been picking as hold buffers.

sky130 signoff: not closed. The full slice at 35 percent utilisation and a
3 ns clock global-routes cleanly (`results/pnr_sky130hd.json`), but the
detailed router did not converge: from 59K violations after the first
iteration it reached floors of 826, 611 and 547 (iterations 23, 32 and 40),
each followed by a rip-up phase every eight or nine iterations that reopened
the same regions (spikes to 3.4K, 1.7K, 1.3K and 2.1K), and it sat at 1,219
from iteration 50 to 53 when the run was stopped after 28 hours. The
residual violations are concentrated, not spread, which points at pin
access on a few sky130 cells with the merged-port LEF rather than
congestion; the next attempts are a lower utilisation (25 percent), the
unmerged LEF with those cells on the dont-use list, or a smaller slice.
The ASAP7 signoff is the clean reference for the column datapath; sky130
is the MPW target and this is the open item for it.

## IHP test chip with HPI PSRAM

The first silicon target is IHP SG13G2 through its open-source MPW
(2,800 EUR per mm², about six months, 40 bare dies, QFN packaging up to
64 pins), with one AP Memory HPI x16 PSRAM per chip as the off-chip
memory. The open SG13G2 pad ring is 3.3 V only, so the PSRAM runs in its
3 V mode: 133 MHz DDR on a 16-bit bus, about 530 MB/s, 64 MB.

That bandwidth, not the capacity, sets the context. Retrieval traffic per
token in one global layer is an index scan that grows with context plus a
selected-KV transfer that does not (32 blocks of 16 positions, 4 KV heads of
256), so cutting context alone gains little; the levers in order are int4
KV at 16:1 compression, fewer retrieved blocks, then context:

| Context | KV, compression | Blocks | Per token | Tokens/s on one device | Contexts per device |
| ---: | --- | ---: | ---: | ---: | ---: |
| 128K | int8, 4:1 | 32 | 3.1 MB | 135 | 1 |
| 128K | int4, 16:1 | 32 | 1.0 MB | 400 | 8 |
| 128K | int4, 16:1 | 8 | 0.66 MB | 650 | 8 |
| 32K | int4, 16:1 | 8 | 0.26 MB | 1,600 | 30 |
| 16K | int4, 16:1 | 8 | 0.20 MB | 2,150 | 60 |
| 4K | int4, 16:1 | 8 | 0.15 MB | 2,900 | 240 |

(80 percent sustained bandwidth; the 128K rows are the design point of the
appliance for comparison.) A 16K to 32K context with int4 KV at 16:1 and 8
retrieved blocks gives one to two thousand tokens per second per global
layer on one device, which is enough to exercise the retrieval path end to
end; the local window must shrink with it, because 512 positions of KV is
512 KB and the die cannot hold that in IHP SRAM (about 1 mm² per 16 KB),
so a 64-position window in on-chip SRAM or the window in the PSRAM too.

Pin budget for a QFN64 with about 52 signal pins: HPI x16 20 (DQ0-15, two
DQS, CLK, CE), the ring link narrowed to 8 data bits (12 signals each way,
24), management SPI 5, reference clock 2, a strap: 52. JTAG goes; the SPI
carries test access. The narrow link at 133 MHz DDR moves 266 MB/s, which
at a 4 KB int8 activation is 65K tokens/s, far above the memory-bound rate.

Die: at 130 nm the ROM bit is roughly 0.65 µm² and a tile's 64 columns
about 0.6 mm², so a full-depth 4096 by 64 tile is about 1.5 mm² and the
chip with two tiles, the HPI controller, a small retrieval engine, SRAM
and the pad ring is 6 to 8 mm², 17 to 22 k EUR of silicon at the open
rate. A reduced tile (256 rows by 16 columns) fits a Tiny Tapeout slot on
IHP for a few hundred dollars and proves the via ROM, the column datapath
and the requantiser, but not the memory path; the full chip is the one that
tests HPI retrieval.

## The rest of the layer

The tile evaluates `y = Wx`. Everything else in a decoder layer is
element-wise or per-head arithmetic on a few thousand values between the
fabric passes, and `layer.py` with `rtl/fabric_vector.sv`, `fabric_norm.sv`,
`fabric_recurrent.sv`, `fabric_ffn.sv` and `fabric_attention.sv` is that
datapath: the bit-exact integer model, the compiler that turns a layer's
float weights and a calibration into the units' constants, the float
reference the integer layer is checked against, and the RTL of every unit.

```text
recurrent layer          fabric pass         vector unit
  h (int16) ---------> rmsnorm ----------> int8 x
  x ---------------> [qkv | z | b | a] ----> conv+silu (8192 ch, 4 taps, history per context)
                                             q,k: L2 norm per head -> int8 unit vectors
                                             b, a: head gates -> beta, decay (U16)
                                             delta state (32 heads, 128x128 int16 S) -> y
                                             gated norm: y * silu(z) -> int8
  -------------------> [out_proj] ---------> residual add (int16)
  h ---------------> rmsnorm -> [gate | up] -> swiglu -> int8 -> [down] -> residual add
global layer
  x ---------------> [q,gate | k | v] -----> q,k: head norm (gain), rotary -> int8
                                             attention core: online softmax over K/V rows,
                                             sigmoid(gate) -> int8
  -------------------> [o_proj] -----------> residual add, then the FFN as above
```

| Format | Where | Definition |
| --- | --- | --- |
| residual `h` | between layers and across both residual adds | int16, one scale per layer |
| fabric activations | every tile input and output | int8, one scale per matrix |
| F16 | input of every nonlinearity, output of SiLU | int16, 10 fraction bits (Q5.10) |
| U16 | sigmoid, exp, softmax weights, decay, beta | unsigned Q0.16, 1.0 clipped to 65535 |
| normalised `n` | inside the norms | `x / sqrt(sum x^2) * 2^14`, int16 |
| recurrent state `S` | per head, per context, in the local memory | int16 at `s_v / 256` |
| requantizer | the end of every unit | `sat((v * mult + 2^(sh-1)) >> sh)`, 16-bit `mult`, 6-bit `sh` |

The nonlinearities are tables with linear interpolation: sigmoid over
[−8, 8) in 256 steps, exp(−t) over [0, 32) in 1024, softplus over [−16, 16)
in 2048, a sine over a turn in 1024, each within 1e-4 of the function
except softplus at 1e-3. The inverse square root and the reciprocal are a
seed from a 768- or 512-entry table over the normalised operand and one
Newton step, within 3e-5 and 6e-5. Nothing in the datapath divides.

Where a per-element weight follows a matrix it is folded into the matrix
before quantisation: the two RMS norm weights of every layer into the rows
of the projections they feed, the gated norm's weight into `out_proj`. The
head norms of the global layer sit between the projection and the rotary,
so their weight rides as a Q3.13 gain through the norm unit instead. The
per-head `exp(A_log)` is a Q6.10 constant and `dt_bias` an F16 one; the
head gates take the fabric's raw 24-bit accumulators for the one-column
`in_proj_a` and `in_proj_b`, so those 32-wide outputs never pass through
the int8 requantizer.

Two findings from building it:

* **The recurrent state cannot be a plain int8.** Round-to-nearest cannot
  apply a slow decay to a narrow value: `S * d` rounds back to `S`
  whenever `|S| < 1 / (1 - d)`, which at `d = 0.999` is every int8 value
  and the bottom three percent of int16. The first build kept the state
  int16 (1 MB per layer per token, read and written), and the sequencer's
  schedule then showed that traffic to be most of a token's time on the
  PSRAM board. The resolution is the int8 state with a scale, below. The
  state engine keeps one head on chip for the two passes of the update,
  so the memory sees each row once each way.
* **A cancellation head cannot survive int8.** On a random-init model a
  head whose delta output is a 1e-3 residual of its state flips sign in
  the integer layer, and the norm then amplifies the flip. Against a float
  reference carrying the same int4 weights the integer datapath tracks
  every intermediate (`test_layer.py`: 0.95 on the state output, 0.998 on
  the residual stream); against the unquantised weights the residual
  stream still holds at 0.99 while the mixer of such a head does not. That
  is the int4 weights' business, and the quantisation-aware training's.

### State traffic: int8 with a per-head scale

The state is stored as `S = g * T * 2^-e`: `T` int8 rows, `g` a U16
scale and `e` a small exponent per head (`e = 0` puts the LSB at `s_v`,
the top byte of the int16 state's range), with the largest `|T|` written
since the last rescale kept as `peak` and the number of saturated
elements written last token as `nsat`; the four ride in one beat ahead
of the head's rows. The decay multiplies the scale, not the stored
values, so it costs no rounding at all; when the scale falls below one
half the head is rescaled once (`T <- T * g * 2^de`, the only rounding
the state ever sees, every ~700 tokens at `d = 0.999`) and the scale
returns to one. The exponent moves to keep the byte in use: up one at a
rescale when the largest value written since the previous rescale was at
most 47 (the byte never more than a third used over the whole period),
and down one at once, any token, when more than one in sixty-four of the
elements saturated. Both rules were chosen by sweep. Shrinking on the
first saturated element halved the resolution of heads that would only
ever clip a percent of their elements, and cost 0.994 to 0.986 on the
slow regime below; and deciding the growth from the rescaled peak rather
than the period's made a head with a near-zero decay, whose state is a
fresh outer product every token, grow and saturate in alternate tokens.
The rank-one
update is added at `1 / g`, a
reciprocal the engine forms by a 32-cycle restoring division per head,
and the read-out `y = q . S'` is the accumulator times `g`, shifted so its
int16 encoding is the one the gated norm already takes. `fabric.state`
runs the candidates against the float rule on synthetic heads (unit int8
keys, int8 values, U16 gates, three decay regimes); the cosine of the
read-out over the last two thousand of three thousand tokens:

| Scheme | slow (d = 0.999) | fast (d = 0.98) | mixed (0.905 to 0.999) |
| --- | ---: | ---: | ---: |
| int16 round-to-nearest (built first) | 0.9999 / 0.9999 | 1.0000 / 0.9999 | 1.0000 / 0.9999 |
| int8 round-to-nearest | 0.9845 / 0.9698 | 0.9015 / 0.8046 | 0.9822 / 0.9545 |
| int8 stochastic rounding | 0.9850 / 0.9650 | 0.9752 / 0.9227 | 0.9781 / 0.9309 |
| **int8 with a per-head scale and exponent** | 0.9945 / 0.9881 | 0.9958 / 0.9843 | 0.9957 / 0.9879 |

(mean / minimum over tokens.) Stochastic rounding, the usual answer, is
unbiased but adds a random walk the slow decay barely damps; the scale
removes the rounding instead of averaging it, and plain round-to-nearest
on the increments is then better than stochastic. The exponent is what
makes the byte robust to a model's actual magnitudes: the random-init
tiny model's states are a few `s_v` with decays near zero, so at a fixed
LSB of `s_v` the int8 layer tracked the int16 one at only 0.95 on the
state, and with the exponent climbing to use the byte it tracks at 0.99
on the state, 0.98 on the state output and 0.999 on the residual stream
(`test_layer.py`). The
alternative of deferring the state write over several tokens (read the
state, apply pending rank-one updates on the fly, write every `m`
tokens) was rejected on silicon: the pending vectors of every context on
the die (49 contexts, three layers, 32 heads) are 1.2 MB of SRAM per
token of deferral. `delta_state_int8` is the model, `fabric_delta_state8`
the engine (int8 rows in and out, the scale beat in and out, one fewer
multiplier per element than the int16 engine in pass 1 and a peak and
saturation count in pass 2), `tb_delta_state8` checks it bit for bit in
the four cases (no rescale, a rescale that keeps the exponent, one that
raises it, a saturation that lowers it), and the memory map's default is
`state_bits = 8` with `state_bits = 16` still available.
The traffic is 512 KB and 512 B per layer per token each way, which is
what the simulator's `recurrent_state_bytes` had assumed.

Per token and layer the arithmetic outside the tiles is about 2M
multiply-adds in the state engine (four operations over 32 × 128 × 128),
16K in the convolution and 12K in SwiGLU, against 866M in the tiles. At
128 lanes a head's state update is 262 cycles, so four engines cover the
32 heads inside one 2048-cycle pass; the attention core at 64 lanes
consumes a 256-wide key or value row in four cycles, 8192 cycles for the
1024 rows of window and retrieved blocks per KV head, so it too wants one
engine per KV head to hide inside the layer's four passes. What the units
do not include is the memory side: the recurrent state and conv history
per context, the KV rows of the window and the retrieved blocks and the
index scan that chooses them, and the head die's top-k and log-sum-exp.

## The memory side

A layer die keeps, for every resident context, the Gated DeltaNet state and
convolution history of its three recurrent layers and, for its global
layer, the local window of recent keys and values, the compressed blocks of
older ones and the index that chooses among them, all in the die's local
memory behind one port. `memory.py` and `rtl/fabric_memory.sv` are that
side: the address map, the record layouts, the block and index arithmetic,
integer and float twins of a context's stores, and the units on the port.

```text
per context                              per token, global layer
  state0..2   32 x 128 x 128 int16         append   this token's K,V -> window slot pos mod 512
  hist0..2    8192 x 3 int8                         block sums; every 16th token the block record
  window      512 x 4 heads x [K|V]                 and its index record (4-bit codes + scale)
  blocks      8192 x 4 heads x [K|V]       scan     index records 0..n-1, score, top-32
  index       8192 x (64 B codes, scale)   read     512 window + 32 block records -> attention core
```

| Record | Layout | Bytes (9B, int8 KV) |
| --- | --- | ---: |
| state row | 128 int16, one row of one head's `S` | 256 |
| KV record | key then value of one KV head at `kv_bits`, each half padded to beats | 512 (256 at int4) |
| index record | `index_dim` 4-bit codes, then a beat with the scale in its first byte | 80 |

The memory port is 128-bit beats, a request of up to 255 beats, write
beats after the request, read beats in order, one request in flight per
requester; `fabric_mem_arbiter` puts several requesters on the one port a
transaction at a time. The units:

* `fabric_row_dma` streams a head's state rows from memory into the delta
  engine and its output rows back, a burst per row, with a FIFO of a
  head's rows on the write side; the test runs a 128 x 128 head through
  the DMA, the engine and the arbiter and checks the memory image after.
* `fabric_kv_append` writes the token's KV records into the window slot,
  keeps the block sums on chip, and at a block's end writes the block
  records and its index record: the block mean, L2-normalised by the norm
  unit, its absolute maximum as the scale, and
  `code = round((u / scale + 1) * 15 / 2)` by the reciprocal unit.
* `fabric_index_scan` reads the eligible index records a page at a time
  (25 records per request), scores each as `scale * sum (2q - 15)(2k - 15)`
  against the query's codes (32 codes per beat), and hands the candidates
  to `fabric_topk`, a sorted list of K entries that a candidate enters
  above the first it strictly beats.
* `fabric_record_reader` takes requests of consecutive records (a page of
  the window, or one block record), reads them in one burst into a record
  buffer, unpacks int4 to int8 where needed, and streams them record by
  record to the attention core as key beats then value beats; the test
  drives a retrieval's requests through it into `fabric_attention` and
  checks the heads' outputs.

Retrieval semantics are the reference model's, and `GlobalContextMemoryFloat`
reproduces `SparseGlobalMixer` token by token to 1e-6 over a sequence
with blocks and retrieval (`test_memory.py`). The integer store differs
from it in three places, each defined here: keys and values are stored as
the int8 (or int4, by a shift of four) values the layer produced, block
means are rounded to int8, and the index codes come from the reciprocal
unit, which differs from the float quantiser in about four percent of the
codes by one level, at rounding ties. On the tiny model the integer store
selects the same blocks as the float store seven times in ten and the
attention output tracks it at 0.95, with the residual stream at 0.995.

### The global layer's traffic

The sequencer's schedule made the global layer the larger of the two once
the state was int8, so its three memory terms were taken in turn:

* **The window is head-major.** A KV head's positions are consecutive
  (`kv_head * W + pos mod W`), so the record reader reads the window in
  page bursts of eight int4 records (four at int8) at 96 percent burst
  efficiency instead of a record per request at 86 (76 at int4), and the
  append's four records per token go to four addresses instead of one.
* **The index scan reads a page per request**, 25 records, at 96 percent
  instead of one 80-byte record per request at 50.
* **Keys and values are int4** by default (`kv_bits = 4`, 8 available),
  which halves the window, the block store and the rows the cores read.
  On the tiny model the attention output tracks the quantised-weight float
  at 0.989 as int4 against 0.9996 as int8, and the residual stream at
  0.99998 either way; the same order of loss the int8 state took, for half
  the largest traffic term.

The program had also been reading sixteen rows per retrieved block where
the model attends over one mean record per block, the same over-count
the simulator makes; corrected, one KV head reads the window and 32
block records, 544 rows at the end of the context, not 1024.

Capacity and traffic for the 9B geometry at 128K context
(`MemoryMap.report_markdown`):

| Store | int8 state, int4 KV (taken) | int8 state, int8 KV | int16 state, int8 KV (first build) |
| --- | ---: | ---: | ---: |
| per context | 11.2 MB | 20.1 MB | 21.7 MB |
| contexts in 1 GB (PSRAM board) | 95 | 53 | 49 |
| contexts in 4 GB (LPDDR5X) | 382 | 213 | 197 |
| traffic per token, three state layers | 3.1 MB | 3.1 MB | 6.3 MB |
| traffic per token, global layer | 1.2 MB | 1.8 MB | 1.8 MB |

Of the global layer's 1.2 MB at the end of the context, 0.65 MB is the
index scan, 0.52 MB the window and 0.03 MB the block records: the scan
is now the largest term, and its next lever is a coarser index (fewer
code bits or a shorter index vector), which is the model's business.

The int8 state is the scaled one of the layer section: 512 KB of rows and
512 B of scales per layer per context, which is the figure the
simulator's `recurrent_state_bytes` carries. What the map still says
about the simulator: its retrieval term fetches `top_blocks x
retrieval_block_size` positions, 512 records per token, where the model
attends over one mean record per block, 32, so the simulator's
global-layer traffic is about 2.5 times what this memory side moves and
the memory-bound sweeps in `sim/README.md` are pessimistic on the global
side until the selected-position count follows this map.

Not here: error detection on the stored state, and context allocation and
eviction, which the FPGA owns. The sequencer that orders the units across
a token and the memory controller and PHY for the chosen device are in
their own sections below.

## The memory device and its controller

The PSRAM board's device is chosen: the **AP Memory APS512XXN-OB9-BG**,
512 Mb, the Xccela DDR interface in its 16-bit HPI mode, 250 MHz, 1.62 to
1.98 V, a 24-ball 6 × 8 mm BGA at 1.0 mm pitch, per its datasheet (rev
1.0, November 2025). Sixteen of them give a layer die 1 GB at 16 GB/s over
twenty signals per device (DQ0-15, DQS/DM0, DQS/DM1, CE#, CLK), a clock
per device (`hw/si.py`: four on one clock cannot meet the device's 0.6 ns
edge), on 1.8 V LVCMOS with no DRAM PHY. `hpi.py` holds
the geometry, the mode-register encodings, the command frame and the
striping; `rtl/fabric_hpi.sv` is the controller.

The protocol, as the controller drives it: CE# low, the instruction byte
on A/DQ[7:0] at the first rising edge, the address bytes A3..A0 on the
second and third clocks' rising and falling edges (row 15 bits, column
10), then for a read the device holds DQS low and starts the data
`latency` clocks after the third clock plus up to `latency` more of
refresh push-out under variable latency, two words per clock with DQS
toggling; for a write the controller presents the first word `WLC`
clocks after the third clock with DM low. Linear bursts (20h, A0h) wrap
inside the 1024-word page; CE# stays low at most 4 µs and rises at least
28 ns between bursts. Initialisation is the 150 µs power-up wait with the
clock stopped, a global reset, MR0 = 18h (variable latency 10), MR4 = 60h
(write latency 9), MR8 = 43h (x16, 1K-word wrap), then MR1 and MR2 read
back and checked for the vendor, the density and the good-die mark.

```text
port (128-bit beats, bursts to 4095)        fabric_hpi_stripe
   |  2 KB stripes on consecutive devices, a burst split into chunks that
   |  run concurrently, read data returned in order from the page buffers
   v
fabric_hpi_channel x 16   init, frame, write latency, DQS capture, page buffer
   |  CLK (a quarter period late), CE#, DQ[15:0], DQS/DM[1:0]
   v
APS512XXN-OB9-BG x 16
```

Three design choices. Reads are captured by the device's strobe, not
counted: DQS delayed a quarter period clocks a small ring in the strobe
domain, gray pointers cross to the controller clock, and the controller
takes pairs as they appear, so the refresh push-out costs nothing and
variable latency is the right mode. The stripe is one page, so a chunk is
one burst and a head's 32 KB state is one burst across all sixteen
devices; the stripe unit tracks issued chunks in a queue and drains their
page buffers in order, and a write is done when every chunk's channel has
finished, not when the data has left the port. The device clock is the
controller clock delayed a quarter period and the capture strobe is DQS
delayed the same, which are the PHY's two delay lines.

The delay lines are in `rtl/fabric_phy.sv`. A tap line's delay is code
times tap, and the tap moves with process, voltage and temperature (25 to
90 ps is the range the design takes), so a master DLL measures the clock
period in taps: a replica line delays the clock, the delayed clock samples
the reference, and the search walks the code up through the high half and
the low half of the period until the sample returns high, which is one
period; after that it dithers one tap either side of the boundary and
tracks. The quarter is that code over four, and the slave lines (the
device clock and the two DQS strobes of every channel) take a new quarter
only while every channel has CE# high, so a code step never lands in a
frame or a burst. The line is 256 taps, so a 25 ps tap still spans the
4 ns period; a line that cannot span it is reported on `range_err`
rather than searched forever. The channel waits for the lock before it
starts the device (`phy_ready`), which costs nothing since the lock
(220 to 680 clocks) is over long before the 150 µs power-up wait.
`fabric_delay_line` is the behavioural stand-in for the hard cell; the DLL
and the gating are synthesizable. `tb_dll` checks the lock, the period
within a tap and the slave delay within a tap and a half at taps of 25,
40, 60 and 90 ps, and `tb_hpi` and `tb_mem_bridge` run the whole
controller through the delay lines and the DLL (`USE_DLL`) instead of the
ideal quarter-period delays.

Burst efficiency at 250 MHz, three command clocks plus the minimum
latency, two words per clock, and tCPH:

| Burst | Data clocks | Total | Efficiency |
| --- | ---: | ---: | ---: |
| a full page, 2 KB | 512 | 532 | 96% |
| a 512 B key-value record | 128 | 148 | 86% |
| a 256 B state row or int4 record | 64 | 84 | 76% |
| an 80 B index record | 20 | 40 | 50% |

The index scan suffered most: at 80 B per record it ran at half rate,
which is why the scan now reads a page of records per request (the
memory-side section above).

The controller runs on the device clock and the arbiter and the units on
the core's, 800 MHz against 250, so the memory port crosses a clock
boundary in `rtl/fabric_cdc.sv`: `fabric_async_fifo` is a dual-clock FIFO
with gray-coded pointers and two-flop synchronisers, full decided on the
write clock from the synchronised read pointer and empty on the read
clock from the synchronised write pointer, so neither flag is ever
optimistic; `fabric_mem_bridge` carries the port's three streams through
three of them, four requests, sixteen write beats and eight read beats
deep. The read stream has no back-pressure at either end, so its FIFO
relies on the core draining faster than the controller fills, one beat
per 1.25 ns against one per 4 ns; a core that stops its clock would
overflow it, and that is latched into `rd_overflow` rather than lost
silently. `tb_async_fifo` runs the FIFO at three unrelated clock ratios
with random bursts of pushes and pops and a scoreboard, and confirms it
fills to the last entry; `tb_mem_bridge` drives the whole memory path from
the core clock through the bridge, the stripe unit, the channels and the
device models at core periods of 1.25, 1.3 and 3.0 ns against the 4 ns
device clock.

`tb_hpi` puts the stripe unit, the channels and a behavioural model of the
device (`fabric_hpi_device`: the frame, the latencies with a pseudo-random
push-out, a different tDQSCK per device, the page wrap, the registers,
and `$error` on a violated tCEM, tCPH, an odd address or an array access
in x8 mode) under random bursts of up to 300 beats, and checks the read
data beat by beat and the device images at the end, over one, two, four
and sixteen devices. Two protocol details come from the datasheet's
figures rather than its text and are recorded as the assumption the
controller and the model share: the first data word sits `latency` clocks
after the third command clock, and the 512 Mb part's RA[14] rides in bit
1 of A3. Both are one-line changes if the vendor's figure says otherwise.

## The token sequencer

Everything above is a unit with a start and a done; `sequencer.py` and
`rtl/fabric_sequencer.sv` are what runs them in order for a token. A layer
is a fixed dataflow graph over a few dozen on-chip vector buffers (the
residual, the pass input, the pass outputs, the per-head state slots), and
the sequencer's program is that graph written out as a list of steps, each
a command to one unit (a tile pass, a norm, a state-engine head, a DMA)
with the buffers it reads and writes. The dependencies come from the
buffer names: a reader waits for its buffer's last writer, a whole vector
for every slice written into it, and a writer for the readers since the
last write, since the buffers are reused. Three things are derived from
the same list, so they cannot drift apart:

* `run_program` executes the steps on the integer model and must
  reproduce `recurrent_layer_int` and `global_layer_int` bit for bit
  (`test_sequencer.py` does, over six tokens of each on the tiny
  geometry), which proves the program's data flow, its head loops and its
  buffer reuse are complete and legally ordered;
* `schedule` runs a list scheduler over the steps with each unit's lanes
  and latency from the RTL and the memory port's bandwidth at the HPI burst
  efficiency, giving cycles per token and where the time goes;
* `emit_program` writes the controller's program image: unit, engine,
  length, the dependency mask over the previous 32 steps, and a barrier
  where a dependency reaches further back (the recurrent layer's head loop
  is 128 steps, so its `out_proj` pass is a barrier).

The controller is a microcoded issue engine, not a state machine per
layer: in program order, when the head step's dependencies have completed
and its unit reports the addressed engine free, it sends the command and
moves on without waiting; a unit returns the step's tag on its engine's
done port, and completed steps are kept in a ring of bits relative to the
head so a dependency is one compare. Units are black boxes to it, which is
what lets `tb_sequencer` check it against stub units of programmed
duration: the trace must respect every dependency and never overlap an
engine, and the last completion must land on exactly the cycle the Python
schedule gives, which it does for the tiny and the full-size programs of
both layers (33, 23, 173 and 41 steps). The three timing rules the two
share: a step issues after the previous issue, one cycle after the last of
its dependencies completed, and one cycle after its engine's previous
step completed.

The full-size schedule is the finding. At the 9B geometry with sixteen
PSRAMs (16 GB/s, 20 bytes per core cycle), one token through one layer:

| Layer | Steps | Cycles | Time | Tiles | Memory port |
| --- | ---: | ---: | ---: | ---: | ---: |
| recurrent, int16 state (first build) | 173 | 140,073 | 175 µs | 9% | 80% |
| recurrent, int8 state with a scale | 173 | 85,673 | 107 µs | 14% | 67% |
| global, position 4095, int8 KV, record bursts, 16 rows per block (first build) | 41 | 152,133 | 190 µs | 8% | 81% |
| global, position 131071, the same | 41 | 215,621 | 270 µs | 6% | 87% |
| global, position 4095, int4 KV, page bursts, one row per block | 41 | 53,606 | 67 µs | 23% | 57% |
| global, position 131071, the same | 41 | 86,620 | 108 µs | 14% | 73% |

The tiles' four passes are 12,352 cycles; the rest is the memory port
moving the recurrent state (1 MB per token per layer in and out as int8
with its scales, 2 MB as int16) and, in the global layer, the index scan
and the window and block rows of each KV head. So on this board the
sequencer's whole job is to keep the port streaming: the state engines
double-buffer a head so the next head's read overlaps the update, and the
four attention cores each own a KV head's stream. The int8 state took the
recurrent layer from 175 to 107 µs, and the head-major window, the page
bursts, int4 keys and values and the corrected block rows took the global
layer from 270 to 108 µs at the end of the context, with both layers now
near 100 µs per token; the DRAM boards are not memory-bound this way.
What the program does not
yet carry is the per-context addressing (the DMA steps take an argument
field for it) and the overlap of one token's memory steps with the previous
token's passes, which the issue engine allows once two tokens' programs
are concatenated with their buffers renamed.

## RTL

`rtl/fabric_tile.sv` is the synthesizable tile with the ROM as a constant
array loaded from a hex image. It is parameterized in rows, columns,
coefficient and activation width, rows per cycle, and accumulator and
requantizer widths. In silicon the ROM array and the multiple-select crossbar
become a hard macro; the RTL is the functional stand-in and the synthesis
baseline for the column datapath.

`rtl/tb_fabric_tile.sv` reads vectors written by `fabric.tile.emit_vectors`,
streams the activations with occasional bubbles, and compares the raw
accumulators and requantized outputs against the model. The tests run three
configurations through Icarus Verilog: 64×8 with chained partial sums, 256×64
at four rows per cycle, and 4096×16 at full depth.

The vector units are `rtl/fabric_vector.sv` (interpolated table, sigmoid,
SiLU, exp, softplus, inverse square root, reciprocal), `fabric_norm.sv`
(the RMS norm, also the L2 normaliser and the gated norm by its
parameters), `fabric_recurrent.sv` (convolution with SiLU, head gates,
delta state), `fabric_ffn.sv` (SwiGLU, residual add) and
`fabric_attention.sv` (rotary table and rotation, the attention core), with
the fixed-point helpers in `fabric_fx.svh`. Each is a streaming unit of
`L` lanes per beat with a stated latency; the tables load from hex images
written by `fabric.layer.write_luts`, the per-channel constants arrive with
the beat from whatever memory the integration keeps them in. The
testbenches `tb_vector_units`, `tb_rmsnorm`, `tb_conv_silu`,
`tb_head_gates`, `tb_delta_state`, `tb_ffn`, `tb_rotary` and
`tb_attention` compare every output bit for bit with vectors from the
`emit_*` functions of `fabric.layer`, at small sizes in the test suite and
at the layer's own sizes (a 4096-element norm, a 128 × 128 state, 8192
conv channels, four 256-wide heads at 64 lanes) when run by hand. These
units are written for function, not for the tile's no-carry-chain
discipline: the norm's sum of squares and the state engine's accumulators
are plain adders.

`rtl/fabric_hpi.sv` holds the device model, the channel controller and the
stripe unit, with `tb_hpi` checking them against `fabric.hpi`;
`rtl/fabric_cdc.sv` the asynchronous FIFO and the bridge, with
`tb_async_fifo` and `tb_mem_bridge`; `rtl/fabric_phy.sv` the delay line
and the DLL, with `tb_dll`; `rtl/fabric_sequencer.sv` the token sequencer,
with `tb_sequencer` running the programs of `fabric.sequencer` over stub
units.

`rtl/fabric_memory.sv` holds the memory side: the behavioural
`fabric_mem_model` for the testbenches, `fabric_mem_arbiter`,
`fabric_row_dma`, `fabric_topk`, `fabric_index_scan`,
`fabric_record_reader` and `fabric_kv_append`, with the testbenches
`tb_topk`, `tb_index_scan`, `tb_kv_append`, `tb_record_reader` and
`tb_row_dma` checking memory images and outputs against `fabric.memory`
bit for bit, at int8 and int4 KV and with the 128-wide index and the
128 x 128 state.

## What is next

1. Detailed routing and a multi-corner pass of the OpenROAD flow above, a
   buffer tree on the walking select for sky130, and then the same flow on
   the target library.
2. A via-ROM compiler cell from the foundry, or a hand-drawn cell for the MPW,
   to replace the ROM area and read energy placeholders.
3. The GDS writer: `via_coordinates` into the ROM macro's bit-cell grid.
4. Done: the simulator now admits a token per pass rather than per layer
   (`layer_pass_cycles`). It is worth almost nothing at real memory
   bandwidths and 2.5x with the memory removed, so the RTL question it
   raises — whether consecutive tokens of different contexts may occupy
   consecutive passes without a shared resource between them — is not
   urgent. What the passes do need is balance: the model takes them as
   equal, and the FFN-down pass reads the wider FFN vector.
5. A multi-token variant of the column datapath for chunked prefill, which
   amortizes the ROM read across a chunk of tokens from one context.
6. Done: the vector datapath between the passes, the memory side, the
   HPI controller for the chosen PSRAM with its clock crossing and its
   PHY, and the token sequencer with its layer programs, above. Open
   behind them: the sequencer's integration with the real units (the
   command bus into each unit's start, the buffer addressing, the
   per-context arguments), the simulator's traffic terms brought in line
   with the map, and synthesis of the units for area. The state traffic
   and the global layer's traffic are done, above; the next memory lever
   is the index scan's record size, which is the model's.
