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

## What is next

1. Detailed routing and a multi-corner pass of the OpenROAD flow above, a
   buffer tree on the walking select for sky130, and then the same flow on
   the target library.
2. A via-ROM compiler cell from the foundry, or a hand-drawn cell for the MPW,
   to replace the ROM area and read energy placeholders.
3. The GDS writer: `via_coordinates` into the ROM macro's bit-cell grid.
4. Pass-level pipelining in the simulator so a layer stage's initiation
   interval is one pass rather than four.
5. A multi-token variant of the column datapath for chunked prefill, which
   amortizes the ROM read across a chunk of tokens from one context.
