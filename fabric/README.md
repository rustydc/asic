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
cycles exposes the raw accumulators and the requantized int8 outputs. Matrices
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
| Area at 2 rows/cycle | 199 mm² | 117 mm² | 138 mm² | 94 mm² |
| Latency per layer, 800 MHz | 10.2 µs | 2.6 µs (die) | 6.4 µs | 1.6 µs (die) |
| Fabric energy per token | 62 µJ | 37 µJ | 32 µJ | 19 µJ |

The head slice fits inside a layer die's tile count in both geometries, which
is what makes the head-mode personalization possible. At 50K tokens/s the
fabric energy is about 3 W per 9B layer die before clocks, memory, and I/O.

The rows-per-cycle knob trades column area for latency. ROM area is fixed by
the coefficient count; the MAC columns scale with rows per cycle.

| Rows/cycle | Clock | 9B layer die area | ROM / MAC | Latency per layer | Pass |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 800 MHz | 178 mm² | 104 / 66 mm² | 20.5 µs | 5.1 µs |
| 2 | 800 MHz | 199 mm² | 104 / 87 mm² | 10.2 µs | 2.6 µs |
| 4 | 800 MHz | 241 mm² | 104 / 129 mm² | 5.1 µs | 1.3 µs |
| 2 | 500 MHz | 199 mm² | 104 / 87 mm² | 16.4 µs | 4.1 µs |

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
| MAC column | 100 µm² per bank | 330 NAND2 equivalents from four-library synthesis, at a 0.30 µm² 28 nm NAND2 |
| Accumulator, requantizer share, output | 210 µm² per column | 700 NAND2 equivalents, same source |
| Tile overhead | 2500 µm² | placeholder: multiples generator, ROM periphery, control |
| Clock | 800 MHz | placeholder: ROM read plus column add in one cycle |
| ROM read | 3 fJ per bit | placeholder |
| MAC | 60 fJ per coefficient | placeholder |

At these numbers a 28 nm-class 9B layer die is 199 mm², about half ROM and
slightly less than half MAC columns. At a 16 nm-class node expect roughly
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

Measured on a 16-column tile (256 rows; column cost does not depend on depth)
at two rows per cycle, on two manufacturable 130 nm libraries and two
predictive advanced-node kits:

| Library | Kind | NAND2 | Cells | Per column | NAND2-eq per column |
| --- | --- | ---: | ---: | ---: | ---: |
| SkyWater sky130 HD (130 nm) | manufacturable | 3.75 µm² | 12,087 | 5221 µm² | 1392 |
| IHP SG13G2 (130 nm) | manufacturable | 7.26 µm² | 12,027 | 9349 µm² | 1288 |
| NanGate 45 (FreePDK45) | predictive | 0.80 µm² | 12,999 | 1043 µm² | 1307 |
| ASAP7 (7 nm FinFET) | predictive | 0.058 µm² | 11,974 | 74 µm² | 1271 |

Four libraries spanning 130 nm to 7 nm agree within ten percent once
normalized to their own NAND2: about 1300 NAND2 equivalents per column at two
rows per cycle, 1000 at one and 1900 at four (sky130: 3916 and 7876 µm²;
ASAP7: 57 and 111 µm²). That is the number to carry, and it makes the
28 nm estimate a NAND2 area, about 0.30 µm², times 700 + 330 per bank. The
first synthesis run also caught a design error: a per-column requantizer
multiplier that tripled the column area, now a single unit shared across the
64 columns.

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
* **Timing: partly.** yosys gives area; the clock needs OpenSTA or a full
  OpenROAD flow, which is the next step. Expect 100 to 200 MHz at 130 nm and
  scale from there.
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

1. Timing of the column datapath with OpenSTA or OpenROAD on sky130, then on
   the target library, to replace the 800 MHz placeholder.
2. A via-ROM compiler cell from the foundry, or a hand-drawn cell for the MPW,
   to replace the ROM area and read energy placeholders.
3. The GDS writer: `via_coordinates` into the ROM macro's bit-cell grid.
4. Pass-level pipelining in the simulator so a layer stage's initiation
   interval is one pass rather than four.
5. A multi-token variant of the column datapath for chunked prefill, which
   amortizes the ROM read across a chunk of tokens from one context.
