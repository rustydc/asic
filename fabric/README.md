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
| Area at 2 rows/cycle | 229 mm² | 134 mm² | 164 mm² | 134 mm² |
| Latency per layer, 800 MHz | 10.2 µs | 2.6 µs (die) | 6.4 µs | 1.6 µs (die) |
| Fabric energy per token | 62 µJ | 37 µJ | 32 µJ | 19 µJ |

The head slice fits inside a layer die's tile count in both geometries, which
is what makes the head-mode personalization possible. At 50K tokens/s the
fabric energy is about 3 W per 9B layer die before clocks, memory, and I/O.

The rows-per-cycle knob trades column area for latency. ROM area is fixed by
the coefficient count; the MAC columns scale with rows per cycle.

| Rows/cycle | Clock | 9B layer die area | ROM / MAC | Latency per layer | Pass |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 800 MHz | 186 mm² | 104 / 74 mm² | 20.5 µs | 5.1 µs |
| 2 | 800 MHz | 229 mm² | 104 / 116 mm² | 10.2 µs | 2.6 µs |
| 4 | 800 MHz | 313 mm² | 104 / 201 mm² | 5.1 µs | 1.3 µs |
| 2 | 500 MHz | 229 mm² | 104 / 116 mm² | 16.4 µs | 4.1 µs |

Two rows per cycle is the baseline. The simulator's stage times are derived
from it: 102 ticks (10.2 µs) per 9B layer, 26 ticks per head die.

Because a layer is four sequential passes, a layer stage could accept a new
token every pass (2.6 µs) while a token takes four passes to get through. The
simulator currently treats a layer as one stage with initiation interval equal
to latency, so it understates fabric throughput by up to 4×. It does not
matter yet because the global stage's memory traffic is the bottleneck at
about 66 µs.

## Density model

`DensityModel` holds the placeholders that the MPW tile is meant to replace:

| Placeholder | Value | What measures it |
| --- | ---: | --- |
| ROM area | 0.03 µm² per bit | via-ROM compiler cell at the target node |
| MAC column | 200 µm² per rows-per-cycle unit | synthesis of the column datapath |
| Accumulator and requantizer | 150 µm² per column | synthesis |
| Tile overhead | 2500 µm² | multiples generator, ROM periphery, control |
| Clock | 800 MHz | ROM read plus column add in one cycle |
| ROM read | 3 fJ per bit | silicon |
| MAC | 60 fJ per coefficient | silicon |

At these numbers a 28 nm-class 9B layer die is 229 mm² with the ROM and the
MAC columns roughly equal. At a 16 nm-class node expect roughly half. If the
ROM cell comes in denser than 0.03 µm² per bit, the MAC columns dominate and
one row per cycle becomes the better trade.

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

1. Column datapath synthesis on the target library to replace the MAC area
   placeholders.
2. A via-ROM compiler cell from the foundry, or a hand-drawn cell for the MPW,
   to replace the ROM area and read energy placeholders.
3. The GDS writer: `via_coordinates` into the ROM macro's bit-cell grid.
4. Pass-level pipelining in the simulator so a layer stage's initiation
   interval is one pass rather than four.
5. A multi-token variant of the column datapath for chunked prefill, which
   amortizes the ROM read across a chunk of tokens from one context.
