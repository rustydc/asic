# ASIC package and ball map

Derived by `python -m hw.pinout` from `hw/board_psram.yaml`. Do not edit.

## Requirements

Rated at 5000 tokens/s the core rail draws 21 A (3.0 pJ per MAC, 0.8 V), one ball per 0.5 A.

| Need | Balls |
| --- | ---: |
| Signals (24 link, 304 memory, 18 management) | 346 |
| Signal ground returns (1 per 4) | 87 |
| Core rail | 43 |
| Ground for the core | 43 |
| VDD_IO_1V8 | 24 |
| VDD_PLL_0V9 | 2 |
| **Total** | **545** |

## Candidates

| Package | Balls | Outer rows | Body | Verdict |
| --- | ---: | ---: | ---: | --- |
| FCBGA784_28x28_P0.8 | 784 | 460 | 23 mm | **selected** |
| FCBGA1225_35x35_P0.8 | 1225 | 600 | 29 mm | not needed |
| FCBGA961_31x31_P1.0 | 961 | 520 | 33 mm | not needed |
| FCBGA2025_45x45_P1.0 | 2025 | 800 | 48 mm | not needed |

## Ball map

FCBGA784_28x28_P0.8: 28 x 28 at 0.8 mm, 23 mm body.

| Use | Balls | Where |
| --- | ---: | --- |
| link_in | 12 | S edge, a two-deep block of 6 positions |
| link_out | 12 | S edge, beside it |
| 16 memory channels | 304 | N and E and W rows, lanes with a ground after each |
| mgmt, jtag, refclk, strap | 14 | S row |
| VDD_CORE | 183 | interior checkerboard |
| GND | 229 | interior checkerboard and lane returns |
| VDD_IO_1V8 | 24 | interior |
| VDD_PLL_0V9 | 2 | interior |

Signal balls deeper than the outer 5 rows: 0 (these need a microvia or build-up escape on the PCB).

## What the packaging house gets

* this map as `asic_ballmap.csv`, with the edge each interface must face;
* the die-edge assignment it implies: link ports on the S and S die edges, the memory PHYs on the N and E and W edges, management on the south;
* the core current (21 A at the rating, 181 A at 50K tokens/s) for the bump map and the substrate power planes.

The substrate design, the bump map and the final ball map come back from them; the loop usually runs two or three times and this file is regenerated from the agreed rules each round.
