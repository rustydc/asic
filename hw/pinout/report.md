# ASIC package and ball map

Derived by `python -m hw.pinout` from `hw/board.yaml`. Do not edit.

## Requirements

Rated at 14500 tokens/s the core rail draws 54 A (3.0 pJ per MAC, 0.8 V), one ball per 0.5 A.

| Need | Balls |
| --- | ---: |
| Signals (72 link, 130 memory, 14 management) | 216 |
| Signal ground returns (1 per 4) | 54 |
| Core rail | 108 |
| Ground for the core | 108 |
| VDD_IO_1V8 | 8 |
| VDD_PLL_0V9 | 2 |
| VDD2H_1V05 | 12 |
| VDDQ_0V3 | 12 |
| **Total** | **520** |

## Candidates

| Package | Balls | Outer rows | Body | Verdict |
| --- | ---: | ---: | ---: | --- |
| FCBGA400_20x20_P1.0 | 400 | 204 | 23 mm | 400 balls, 520 needed |
| FCBGA784_28x28_P0.8 | 784 | 300 | 23 mm | **selected** |
| FCBGA1225_35x35_P0.8 | 1225 | 384 | 29 mm | not needed |
| FCBGA961_31x31_P1.0 | 961 | 336 | 33 mm | not needed |

## Ball map

FCBGA784_28x28_P0.8: 28 x 28 at 0.8 mm, 23 mm body.

| Use | Balls | Where |
| --- | ---: | --- |
| link_in | 36 | W edge, columns 1-2, rows 6-23 |
| link_out | 36 | E edge, same rows |
| lpddr_ch0, lpddr_ch1 | 130 | N rows, byte lanes with a ground after each |
| mgmt, jtag, refclk, strap | 14 | S row |
| VDD_CORE | 243 | interior checkerboard |
| GND | 291 | interior checkerboard and lane returns |
| VDD_IO_1V8 | 8 | interior |
| VDD_PLL_0V9 | 2 | interior |
| VDD2H_1V05 | 12 | interior |
| VDDQ_0V3 | 12 | interior |

Signal balls deeper than the outer 3 rows: 41 (these need a microvia or build-up escape on the PCB).

## What the packaging house gets

* this map as `asic_ballmap.csv`, with the edge each interface must face;
* the die-edge assignment it implies: link ports on the west and east die edges, both LPDDR5X PHYs on the north edge, management on the south;
* the core current (54 A at the rating, 177 A at 50K tokens/s) for the bump map and the substrate power planes.

The substrate design, the bump map and the final ball map come back from them; the loop usually runs two or three times and this file is regenerated from the agreed rules each round.
