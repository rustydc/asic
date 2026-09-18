# ASIC package and ball map

Derived by `python -m hw.pinout` from `hw/board_27b.yaml`. Do not edit.

## Requirements

Rated at 153000 tokens/s the core rail draws 249 A (0.35 pJ per MAC, 0.8 V), one ball per 0.5 A.

| Need | Balls |
| --- | ---: |
| Signals (72 link, 0 memory, 14 management) | 86 |
| Signal ground returns (1 per 4) | 22 |
| Core rail | 498 |
| Ground for the core | 498 |
| VDD_IO_1V8 | 8 |
| VDD_PLL_0V9 | 2 |
| VDD_HBM_1V1 | 24 |
| VDDQ_HBM_0V4 | 12 |
| **Total** | **1150** |

## Candidates

| Package | Balls | Outer rows | Body | Verdict |
| --- | ---: | ---: | ---: | --- |
| FCBGA1225_35x35_P0.8 | 1225 | 384 | 29 mm | **selected** |
| FCBGA2025_45x45_P1.0 | 2025 | 504 | 48 mm | not needed |
| FCBGA3025_55x55_P1.0 | 3025 | 624 | 55 mm | not needed |
| FCBGA3844_62x62_P1.0 | 3844 | 708 | 62 mm | not needed |

## Ball map

FCBGA1225_35x35_P0.8: 35 x 35 at 0.8 mm, 29 mm body.

| Use | Balls | Where |
| --- | ---: | --- |
| link_in | 36 | W edge, columns 1-2, rows 9-26 |
| link_out | 36 | E edge, same rows |
| memory | 0 | in the package (HBM on the interposer), no balls |
| mgmt, jtag, refclk, strap | 14 | S row |
| VDD_CORE | 522 | interior checkerboard |
| GND | 571 | interior checkerboard and lane returns |
| VDD_IO_1V8 | 8 | interior |
| VDD_PLL_0V9 | 2 | interior |
| VDD_HBM_1V1 | 24 | interior |
| VDDQ_HBM_0V4 | 12 | interior |

Signal balls deeper than the outer 3 rows: 0 (these need a microvia or build-up escape on the PCB).

## What the packaging house gets

* this map as `asic_ballmap.csv`, with the edge each interface must face;
* the die-edge assignment it implies: link ports on the west and east die edges, the HBM PHY on the north edge towards the stack, management on the south;
* the core current (249 A at the rating, 90 A at 50K tokens/s) for the bump map and the substrate power planes.

The substrate design, the bump map and the final ball map come back from them; the loop usually runs two or three times and this file is regenerated from the agreed rules each round.
