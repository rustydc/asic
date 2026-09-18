# Appliance architecture simulator

This is a deterministic, cycle-stepped transaction-level simulator for the
eight-ASIC appliance. It moves tagged work items rather than real tensors and
models the resources that determine system throughput:

* 32 finite-queue layer stages in `R/R/R/G` order, followed by one stage per
  head-mode ASIC (`num_head_asics`, `head_cycles`) that forwards the hidden
  vector plus `head_result_bytes` of top-k partial result;
* one unresolved token per autoregressive context;
* round-robin FPGA injection and configurable sampling delay;
* serialized links at ASIC boundaries;
* identical FULL retrieval at all global stages;
* context-length-dependent index/KV reads and append writes;
* configurable sustainable LPDDR efficiency and per-ASIC bandwidth;
* overlapped index, top-K, KV, attention, and output global-stage phases;
* downstream backpressure, utilization, latency, and FIFO watermarks.

The global memory model is intentionally coarse. It converts index and KV bytes
to service cycles using sustained bandwidth; it does not yet model LPDDR banks,
row locality, refresh, or outstanding transaction limits. The simulator should
eventually consume timing measured from RTL and emit memory traces for a detailed
DRAM simulator.

## Run

From the repository root:

```bash
python -m unittest discover -s sim/tests -t .
python -m sim.run --config sim/config/baseline.json      # Qwen3.5-9B geometry
python -m sim.run --config sim/config/qwen35_4b.json     # Qwen3.5-4B geometry
python -m sim.run --config sim/config/qwen35_9b_7nm_hbm.json   # 7 nm-class die, HBM per ASIC
python -m sim.run --config sim/config/qwen35_9b_3nm_hbm.json   # 3 nm-class die, HBM per ASIC
python -m sim.run --config sim/config/baseline.json --trace trace.json
python -m sim.sweep --kv-element-bytes 1,0.5 --top-blocks 32,16 --output retrieval-sweep.csv
```

The trace command writes a Chrome/Perfetto-compatible trace. Open it in Perfetto
to inspect stage occupancy by context and token.

Warm-up tokens are simulated but excluded from reported throughput and latency,
which reduces pipeline fill bias. `global_max_inflight` controls how many
contexts can occupy different global-engine phases at once. Index and selected
KV transfers still share the configured memory interval.

The sweep reports bytes/token, operation latency, initiation interval, and the
global-stage throughput ceiling across index dimension, index precision,
compression, KV element size, top-K, and bandwidth combinations. It is
analytical and therefore much faster than running every point through the full
pipeline.

## Configurations

Both configurations use the Qwen3.5 head geometry: `kv_heads=4` KV heads of
`head_dim=256`, so a stored position costs 2 KB at int8. Both add two
head-mode ASICs after the last layer chip. Fabric stage times come from the
tile model in `fabric/` at two rows per cycle and 800 MHz: four passes of
2048 cycles per 9B layer (`recurrent_cycles` 102 ticks) and one pass per head
die (26 ticks); the 4B tile is 2560 deep, so 64 and 16 ticks. A layer is
modeled as one stage whose initiation interval equals its latency; pass-level
pipelining would cut the interval to one pass but the global stage is the
bottleneck regardless. The 4B configuration
keeps every memory parameter identical and scales fabric cycles by the
446M/865M per-shard parameter ratio. `activation_bytes` assumes int8 activations
of the hidden width (4096 or 2560).

Two further configurations ask where the ceiling goes once the memory
wall is removed. `qwen35_9b_7nm_hbm.json` puts the 9B geometry on a
7 nm-class die at the ASAP7 signoff clock of 1.43 GHz (57 ticks per layer)
with one HBM3E stack per layer ASIC (1 TB/s, int4 KV at 16:1, 128 resident
contexts), an 8 GB/s ring link, 1 pJ per MAC and 200 W of static power for
the stacks and FPGA. `qwen35_9b_3nm_hbm.json` extrapolates to 2 GHz
(41 ticks), 1.5 TB/s, 256 contexts, a 16 GB/s link and 0.5 pJ per MAC. The
global index-scan compute placeholder scales with the clock and with the
four times fewer compressed positions. Neither models pass-level
pipelining, so a layer stage still admits one token per layer latency.

Each result also carries an energy model: `mac_energy_pj` times the MACs
per token of every layer and head ASIC (`layer_macs_per_token`,
`head_macs_per_token`) gives compute energy per token, the accumulated
global-stage memory traffic times `memory_energy_pj_per_byte` gives memory
energy, and `static_power_w` is the fixed floor (FPGA, DRAM idle, ASIC
leakage and I/O). The result reports energy per token, compute, memory and
board power at the measured throughput, and the dynamic power of one layer
ASIC. The 3 pJ per MAC in both configurations is the ASAP7 signoff figure
(0.39 pJ at default activity, `fabric/results/pnr_asap7_signoff.json`)
derated for real activity and projected to a 28 nm-class node; a 7 nm-class
die would be about 1 pJ. The sweep adds the same two columns at each
point's global-stage ceiling.

Fabric cycle scales, global phase timing, sustained memory efficiency, and link
throughput are hypotheses to be replaced with RTL and memory-system
measurements. The baseline uses a 10 MHz architecture tick (100 ns), while
preserving the intended microsecond service times and bytes/second. This is a
simulation time quantum rather than the proposed RTL clock and makes long
pipeline sweeps fast.

## Two corrections to every number below

Until September 2026 the memory model was wrong in two ways that both
flattered it, and the figures in this file are the corrected ones.

* **Recurrent-state traffic was missing.** A Gated DeltaNet layer holds a
  512 KB state (32 value heads of 128 x 128, int8) and the delta rule is a
  rank-1 update to all of it. A context's next token is a whole ring behind
  its last one, so the state cannot stay on chip between tokens: every token
  reads it and writes it back. That is 1 MB per recurrent layer per token,
  and for the 9B it is larger than the retrieval traffic the model did
  count. `recurrent_state_bytes` now carries it.
* **Every stage had its own memory port.** A die with four stages could pull
  four times its device bandwidth. There is one memory interface per ASIC, so
  a transfer now reserves it and the other stages of that die wait.

Together they cut the 9B design point from 14.5K to 8.3K tokens/s at int8 KV,
and they make the number of dies matter, which it did not before: memory ports
scale with dies, so splitting the same model over more of them buys
throughput. The 27B on 2 nm goes from 78K tokens/s on four layer dies to 158K
on eight, for the same silicon and twice the masks.

## Current finding

With int8 KV at 4:1 compression, 32 retrieved blocks, and 64 GB/s sustained
bandwidth, both geometries saturate the global stage at about 8.3K tokens/s
with a 128K context, below the 25K to 50K target. The recurrent stages sit at
15 percent (9B) or 10 percent (4B) utilization, the head stages under 5
percent, and the ring links under 4 percent. The global memory interval is
half index scan and half selected-KV transfer. The sweep shows int4 KV plus
8:1 compression recovers 30K tokens/s at 75 GB/s and 40K at 100 GB/s; 16:1
compression with int4 KV reaches the 50K ceiling at 100 GB/s.

Power scales with that throughput, not with the fabric clock. At 3 pJ per
MAC the 9B token costs 24 mJ of compute and about 1.3 mJ of memory traffic,
so the 14.5K tokens/s baseline is 346 W of compute, 20 W of memory and
436 W for the board, 38 W of it in each layer ASIC; the sweep points that
reach 40K tokens/s cost about 1.06 kW. The energy per MAC, which the
process sets, is the lever: at 1 pJ the same points are 150 W and 400 W.

With HBM the wall moves to the fabric. The 7 nm-class configuration
reaches 169K tokens/s at 230 µs mean latency with every layer stage at
99 percent, the global memory phases short, the links at 12 percent and
the head stages at 25 percent; the 3 nm-class one reaches 236K tokens/s
at 165 µs. The limiter in both is the un-pipelined pass rate, one token
per 8192 fabric cycles per stage, so pass-level pipelining (item 4 in the
fabric README) is worth up to four times more, and the 128 or 256
contexts are not yet binding. Power is the other wall: 1.6 kW for the
board at 7 nm (147 W per layer ASIC, 7.9 mJ per token) and 1.2 kW at 3 nm
(102 W per layer ASIC, 4.0 mJ per token), so both are liquid-cooled boxes
rather than cards.

## Sixteen PSRAM devices instead of a DRAM PHY

`qwen35_9b_psram16.json` asks what the 9B does with no DRAM PHY anywhere in
the design: sixteen HPI x16 PSRAM devices per layer ASIC at 1.8 V and
250 MHz DDR, 1 GB/s and 64 MB each, so 16 GB/s and 1 GB per die across about
320 signal balls. int4 KV at 16:1 keeps a 128K context inside the gigabyte.
The point is that a PSRAM is plain CMOS at a fifth of an LPDDR5X bit rate:
no PLL, no per-bit deskew, no training, no analog IP anyone has to license
or design.

| Devices per ASIC | Bandwidth | Tokens/s | Latency | Board |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 4 GB/s | 642 | 36 ms | 87 W |
| 8 | 8 | 1,283 | 18 ms | 103 W |
| 16 | 16 | 2,564 | 9.2 ms | 136 W |
| 24 | 24 | 3,842 | 6.1 ms | 169 W |
| 32 | 32 | 5,076 | 4.7 ms | 201 W |

Throughput is linear in device count, because the memory interface is the
only thing binding: at sixteen devices the recurrent stages sit at 24 percent
and the global stages at 34 percent, so three quarters of the fabric is idle.
Two LPDDR5X-9600 x32 give 77 GB/s, so sixteen PSRAMs are about a fifth of the
LPDDR board and matching it would take seventy-odd devices, which is not a
board. Cheaper state and shorter contexts recover some of it: int4 state and
8 retrieved blocks at a 32K context reach 5.1K tokens/s on the same sixteen
devices.

So the trade is real but it is not a wash. A PSRAM wall buys the removal of
the single largest IP and analog risk in the design, and costs roughly five
times the throughput per board. It also says a PSRAM-based die should hold
more layers than an LPDDR one, since its fabric is mostly idle.

## The high end: a 27B-class model on a 2 nm-class die with HBM

`qwen35_27b_2nm_hbm.json` retargets the appliance to the 27B-class dense
hybrid geometry (`qwen3_5_27b` preset: hidden 5120, 64 layers, FFN 17408,
22.9B body coefficients; the Qwen3.5-27B shapes stand in until a released
27B checkpoint fixes them) on a 2 nm-class die with one HBM4 stack per
layer ASIC: four layer ASICs of sixteen layers, four `R,R,R,G` groups
each, so `global_every` is 4 (the default global layer per ASIC would
quarter the retrieval traffic), and one head ASIC holding the whole LM
head. The die is sized by the fabric mapping at a tenth of the 28 nm-class
density placeholder: 5.71B coefficients in 18,664 tiles is about 152 mm²
of fabric per layer die, and the head fills 42 percent of the same die
(227 mm² at the 3 nm-class 0.15 scale). The fabric runs at 2.5 GHz, so a
layer is four passes of 2560 cycles, 4.1 µs: the deeper tile costs exactly
what the faster clock returns. `qwen35_9b_2nm_hbm.json` is the same die
family with the 9B geometry on eight dies, and `qwen35_27b_3nm_hbm.json`
the 27B on the 3 nm die, as controls. HBM4 is taken as 2 TB/s per die at
25 pJ per byte, the MAC at 0.35 pJ, and the static floor at 250 W.

| Configuration | Dies | Tokens/s | Latency (µs) | Board (W) | Memory per die |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9B, 3 nm-class | 8 + 2 | 190K | 212 | 1,170 | 1.12 TB/s |
| 27B, 2 nm-class | 4 + 1 | 78K | 960 | 1,060 | 1.85 TB/s |
| 27B, 2 nm-class | 8 + 2 | 158K | 529 | 1,880 | 1.86 TB/s |
| 27B, 2 nm-class, int4 state | 8 + 2 | 214K | 340 | 2,320 | 1.68 TB/s |

Four things follow. With the memory model corrected the 27B on 2 nm is
memory-bound, not fabric-bound: sixteen layers on a die means twelve
recurrent states crossing one HBM stack every token, 1.85 of its 2 TB/s. So
the number of dies now matters a great deal, where it did not before.
Splitting the same model over eight dies doubles throughput to 158K because
it doubles the memory ports, and halving the state to int4 takes it to 214K.
The 4 + 1 arrangement is therefore a real trade, half the masks for half the
throughput, not the free win it looked like. Latency follows the same way.
Power is the wall beyond that: 8.4 mJ per token is 2.0 kW of compute at that
throughput, 468 W per layer die on the 4 + 1 board, which is liquid cooling
and a bigger supply than the 9B's 1U; at the 50K tokens/s target the same
appliance draws about 0.7 kW, 100 W per die, and fits the current chassis.
The HBM is not what the model size needs: the sixteen global layers move
8 MB per token per die, which is 1.8 TB/s per die at 234K tokens/s but
400 GB/s at 50K, within reach of GDDR7 or four LPDDR5X channels, so HBM
buys the throughput, not the 27B. And the package grows: 468 W at 0.8 V is
about 650 A per die, which the pinout rules turn into a 55 × 55 array
(`hw/board_27b.yaml`).
