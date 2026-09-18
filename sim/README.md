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
pipeline. It is also a global-stage analysis only: it does not count the
recurrent-state traffic that shares the same memory interface, so its ceiling
is an upper bound and the full simulation is the number to quote.

## Configurations

Both configurations use the Qwen3.5 head geometry: `kv_heads=4` KV heads of
`head_dim=256`, so a stored position costs 2 KB at int8. Both add two
head-mode ASICs after the last layer chip. Fabric stage times come from the
tile model in `fabric/` at two rows per cycle and 800 MHz: four passes of
2048 cycles per 9B layer (`recurrent_cycles` 102 ticks) and one pass per head
die (26 ticks); the 4B tile is 2560 deep, so 64 and 16 ticks. A layer is
modeled as one stage whose latency is the whole cascade but whose initiation
interval is one pass (`layer_pass_cycles`, a quarter of the layer in every
shipped geometry): the passes have separate via-programmed tiles, nothing is
reused between them, so a pass is free as soon as its token leaves it and the
layer holds one token per pass. The 4B configuration
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
four times fewer compressed positions.

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

## Three corrections to every number below

Until September 2026 the stage model was wrong in three ways, two of which
flattered it and one of which did not. The figures in this file are the
corrected ones.

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
* **A layer admitted one token per layer, not one per pass.** A layer is a
  chain of passes and each pass has its own tiles: nothing is reused, so the
  first pass is free the moment its token moves on. A stage now holds as many
  tokens as fit between its latency and its interval, which also lets a
  token's state write overlap the next token's state read. `layer_pass_cycles`
  carries the pass.

The first two cut the 9B design point from 14.5K to 8.2K tokens/s at int8 KV,
and they make the number of dies matter, which it did not before: memory ports
scale with dies, so splitting the same model over more of them buys
throughput. The 27B on 2 nm runs at 77K tokens/s on four layer dies and 153K
on eight, for the same silicon and twice the masks — which is why the board in
`hw/board_27b.yaml` is now eight layer dies and two head dies.

The third correction was the one I expected to be worth four times, and it is
worth almost nothing, because after the first two the memory is the wall and
not the pass rate. It is exactly neutral on every LPDDR5X and PSRAM
configuration, worth 4 to 8 percent on the 9B with HBM (the 3 nm 9B goes from
234K to 244K), and slightly negative on the 27B, where holding more tokens per
stage lengthens latency against a fixed pool of resident contexts. Take the
memory away — the same 27B configuration with ten times the bandwidth — and it
is worth 2.5 times, 240K to 597K tokens/s, with the global phases and the head
dies as the next wall. That is the honest size of the pass-rate lever: it is
real, and nothing in reach of this design is close enough to the fabric
ceiling to collect it.

## Current finding

With int8 KV at 4:1 compression, 32 retrieved blocks, and 64 GB/s sustained
bandwidth, both geometries run at about 8.2K tokens/s with a 128K context,
below the 25K to 50K target, and the limit is the one memory interface per
die: the four stages on a die want 3 MB of recurrent state plus 4.34 MB of
retrieval every token, and that 7.34 MB at 8.2K tokens/s is 60.3 GB/s of the
63.75 the interface sustains — 95 percent. Nothing else is close. The
recurrent stages are 23 percent busy, the head stages 2 percent and the ring
links 2; the global stages read 62 percent, and that is mostly their own
memory phases. Raising the resident contexts from 32 to 256 buys 5 percent.
The sweep shows int4 KV plus 8:1 compression recovers 30K tokens/s at
75 GB/s and 40K at 100 GB/s for the retrieval half of that traffic; the
state half moves only with a cheaper state or more interfaces.

Power scales with that throughput, not with the fabric clock. At 3 pJ per
MAC the 9B token costs 24 mJ of compute and about 2.3 mJ of memory traffic,
so the 8.2K tokens/s design point is 196 W of compute, 19 W of memory and
285 W for the board, 21 W of it in each layer ASIC. The energy per MAC,
which the process sets, is the lever: at 1 pJ the same point is 65 W of
compute and 154 W for the board.

With HBM the fabric and the memory arrive at the wall together. The
7 nm-class configuration reaches 164K tokens/s at 424 µs mean latency, the
3 nm-class one 244K at 277 µs and the 2 nm-class one 315K at 216 µs. In all
three the layer and global stages are 96 to 98 percent busy *and* the stack
is at 776, 1152 and 1486 GB/s, which is about 90 percent of what the
0.85 efficiency allows; the head stages sit at 25 percent and the links
under 10. So neither doubling the stack nor pipelining the passes moves
these much on its own — 4.7 MB per die per token is the number to attack,
and three quarters of it is recurrent state. Power is the other wall:
1.7 kW for the board at 7 nm (142 W per layer ASIC, 7.9 mJ per token),
1.4 kW at 3 nm (106 W, 4.0 mJ) and 1.4 kW at 2 nm (95 W, 2.8 mJ), so all
three are liquid-cooled boxes rather than cards.

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
| 4 | 4 GB/s | 631 | 40 ms | 86 W |
| 8 | 8 | 1,262 | 21 ms | 102 W |
| 16 | 16 | 2,506 | 10.3 ms | 134 W |
| 24 | 24 | 3,778 | 6.9 ms | 167 W |
| 32 | 32 | 4,984 | 5.2 ms | 198 W |

Throughput is linear in device count, because the memory interface is the
only thing binding: at sixteen devices the recurrent stages sit at 24 percent
and the global stages at 34 percent, so three quarters of the fabric is idle
and pipelining its passes changes the throughput by nothing at all.
Two LPDDR5X-9600 x32 give 77 GB/s, so sixteen PSRAMs are about a fifth of the
LPDDR board and matching it would take seventy-odd devices, which is not a
board. Cheaper state and shorter contexts recover some of it: int4 state and
8 retrieved blocks at a 32K context reach 5.0K tokens/s on the same sixteen
devices, which is what thirty-two devices buy at int8.

So the trade is real but it is not a wash. A PSRAM wall buys the removal of
the single largest IP and analog risk in the design, and costs roughly five
times the throughput per board: the same 9B on two LPDDR5X-9600 per die, with
the same int4 KV at 16:1, runs at 12.4K tokens/s against 2.5K. The board that
carries it is `hw/board_psram.yaml`: sixteen devices are 304 memory signals,
which puts every chip on its own card. It also says a PSRAM-based die should hold
more layers than an LPDDR one, since its fabric is mostly idle.

## The high end: a 27B-class model on a 2 nm-class die with HBM

`qwen35_27b_2nm_hbm.json` retargets the appliance to the 27B-class dense
hybrid geometry (`qwen3_5_27b` preset: hidden 5120, 64 layers, FFN 17408,
22.9B body coefficients; the Qwen3.5-27B shapes stand in until a released
27B checkpoint fixes them) on a 2 nm-class die with one HBM4 stack per
layer ASIC: eight layer ASICs of eight layers, two `R,R,R,G` groups each,
so `global_every` is 4 (the default global layer per ASIC would halve the
retrieval traffic), and two head ASICs splitting the LM head by
vocabulary. The die is sized by the fabric mapping at a tenth of the
28 nm-class density placeholder: 2.86B coefficients in 9,332 tiles is
about 76 mm² of fabric per layer die, and half the head is 22 percent of
the same die. The fabric runs at 2.5 GHz, so a layer is four passes of
2560 cycles, 4.1 µs: the deeper tile costs exactly what the faster clock
returns. `qwen35_9b_2nm_hbm.json` is the same die family with the 9B
geometry on eight dies, and `qwen35_27b_3nm_hbm.json` the 27B on the 3 nm
die, as controls. HBM4 is taken as 2 TB/s per die at 25 pJ per byte, the
MAC at 0.35 pJ, and the static floor at 50 W for the board plus 40 W per
die and stack.

| Configuration | Dies | Tokens/s | Latency (µs) | Board (W) | Memory per die |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9B, 3 nm-class | 8 + 2 | 244K | 277 | 1,446 | 1.15 TB/s |
| 27B, 2 nm-class | 4 + 1 | 77K | 1,580 | 1,042 | 1.45 TB/s |
| 27B, 2 nm-class | 8 + 2 | 153K | 638 | 2,035 | 1.45 TB/s |
| 27B, 2 nm-class, int4 state | 8 + 2 | 216K | 506 | 2,549 | 1.33 TB/s |
| 27B, 3 nm-class | 8 + 2 | 119K | 971 | 2,055 | 1.12 TB/s |

Four things follow. With the memory model corrected the 27B on 2 nm is
memory-bound, not fabric-bound: eight layers on a die is six recurrent
states plus two index scans crossing one HBM stack every token, 9.4 MB, and
at 153K tokens/s that is 1.45 of the stack's 2 TB/s — 85 percent of what the
0.85 sustained efficiency allows. So the number of dies matters a great
deal, where it did not before: the same silicon on four dies of sixteen
layers runs at 77K, half the masks for half the throughput, and halving the
state to int4 takes the eight-die board to 216K. That is why the board in
`hw/board_27b.yaml` is eight layer dies and two head dies. Latency follows
the same way. Power is the wall beyond that: 8.4 mJ per token is 1.3 kW of
compute at that throughput and 2.0 kW for the board, 153 W per layer die,
which is liquid cooling and a bigger supply than the 9B's 1U; at the 50K
tokens/s target the same appliance draws about 1.0 kW, 50 W per die. The
HBM is what the *state* needs, not the model size: 9.4 MB per token per die
is 1.45 TB/s at 153K but 470 GB/s at 50K, within reach of GDDR7 or four
LPDDR5X channels, so HBM buys the throughput, not the 27B. And the package
shrinks with the die count: 153 W at 0.8 V is about 250 A including the
static floor, which the pinout rules turn into a 35 × 35 array rather than
the 55 × 55 the four-die arrangement needed.
