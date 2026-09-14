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

Fabric cycle scales, global phase timing, sustained memory efficiency, and link
throughput are hypotheses to be replaced with RTL and memory-system
measurements. The baseline uses a 10 MHz architecture tick (100 ns), while
preserving the intended microsecond service times and bytes/second. This is a
simulation time quantum rather than the proposed RTL clock and makes long
pipeline sweeps fast.

## Current finding

With int8 KV at 4:1 compression, 32 retrieved blocks, and 64 GB/s sustained
bandwidth, both geometries saturate the global stage at about 14.5K tokens/s
with a 128K context, below the 25K to 50K target. The recurrent stages sit at
15 percent (9B) or 10 percent (4B) utilization, the head stages under 5
percent, and the ring links under 4 percent. The global memory interval is
half index scan and half selected-KV transfer. The sweep shows int4 KV plus
8:1 compression recovers 30K tokens/s at 75 GB/s and 40K at 100 GB/s; 16:1
compression with int4 KV reaches the 50K ceiling at 100 GB/s.
