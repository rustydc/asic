# Appliance architecture simulator

This is a deterministic, cycle-stepped transaction-level simulator for the
eight-ASIC appliance. It moves tagged work items rather than real tensors and
models the resources that determine system throughput:

* 32 finite-queue layer stages in `R/R/R/G` order;
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
python -m unittest discover -s sim/tests -v
python -m sim.run --config sim/config/baseline.json
python -m sim.run --config sim/config/baseline.json --trace trace.json
python -m sim.sweep --output retrieval-sweep.csv
```

The last command writes a Chrome/Perfetto-compatible trace. Open it in Perfetto
to inspect stage occupancy by context and token.

Warm-up tokens are simulated but excluded from reported throughput and latency,
which reduces pipeline fill bias. `global_max_inflight` controls how many
contexts can occupy different global-engine phases at once. Index and selected
KV transfers still share the configured memory interval.

The sweep reports bytes/token, operation latency, initiation interval, and the
global-stage throughput ceiling across index dimension, index precision,
compression, and bandwidth combinations. It is analytical and therefore much
faster than running every point through the full pipeline.

The baseline is a hypothesis, not a performance claim. It provisionally moves
fixed-weight work toward recurrent stages with `recurrent_weight_scale=1.25`
and `global_weight_scale=0.75`; the PyTorch model exposes corresponding distinct
FFN widths for training experiments. Stage-cycle scales, global phase timing,
sustained memory efficiency, and link throughput must ultimately be replaced
with RTL and memory-system measurements.

The baseline uses a 10 MHz architecture tick (100 ns), while preserving the
intended microsecond service times and bytes/second. This is a simulation time
quantum rather than the proposed RTL clock and makes long pipeline sweeps fast.
