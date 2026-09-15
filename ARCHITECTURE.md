# Fixed-Weight LLM Inference Appliance

**Status:** Concept / pre-architecture  
**Target:** Dense Qwen3.5-9B-geometry inference appliance with 128K context,
with the Qwen3.5-4B geometry as a narrow first build  
**Primary objective:** Test whether immutable model weights implemented directly
in silicon can sustain high decode throughput in a simple, multi-chip pipeline.

All performance, capacity, cost, and die-size figures in this document are
targets to validate. They are not characterized-silicon results.

## Architectural thesis

The appliance removes the two dominant sources of conventional inference
traffic:

* model coefficients are physical configuration rather than fetched weights;
* most layers use bounded recurrent state, while the remaining layers retrieve
  a sparse subset of compressed long-context state.

The model and each context's persistent state stay local. Only an activation and
its routing metadata move between stages.

```text
moving:              hidden vector, context ID, position, flags
persistent locally:  recurrent state, compressed global-attention context
fixed at manufacture: model weights
```

## Reference model and package partition

The reference model is the Qwen3.5 dense hybrid geometry, which already has the
`R/R/R/G` layer pattern the appliance needs:

* 24 Gated DeltaNet bounded-state recurrent layers (`R`): 32 value heads and
  16 key heads of 128 dimensions, a 4-tap causal convolution, and an output gate;
* 8 gated-attention global layers (`G`): 16 query heads sharing 4 KV heads of
  256 dimensions with rotary embedding on the first 64 dimensions;
* a dense SwiGLU feed-forward block in every layer.

| Preset | Hidden | FFN width | R layer | G layer | Per-ASIC shard | Body | Embedding + head |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3_5_9b` | 4096 | 12288 | 218M | 211M | 866M | 6.93B | 2.03B (untied) |
| `qwen3_5_4b` | 2560 | 9216 | 113M | 108M | 447M | 3.58B | 0.64B (tied) |

The head geometry is identical in both presets, so every per-context state
format, the recurrent update logic, and the retrieval engine are shared. Only
the fabric width changes. Recurrent and global layers are within four percent
of each other in parameters, so the stages are balanced without moving FFN
capacity between them. The software model still permits distinct recurrent and
global FFN widths, but any such change discards the copied feed-forward weights
and must be justified by training results.

The 248K-entry vocabulary makes the LM head 1.0B multiply-accumulates per token
(0.64B for the 4B). At the throughput targets that is beyond an FPGA, and the
head does not fit one layer die's 866M-coefficient fabric. The head is
therefore two additional layer dies in **head mode**: the same base design with
a head personalization, each holding half of the vocabulary rows. The
embedding table lives in FPGA memory.

Head mode needs three features in the base design:

* an input broadcast so all four stage fabrics see the same hidden vector,
  with residual and norm bypassed;
* a top-k plus log-sum-exp reduce on the output, so a head chip emits a few
  hundred bytes instead of 124K logits;
* a two-pin mode strap.

A head chip forwards the incoming work item unchanged and appends its partial
result, so the ring needs no broadcast path. The FPGA merges the two lists and
samples. Head chips have no external memory.

Eight ASICs each implement four consecutive layers, and two more ASICs of the
same design run in head mode. Every layer is an independent physical pipeline
stage.

```text
FPGA -> ASIC 0 -> ASIC 1 -> ... -> ASIC 7 -> HEAD 0 -> HEAD 1 -> FPGA
        L0-L3     L4-L7           L28-L31   vocab     vocab
                                            rows 0-   rows
                                            124159    124160-

inside each layer ASIC:
input -> recurrent -> recurrent -> recurrent -> sparse global -> output
```

All ten parts share arithmetic, control, interfaces, and base physical design.
Only the mask-programmed coefficient connectivity and the mode strap differ.
The board-level capture of this topology is in `hw/`.

## Fixed-weight layer fabric

The fixed fabric evaluates `y = Wx` with no weight bus, cache, or DRAM traffic
in the inference path. Its unit is the **tile** defined in `fabric/`: a
via-programmed ROM of 4096 × 64 int4 coefficients read two rows per cycle,
feeding 64 multiply-accumulate columns. Each activation's coefficient
multiples are formed once per bank; every column selects its multiple through
the ROM word and adds or subtracts it. The coefficient never leaves the tile;
the read is a few tens of micrometres of bitline.

Coefficients are ROM bits, not wiring. A coefficient that is literally a wire
cannot be time-multiplexed, so pure wiring would need one adder per
coefficient. A mask-programmed bit read by a wordline is what lets the
arithmetic be shared, and a via ROM is denser than wiring or SRAM in any case.
Each of the ten chips is therefore the same base mask set plus one via mask.

A 9B layer die maps to 3306 tiles at 99.9 percent utilization, about 240 mm²
at a 28 nm-class node with the ROM cell still a placeholder and the MAC
columns calibrated by open-tooling synthesis on four libraries from 130 nm
to a predictive 7 nm. Static timing of the column after OpenROAD placement,
clock-tree synthesis and global routing gives 241 MHz on sky130 and
1.42 GHz on ASAP7, which brackets the 28 nm clock at roughly 0.6 to
1.2 GHz; the 800 MHz assumption sits inside that range. A layer is four
sequential passes of 2048 cycles, 10.2 µs at 800 MHz. The head die is
1940 tiles.

The gating physical-design experiment is a tile on the candidate PDK, and it
must measure:

* via-ROM bit-cell area and read energy, which set the ROM half of the die;
* column datapath area at one, two, and four rows per cycle;
* the ROM-read-plus-add cycle time;
* power, IR drop, and thermal density at full pass rate;
* that the via layer is the only mask that changes between variants.

If four approximately 215M-parameter layers do not fit economically, the
fallbacks are the 4B geometry (four approximately 110M-parameter layers per
device) or sixteen two-layer devices.

## Long-context state

Each global layer combines:

1. exact causal attention over a recent window;
2. compressed long-term key/value blocks;
3. a learned, low-dimensional block index;
4. top-K block retrieval followed by dynamic attention over the selected data.

Block selection is intentional: selected context reads become contiguous memory
bursts rather than arbitrary token gathers. Representative training variables
are a 256-1024 token exact window, approximately 4:1 long-term compression, a
128-dimensional index, and 256-1024 retrieved compressed positions. These values
remain software parameters until model quality and hardware traces justify
freezing them.

### Balanced global-layer policy

Every global layer performs the same **FULL** retrieval algorithm against its
own local context. The design does **not** alternate FULL and REUSE layers and
does not send candidate lists between ASICs.

This choice deliberately trades redundant index work for a regular pipeline:

* all eight global stages have the same latency and bandwidth envelope;
* the inter-chip work item remains small and deterministic;
* no global layer depends on retrieval decisions made from another layer's
  representation;
* scheduling does not need separate FULL/REINDEX/REUSE stage classes.

Hierarchical or chunked index scanning remains an allowed implementation detail
inside every global stage. It must preserve the same observable FULL-retrieval
semantics and a common stage budget.

A stored position carries four 256-dimensional KV heads, 2 KB at int8 or 1 KB
at int4. At 4:1 compression a 128K-token context therefore needs about 64 MB
per global layer at int8 plus the 128-dimensional index, and about 32 MB at
int4. Since each ASIC owns one global layer, 32 resident contexts consume
roughly 1-2 GB before allocator and metadata overhead. A 2-4 GB local memory
device still fits, but the margin is smaller than the earlier single-KV-head
estimate, and the simulator shows bandwidth, not capacity, is the binding
constraint (see "Performance targets").

## Context ownership and work item

The initial system supports at least 32 resident contexts. Every ASIC owns the
state for its local layers, indexed by the context identifier.

```c
struct WorkItem {
    uint8_t context_id;   // five significant bits in revision A
    uint8_t flags;
    uint32_t position;
    activation_t hidden[4096];   // 2560 in the 4B build
};
```

Candidate IDs are intentionally absent. A simple PoC may statically reserve
equal memory regions for every context.

## Pipeline scheduling

Autoregression prevents unresolved tokens from one context from occupying
multiple stages, because token `N+1` cannot enter layer 0 until token `N` has
completed layer 31 and sampling. Independent contexts can occupy different
stages simultaneously. Thirty-two runnable contexts nominally fill the pipeline;
additional contexts provide slack for memory and control variability.

The slowest stage sets aggregate throughput. Because all global stages use the
same policy, their target service time must be balanced explicitly against one
another and should be made as deterministic as practical.

The architecture simulator represents a global operation as index scan, top-K,
selected-KV transfer, attention, and output phases. Different contexts may
occupy different phase engines concurrently, while index and KV traffic contend
for the same local memory interface. Reported steady-state metrics exclude a
configurable warm-up interval. Memory bandwidth is derated by an explicit
efficiency factor and includes context append writes.

## Interfaces and memory

PCIe terminates at the controller FPGA. Adjacent ASICs use a narrow,
source-synchronous, ready/valid packet link with framing and CRC. A candidate
revision-A link is 32 data bits at 250 MHz DDR (2 GB/s raw). At 50K work items/s,
a BF16 4096-element hidden vector consumes about 400 MB/s before framing, leaving
substantial raw-link margin.

Each layer ASIC provisionally receives two LPDDR5X x32 devices (4 GB, about
77 GB/s raw at 9600 MT/s) for its one global layer. The bandwidth target is at
least 50 GB/s per ASIC, preferably 75-100 GB/s; GDDR6 is the fallback if the
sweep demands the top of that range. Head ASICs have no external memory.
Actual compressed-state formats and measured access traces must close both
capacity and bandwidth before architecture freeze.

The FPGA owns PCIe Gen4 (x8 wired, x4 sufficient) and 4 GB of DDR4 for the
embedding table and context metadata. Power-over-Ethernet cannot supply a
board that draws hundreds of watts, and PCIe gives lower host latency than a
network hop, so the first board is a host-attached 1U: one board in a rack
chassis with redundant CRPS supplies and a BMC, reached over a SlimSAS PCIe
cable from the host, with an optional SFP+ cage for a later standalone mode
(see `hw/README.md`). The energy model moved it out of a PCIe slot: at 3 pJ
per MAC the board draws 0.5 to 1.6 kW across the throughput range.

The FPGA owns the host protocol, scheduling, sampling, context allocation,
bring-up, telemetry, error recovery, and performance counters. Speculative
generation is deferred until ordinary autoregressive decode is characterized.

## Performance targets

| Condition | Architectural target |
| --- | ---: |
| 128K, one context | 2K-4K token/s |
| 128K, pipeline full | 25K-50K token/s |
| Short context, one context | 4K-8K token/s |
| Short context, pipeline full | 40K+ token/s |

These targets imply roughly 250-500 microseconds of end-to-end single-context
latency and a 20-40 microsecond worst-stage service time. A cycle-accurate model
must derive both figures from fixed-fabric, memory, link, and queue timing rather
than assume them.

The transaction-level simulator with the Qwen3.5 head geometry, int8 KV, 4:1
compression, 32 retrieved blocks, and 64 GB/s sustained bandwidth saturates the
global stage at about 14.5K tokens/s for both presets at 128K context, with the
recurrent stages under 10 percent busy. Half of the global memory interval is
the index scan and half is the selected-KV transfer. Reaching the 25K-50K
target needs some combination of int4 KV storage, 8:1 or 16:1 compression,
fewer retrieved blocks, and 75-100 GB/s sustained bandwidth. Those are now the
first parameters the software model must qualify.

Power follows throughput. The column datapath measures 0.39 pJ per
multiply-accumulate on ASAP7 at default activity after detailed routing and
extraction (`fabric/results/pnr_asap7_signoff.json`); derated for real
activity and projected to a 28 nm-class node that is about 3 pJ, so a 9B
token costs roughly 24 mJ of compute across the ten ASICs. The simulator and
the board model both carry that figure: the 14.5K tokens/s baseline is about
440 W of board load, 25K tokens/s about 700 W, and 50K tokens/s 1.3 kW, with a
layer ASIC's core rail between 50 A and 180 A. A 7 nm-class process at about
1 pJ per MAC divides all of those by three. The choice of node is therefore
as much a power-supply decision as an area one, and the on-die grid must be
sized from measured current: the ASAP7 slice lost half its supply on the
platform's default grid at 1.7 W/mm².

## Development gates

1. **Software model:** import the Qwen3.5 weights, recover acceptable quality
   at 128K with the retrieval replacement in the eight global layers, and
   freeze retrieval, KV precision, compression, and fabric quantization formats.
2. **Architecture simulator:** validate the 32-stage multi-context schedule,
   local state management, memory traces, backpressure, and packet protocol.
3. **Fixed-weight tile macro:** demonstrate credible ROM density, column
   datapath timing, power, and via-mask personalization on the target process,
   driven by the coefficient compiler in `fabric/`.
4. **Small silicon:** validate the coefficient/connectivity fabric against the
   physical estimates on an MPW test chip. The head-mode personalization is
   the natural vehicle: one matrix, no state, no memory interface.
5. **Full appliance:** tape out the eight layer variants and the two head
   variants only after the preceding gates pass.

The largest open risks are fixed-connectivity routing density, global-stage
memory bandwidth at the Qwen3.5 KV width, model-quality loss from replacing
full attention with windowed retrieval in the eight global layers,
architecture-specific weight quantization (especially below 4 bits on the 4B),
placement of the 1B-parameter LM head, external-memory PHY effort, power
delivery at hundreds of watts per board and over 100 A per die, and yield
across eight large coefficient variants.
