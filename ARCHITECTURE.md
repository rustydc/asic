# Fixed-Weight 8B LLM Inference Appliance

**Status:** Concept / pre-architecture  
**Target:** Dense approximately 8B-parameter inference appliance with 128K context  
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

The starting geometry is a 32-layer, 4096-hidden dense decoder with roughly 8B
parameters. Layers repeat an `R/R/R/G` pattern:

* 24 bounded-state recurrent layers (`R`);
* 8 sparse global-attention layers (`G`);
* a dense SwiGLU feed-forward block in every layer.

The software model permits different FFN widths in recurrent and global layers.
An important training experiment is to move a fixed total parameter budget from
the memory-bound global stages into the recurrent stages. For example, recurrent
FFNs can be widened while global FFNs are narrowed, keeping the approximate 8B
total unchanged. This may recover capacity in the layers doing most sequence
mixing work while moving fixed-fabric cycles toward otherwise underutilized
recurrent stages. It is a quality/latency hypothesis, not yet a frozen geometry.

Eight ASICs each implement four consecutive layers. Every layer is an
independent physical pipeline stage.

```text
FPGA -> ASIC 0 -> ASIC 1 -> ... -> ASIC 7 -> FPGA
        L0-L3     L4-L7           L28-L31

inside each ASIC:
input -> recurrent -> recurrent -> recurrent -> sparse global -> output
```

The eight parts should share arithmetic, control, interfaces, and base physical
design. Only the mask-programmed coefficient connectivity differs.

## Fixed-weight layer fabric

The fixed fabric evaluates `y = Wx` without an addressed weight store or a large
weight bus in the inference path. A low-precision implementation can generate a
small set of coefficient multiples for an input element and use mask-programmed
connectivity to route the selected multiple into output accumulators.

The implementation may consume one or more input dimensions per cycle. The
gating physical-design experiment is place-and-route of a representative layer
macro on the candidate PDK. It must measure:

* parameters per square millimetre and routing congestion;
* input dimensions processed per cycle and achievable clock frequency;
* accumulator, clock-tree, and nonlinear/vector-unit area;
* power, IR drop, and thermal density;
* which masks must change between the eight coefficient variants.

If four approximately 250M-parameter layers do not fit economically, the
fallback is sixteen two-layer devices.

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

The preliminary state budget is 12-24 MB per global layer per 128K-token
context. Since each ASIC owns one global layer, 32 resident contexts consume
approximately 384-768 MB before allocator and metadata overhead. A 2-4 GB local
memory device therefore leaves capacity margin, but bandwidth still requires
trace-based validation.

## Context ownership and work item

The initial system supports at least 32 resident contexts. Every ASIC owns the
state for its local layers, indexed by the context identifier.

```c
struct WorkItem {
    uint8_t context_id;   // five significant bits in revision A
    uint8_t flags;
    uint32_t position;
    activation_t hidden[4096];
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

Each ASIC provisionally receives 2-4 GB of commodity dynamic memory for its one
global layer. The bandwidth target is at least 50 GB/s per ASIC, preferably
75-100 GB/s. Actual compressed-state formats and measured access traces must
close both capacity and bandwidth before architecture freeze.

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

## Development gates

1. **Software model:** recover acceptable quality at 128K and freeze retrieval,
   state, and quantization formats.
2. **Architecture simulator:** validate the 32-stage multi-context schedule,
   local state management, memory traces, backpressure, and packet protocol.
3. **Fixed-weight GDS macro:** demonstrate credible density, routing, timing,
   power, and mask personalization on the target process.
4. **Small silicon:** validate the coefficient/connectivity fabric against the
   physical estimates on an MPW test chip.
5. **Full appliance:** tape out the variants only after the preceding gates pass.

The largest open risks are fixed-connectivity routing density, model-quality
recovery after replacing 24 attention layers, architecture-specific
quantization, external-memory PHY effort, global-stage service time, and yield
across eight large coefficient variants.
