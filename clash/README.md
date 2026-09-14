# Clash RTL seed

This directory contains a synthesizable **structural seed**, not the complete 8B
model implementation. It makes the first hardware boundary concrete:

* a tagged activation packet containing no persistent context state;
* four registered, independently operating stages in `R/R/R/G` order;
* literal fixed coefficients that synthesize as constants/connectivity rather
  than an addressed model-weight memory;
* saturating accumulator-to-activation requantization;
* a top entity suitable for Verilog or VHDL generation.

The demo activation width is 16 elements. Expanding it directly to 4096 and
copying the placeholder matrix is not the intended implementation. The next
step is to replace a stage shell with a tiled/phase-driven fixed fabric, then use
its synthesis and place-and-route results to select parallelism.

## Generate RTL

With GHC, Cabal, and Clash 1.8 installed:

```bash
cd clash
cabal update
cabal build all
cabal test
cabal exec clash -- --verilog src/FixedLLM.hs -main-is FixedLLM.topEntity
```

Generated HDL appears below `verilog/FixedLLM.topEntity/`. The exact location
may include Clash version-specific naming.

## Deliberate omissions

The current global stage is the same registered fixed-projection shell as the
recurrent stages. It does not yet implement recurrent state RAM, local-window
storage, compressed block construction, top-K FULL retrieval, dynamic Q/K/V
attention, backpressure, or the external-memory interface. Those blocks should
be introduced behind the stable `WorkItem` and stage boundaries, with every
global stage retaining the same FULL-retrieval service contract.

