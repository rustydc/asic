# ASIC LLM research prototype

This repository contains the software-model starting point for a fixed-weight
LLM inference appliance built around the Qwen3.5 dense hybrid geometry.

* [`ARCHITECTURE.md`](ARCHITECTURE.md) records the current appliance and pipeline
  decisions, including the balanced FULL-retrieval policy for every global layer.
* [`fixed_llm_poc.py`](fixed_llm_poc.py) is the correctness-oriented PyTorch
  reference model. Its parameter names match the Hugging Face `Qwen3_5` text
  model so released checkpoints load with a prefix rename. It is not an
  optimized long-context training kernel.
* [`fabric/`](fabric/) defines the fixed-weight fabric tile: a via-programmed
  coefficient ROM feeding 64 MAC columns. It holds the bit-exact reference
  model, the coefficient compiler and via-pattern export, the model-to-tile
  mapping with area and latency estimates, synthesizable RTL verified
  against the model with Icarus Verilog, a yosys flow that measures the
  column datapath on open liberty files (sky130, IHP SG13G2, NanGate 45,
  ASAP7), an OpenSTA driver for pre-layout timing, and an OpenROAD
  place-and-route driver with its sky130 and ASAP7 results. `fabric/layer.py`
  and the vector-unit RTL beside the tile are the rest of the layer
  datapath: the norms, the causal convolution, the Gated DeltaNet state
  update and gates, SwiGLU, the residual adds, the rotary embedding and
  the attention core, as a bit-exact fixed-point model with the compiler
  for their constants, checked against the PyTorch reference and against
  the RTL in Icarus. `fabric/memory.py` and its RTL are the memory side:
  the per-context address map, the state DMA, the window and block
  append with the 4-bit index, the index scan and top-K, and the record
  reader into the attention core, with a float twin that reproduces the
  reference retrieval token by token. `fabric/hpi.py` and its RTL are the
  controller for the PSRAM board's chosen device, the AP Memory
  APS512XXN in x16 HPI mode, with a stripe unit over sixteen of them, the
  clock crossing to the core, and the PHY's delay lines with their DLL.
  `fabric/sequencer.py` and its RTL are the token sequencer: a layer's
  dataflow as a program of unit commands, proven against the integer
  layer, scheduled at full size, and run by a microcoded issue engine.
* [`clash/`](clash/) contains a synthesizable four-stage ASIC-shard seed with a
  deliberately small fixed-coefficient datapath for early RTL and P&R work.
* [`sim/`](sim/) contains the cycle-stepped 32-stage appliance simulator for
  scheduling, phased global execution, backpressure, link, memory-bandwidth,
  and retrieval-geometry sweep experiments, with an energy model that turns
  the measured energy per MAC into board power at the simulated throughput.
* [`training/`](training/) contains the Qwen3.5 checkpoint importer, the
  retrieval candidates, and a layer-wise activation-distillation harness.
* [`hw/`](hw/) contains the block-level board descriptions of the host-attached
  1U: the 9B board (eight layer ASICs, two head-mode ASICs, sixteen LPDDR5X
  devices, one FPGA on a PCIe cable, a BMC, redundant CRPS supplies) and
  the 27B-class high-end variant (four sixteen-layer ASICs with HBM4 in
  the package, one head ASIC, liquid cooled), their consistency checks
  including a throughput-driven power budget, the rendered block diagrams,
  a derivation of the ASIC package and ball map from the power model and
  the interface list, a generator that turns either description into a
  KiCad 7 project with the board outline and chassis keep-outs, the ring
  placed as a regular polygon of rotated chips, stackup, all nets, and the
  activation ring routed, and a time-domain signal-integrity check of the
  PSRAM clock nets that settled on one clock per device.

## Reference geometries

| Preset | Source model | Hidden | Layers | Body params | Per-ASIC shard |
| --- | --- | ---: | ---: | ---: | ---: |
| `qwen3_5_9b` (default) | Qwen3.5-9B-Base | 4096 | 8 × (3 R + 1 G) | 6.9B | 866M |
| `qwen3_5_4b` | Qwen3.5-4B-Base | 2560 | 8 × (3 R + 1 G) | 3.6B | 447M |
| `qwen3_5_27b` | 27B-class stand-in (Qwen3.5-27B shapes) | 5120 | 16 × (3 R + 1 G) | 22.9B | 1.43B per group, two per die |

Both presets share the same recurrent heads (32 value × 16 key, 128-dim),
attention heads (16 query × 4 KV, 256-dim), and therefore the same per-context
state formats. Only the fabric width differs, so the 4B is a narrow build of the
same design. Select a preset with `ASICLMConfig.from_preset(name)`; every
variant in `training/variants.json` names one.

## Run

Run the tiny model smoke test and the geometry report in an environment with
PyTorch installed:

```bash
python fixed_llm_poc.py
python -m unittest discover -s training/tests -t .
python -m unittest discover -s sim/tests -t .
python -m unittest discover -s fabric/tests -t .   # RTL tests need iverilog
python -m unittest discover -s hw/tests -t .
```

Import a released checkpoint (needs the `safetensors` package):

```bash
python -m training.qwen35_import --checkpoint /models/Qwen3.5-9B-Base \
    --output checkpoints/asic_init.pt
```

Simulate both geometries:

```bash
python -m sim.run --config sim/config/baseline.json     # 9B
python -m sim.run --config sim/config/qwen35_4b.json    # 4B
python -m sim.run --config sim/config/qwen35_9b_7nm_hbm.json   # 9B, 7 nm-class die, HBM
python -m sim.run --config sim/config/qwen35_9b_3nm_hbm.json   # 9B, 3 nm-class die, HBM
python -m sim.run --config sim/config/qwen35_27b_2nm_hbm.json  # 27B-class, 2 nm-class die, HBM4 (the high end)
```
