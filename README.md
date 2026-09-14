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
  against the model with Icarus Verilog, and a yosys flow that measures the
  column datapath on open liberty files (sky130, IHP SG13G2).
* [`clash/`](clash/) contains a synthesizable four-stage ASIC-shard seed with a
  deliberately small fixed-coefficient datapath for early RTL and P&R work.
* [`sim/`](sim/) contains the cycle-stepped 32-stage appliance simulator for
  scheduling, phased global execution, backpressure, link, memory-bandwidth,
  and retrieval-geometry sweep experiments.
* [`training/`](training/) contains the Qwen3.5 checkpoint importer, the
  retrieval candidates, and a layer-wise activation-distillation harness.
* [`hw/`](hw/) contains the block-level board description (eight layer ASICs,
  two head-mode ASICs, sixteen LPDDR5X devices, one PCIe FPGA), its consistency
  checks, and the rendered block diagram.

## Reference geometries

| Preset | Source model | Hidden | Layers | Body params | Per-ASIC shard |
| --- | --- | ---: | ---: | ---: | ---: |
| `qwen3_5_9b` (default) | Qwen3.5-9B-Base | 4096 | 8 × (3 R + 1 G) | 6.9B | 866M |
| `qwen3_5_4b` | Qwen3.5-4B-Base | 2560 | 8 × (3 R + 1 G) | 3.6B | 447M |

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
```
