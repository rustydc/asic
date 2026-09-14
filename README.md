# ASIC LLM research prototype

This repository contains the software-model starting point for a fixed-weight,
approximately 8B-parameter LLM inference appliance.

* [`ARCHITECTURE.md`](ARCHITECTURE.md) records the current appliance and pipeline
  decisions, including the balanced FULL-retrieval policy for every global layer.
* [`fixed_llm_poc.py`](fixed_llm_poc.py) is the correctness-oriented PyTorch
  reference model. It is not an optimized long-context training kernel.
* [`clash/`](clash/) contains a synthesizable four-stage ASIC-shard seed with a
  deliberately small fixed-coefficient datapath for early RTL and P&R work.
* [`sim/`](sim/) contains the cycle-stepped 32-stage appliance simulator for
  scheduling, phased global execution, backpressure, link, memory-bandwidth,
  and retrieval-geometry sweep experiments.
* [`training/`](training/) defines the three retrieval candidates and a
  layer-wise activation-distillation harness.

Run the tiny model smoke test in an environment with PyTorch installed:

```bash
python fixed_llm_poc.py
```

To test a roughly parameter-neutral shift of FFN capacity toward recurrent
layers in the full geometry, set `recurrent_intermediate_size=16_384` and
`global_intermediate_size=8_192`. Train and evaluate this allocation before
freezing it; the simulator's independent recurrent/global weight-scale knobs
only estimate its timing effect.
