# Training plan

The reference model reuses the Qwen3.5 dense hybrid weights. Twenty-four Gated
DeltaNet layers, all thirty-two feed-forward blocks, and the embeddings copy over
unchanged, so the recovery problem shrinks to three parts:

1. **Import.** `qwen35_import.py` renames the Hugging Face text weights, drops
   the vision tower, and warm-starts each global layer's retrieval index from
   the top singular directions of its key projection.
2. **Retrieval recovery.** Only the eight global layers change: full causal
   attention becomes a 512-token exact window plus top-K retrieval over
   block-mean keys and values. Freeze everything else and distill those eight
   mixers against the teacher's attention output at 8K to 32K context. This
   fits on one 80 GB GPU because student and teacher share every frozen weight.
3. **Fabric-precision QAT.** Quantize weights to the fixed-fabric precision with
   a straight-through estimator and run a short end-to-end distillation pass.
   Run post-training quantization first to measure the gap; at 4-bit it is
   usually small, at 2-bit or ternary expect this phase to dominate.

Then evaluate at increasing context lengths up to 128K; do not begin there.

## Layer-wise harness

`layerwise_distill.py` fits a single decoder layer against cached activations.
With a Qwen3.5 start it is only needed for the global layers, or as a warm start
when a geometry change (for example an FFN reallocation) invalidates copied
weights. Pure layer-wise fitting cannot correct errors accumulated across
replacements; follow it with joint tuning.

## Variants

`variants.json` defines retrieval candidates. Each names a geometry preset
(`qwen3_5_9b` or `qwen3_5_4b`) plus the retrieval knobs:

* 64-dimensional FP4-ish index at 4:1 compression;
* 128-dimensional two-bit index at 4:1 compression;
* 128-dimensional FP4-ish index at 8:1 compression;
* the 4B geometry with the 128-dimensional FP4-ish index at 4:1.

Compression is represented by `retrieval_block_size`. Index projections are
fake-quantized during the forward pass with a straight-through estimator. FFN
widths are never overridden so the teacher's feed-forward weights stay valid.

## Activation cache contract

Each `.pt` cache file must contain:

```python
{
    "input_hidden": Tensor[examples, sequence, hidden],
    "teacher_output": Tensor[examples, sequence, hidden],
}
```

For layer zero, `input_hidden` is the teacher embedding output. For later layers,
it should come from the already-partially-converted student so the fitted layer
learns to handle upstream approximation error. `teacher_output` remains the
corresponding teacher layer output.

Each cached token costs 16 KB per layer, so this contract suits tens of
millions of tokens, not billions. For larger runs generate activations online
with the teacher resident on the GPU.

Example:

```bash
python -m training.layerwise_distill \
  --variant index64_fp4_compress4 \
  --layer-index 3 \
  --cache caches/index64/layer03 \
  --initial-layer checkpoints/teacher_layer03.pt \
  --output runs/index64/layer03.pt
```

The repository does not include a checkpoint, a training corpus, or activation
caches. The harness cannot launch a meaningful training run until those inputs
and GPU PyTorch are available.
