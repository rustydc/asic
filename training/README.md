# Layer-wise training plan

Train the three retrieval variants **progressively**, but do not stop at isolated
layer regression. Pure layer-wise fitting cannot correct errors accumulated
across 32 replacements.

1. Cache teacher inputs and outputs for the layer being replaced.
2. Fit that student layer with `layerwise_distill.py`.
3. Insert it into the student checkpoint and regenerate downstream activation
   caches from the partially converted student.
4. After each four-layer `R/R/R/G` shard, jointly tune that shard.
5. After all eight shards, perform a short end-to-end distillation pass with
   language-model, logit-KL, and selected hidden-state losses.
6. Repeat at increasing context lengths; do not begin with 128K examples.

The layer-wise phase is therefore a memory-efficient warm start, not a substitute
for global recovery training.

## Variants

`variants.json` defines equal-FFN-budget recurrent-heavy candidates:

* 64-dimensional FP4-ish index at 4:1 compression;
* 128-dimensional two-bit index at 4:1 compression;
* 128-dimensional FP4-ish index at 8:1 compression.

Here compression is represented by `retrieval_block_size`. Index projections are
fake-quantized during the forward pass with a straight-through estimator.

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

Example:

```bash
python -m training.layerwise_distill \
  --variant index64_fp4_compress4 \
  --layer-index 3 \
  --cache caches/index64/layer03 \
  --initial-layer checkpoints/teacher_layer03.pt \
  --output runs/index64/layer03.pt
```

The repository does not include a teacher checkpoint, licensed training corpus,
or activation caches. The harness cannot launch a meaningful training run until
those inputs and GPU PyTorch are available.

