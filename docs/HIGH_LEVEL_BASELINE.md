# Matched wa-hls4ml layer/config baseline

This benchmark tests whether a graph of high-level neural-network layers and
hls4ml configuration parameters is a stronger surrogate input than the LLVM CDFG.
It uses the input schema from `../wa_hls4ml_models/GNN`, but deliberately keeps
the ll-hls4ml experiment contract wherever that is necessary for a valid matched
comparison.

## Deliberate pipeline changes

1. **Input source:** layer architecture and hls4ml settings come from the complete
   merged records under `../data/labels/wa-hls4ml`. The retained
   `hls4ml_config.yml` files are not used because they do not contain the full
   Keras/QKeras `model_config`.
2. **Split identity:** training, validation, synthetic test, and exemplar members
   come directly from the saved 25/50/100/200% CDFG `split_manifest.json` files.
   The original wa-hls4ml split of a record is not reused.
3. **Target identity:** targets come from `../data/tensors/labels.json` in
   ll-hls4ml order: LUT, FF, DSP, BRAM, cycles, II. This is essential because the
   current upstream GNN converter reads `hls_resource_report`, while ll-hls4ml
   predicts post-logic-synthesis `resource_report` values.
4. **Feature conversion:** the upstream 18-column `ModelProcessor` is used for
   layer features. Its downstream preprocessing is retained: 12 normalized
   numerical features, one-hot layer/activation/padding types, sequential layer
   edges, and graph-level strategy/I/O type. Dropout is omitted as upstream does.
   The upstream dataset defines a `batchnorm` column but does not include it in
   either its numerical or categorical model inputs; this matched adapter retains
   that omission rather than silently changing the published feature contract.
5. **BiPC fallback:** the upstream converter drops residual `Add` nodes and then
   leaves precision, reuse factor, and strategy empty for some BiPC exemplars.
   Only for those records, the adapter fills the missing values from the same
   record's model-level HLS configuration. The cache records every affected ID.
6. **Training contract:** target normalization, log-Huber hurdle loss, DSP/BRAM
   threshold heads, early stopping on validation macro SMAPE, metrics, and
   per-family reporting match the CDFG scaling experiment. This isolates the
   representation better than reusing the upstream raw-MSE training script.
7. **Matched model shape:** the default is a 64-wide, three-layer, one-head GATv2
   with residual normalization, multi-pooling, split heads, and the same optimizer
   settings as the primary CDFG curve. This is a controlled representation
   baseline, not a claim to reproduce the paper's large-model training recipe.

## Commands

Build the compact cache without training:

```bash
python scripts/train_high_level.py --build-cache-only
```

Run the primary three-seed scaling benchmark:

```bash
python scripts/train_high_level.py \
  --scales 25 50 100 200 \
  --seeds 42 43 44
```

Runs are resumable through backup checkpoints written every 25 epochs by
default (`--backup-interval` changes this).
Completed runs are skipped unless `--force` is supplied.
