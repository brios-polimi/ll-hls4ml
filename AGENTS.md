# ll-hls4ml

## Purpose and boundary

This repository contains exploratory EDA, preprocessing, surrogate modeling, and
training for predicting HLS resource/timing labels from LLVM CDFGs. Upstream graph
production is owned by `../hls4ml_pipeline`; source LLVM/graph data lives outside
this repo under `../data` by default.

This is research software under active exploration. Optimize for understandable,
reproducible experiments and practical iteration. Prefer just-enough structure and
focused checks over framework-building, speculative abstraction, or exhaustive
test infrastructure.

## Start here

- `configs/default.yaml` and `src/ll_hls4ml/config.py`: paths/configuration;
  `LL_HLS4ML_DATA_ROOT` overrides the default data root.
- `src/ll_hls4ml/io/`: CDFG schema, discovery, and JSON loading.
- `src/ll_hls4ml/data/`: vocabulary, JSON-to-`HeteroData` tensorization, datasets,
  and splits.
- `src/ll_hls4ml/features/`: graph statistics and tabular EDA features.
- `src/ll_hls4ml/models/`: MLP/R-GCN implementations and model registry.
- `src/ll_hls4ml/training/`: loaders, loops, target normalization, and distributed helpers.
- `src/ll_hls4ml/viz/`: dataset, EDA, and training plots.
- `scripts/build_vocab.py`, `scripts/build_tensors.py`, `scripts/train.py`, and
  `scripts/train_transductive.py`: batch entry points. Notebooks are the primary
  exploratory workflow.

## Data and artifact rules

- Do not recursively inspect `../data`, `.test_artifacts/`, `artifacts/`, notebook
  outputs, `.pt` tensors, JSON graphs, checkpoints, or run exports. Use a specific
  path/sample only when the task needs it.
- `build_vocab.py` scans graphs and writes the configured vocabulary;
  `build_tensors.py` reads graph JSON and writes tensors. Run them only with an
  intentionally small scope (`--kernel`, `--kernels`, and/or `--max-archives`) for
  validation unless a full run is explicitly requested.
- Training scripts can be compute-intensive and write checkpoints/results. Inspect
  their configuration and use a deliberately small experiment before launching any
  run; do not start distributed training or `torchrun` by default.
- Keep output paths configurable. Do not hard-code machine-specific data paths;
  use YAML configuration or `LL_HLS4ML_DATA_ROOT`.

## Development guidance

- Maintain the handoff contract with `hls4ml_pipeline`: CDFG JSON has nodes, links,
  and synthesis labels; tensorization converts it to PyG `HeteroData`.
- When altering schema, features, tensorization, model inputs, or labels, trace the
  smallest affected chain rather than reading the entire repository.
- Prefer focused smoke checks (imports, config loading, a deliberately tiny input)
  and add tests only where they guard a real research regression. No established
  automated test suite was found.
