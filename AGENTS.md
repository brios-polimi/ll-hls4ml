# hls-surrogate-lab agent guide

## Scope

This repository owns learning-side work: graph loading/tensorization, optional
hls4ml modalities, augmentations, features, surrogate models, training,
evaluation, and reporting. `../hls-ir-graph` owns compiler frontends and graph
semantics. `../wa-hls4ml-ingest` owns benchmark acquisition and registry state.

The distribution name is `hls-surrogate-lab`; retain the `ll_hls4ml` Python
package name for notebook and experiment compatibility.

## Start here

- `configs/default.yaml` and `src/ll_hls4ml/config.py`: paths.
- `io/schema.py`: upstream graph contract.
- `data/tensorize.py`, `data/dataset.py`, `data/high_level.py`: modalities.
- `models/registry.py` and `models/fusion.py`: model selection and multimodality.
- `training/`: execution and target semantics.

Scoped `AGENTS.md` files under `data`, `models`, and `training` contain the local
rules needed for those areas.

## Safety and reproducibility

- Treat `../data`, `.test_artifacts`, and `artifacts` as stateful. Never remove,
  rewrite, scan broadly, or regenerate them for validation.
- Use `HLS_SURROGATE_DATA_ROOT` for path overrides; the legacy variable remains
  a compatibility fallback.
- Use standard-library `unittest` with small synthetic inputs. Do not start full,
  distributed, or GPU training unless explicitly requested.
- Do not commit generated checkpoints, caches, plots, predictions, or run
  directories. Preserve them locally and summarize durable findings separately.
