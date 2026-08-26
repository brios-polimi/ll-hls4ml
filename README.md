# hls-surrogate-lab

Surrogate-model research for HLS resource and timing prediction from LLVM CDFGs,
optional hls4ml high-level IR, and future backend/modalities. The Python import
name remains `ll_hls4ml` so existing notebooks and recorded experiments continue
to run.

## Repository boundary

| Repository | Responsibility |
| --- | --- |
| `hls-ir-graph` | Backend-aware HLS project to LLVM/directives/graph |
| `wa-hls4ml-ingest` | wa-hls4ml downloads, labels, manifests, registry |
| `hls-surrogate-lab` | Tensorization, augmentation, modeling, training, evaluation |

Graphs and tensors remain outside this repository under `../data` by default.
Override that root with `HLS_SURROGATE_DATA_ROOT`; the legacy
`LL_HLS4ML_DATA_ROOT` variable is still accepted.

## Structure

```text
src/ll_hls4ml/
  io/                 graph schema and discovery
  data/               vocab, tensorization, splits, modalities
    augmentations/    composable learning-time graph transforms
  features/           tabular and graph statistics
  models/             encoders, fusion models, prediction heads, registry
  training/           loaders, loops, targets, telemetry
  reporting/          experiment accounting and comparisons
  viz/                EDA and training visualization
```

The existing Fusion model remains intentionally hls4ml-aware, while its LLVM
encoder and prediction heads are usable independently. New compiler/IR-level
augmentation belongs upstream in `hls-ir-graph`; tensor/graph augmentation
belongs here.

## Development

```bash
pip install -e .
python -m unittest discover -s tests -v
```

Training writes stateful outputs under `artifacts/`, which is ignored. Each run
should remain self-describing through its resolved configuration, split manifest,
revision, metrics, and provenance. See
[docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md).
