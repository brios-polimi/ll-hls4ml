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

## Type feature compatibility

New tensors carry `type_embedding_schema_version = 3`. AP and AC fixed-point
rounding/overflow modes now share feature positions by meaning, rather than by
library enum ordinal. AP feature positions and the embedding width are unchanged;
AC rounding-to-odd and AP sign-magnitude wrap remain distinct modes.

The AP parser also now matches whole enum tokens: previously `AP_RND_CONV`
could be read as `AP_RND`, and `AP_SAT_ZERO` as `AP_SAT`. Other suffixed modes
(`RND_ZERO`, `RND_MIN_INF`, `RND_INF`, `TRN_ZERO`, `SAT_SYM`, `WRAP_SM`) were
affected too.

Older tensors without this field used raw AC enum positions and the old AP
parser. Regenerate tensors containing AC fixed-point types or affected AP modes
from graph JSON; models trained on those old features need retraining or an
explicit migration. Do not mix old and new tensors merely because their shapes
match. The version field records the convention; loaders do not automatically
migrate or reject old tensors. This change does not restore type names missing
from upstream graphs. Version 3 additionally recognizes debug-qualified
`ac_private::iv`, `iv_base`, `iv_conv`, `ac_int`, channel/fifo payloads, unsigned
array lengths (`32U`), and negative fixed-point integer-bit counts. `iv`'s second
template argument is a storage flag; signedness comes from its fourth argument.
Regenerate affected AC tensors produced by earlier versions.
