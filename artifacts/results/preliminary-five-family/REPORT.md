# Preliminary five-family benchmark

Generated: 2026-07-24 20:49:07 CEST

## Scope

This is an engineering benchmark of the current tensor snapshot, not a final
comparison with wa-hls4ml. The graph compiler, static-initializer cleanup, type
encoding, and pragma injection are all research variables that may change.

- Tensor snapshot: `bdac046bad6a9587`
- Families present: 2layer, 3layer, conv1d, conv2d, dense_resource, dense_latency, rule4ml, exemplar
- Split counts: `{"exemplar": {"exemplar": 100}, "synthetic_test": {"2layer": 30, "3layer": 15, "conv1d": 8, "conv2d": 4, "dense_latency": 10, "dense_resource": 9, "rule4ml": 15}, "train": {"2layer": 140, "3layer": 70, "conv1d": 34, "conv2d": 20, "dense_latency": 47, "dense_resource": 40, "rule4ml": 70}, "validation": {"2layer": 30, "3layer": 15, "conv1d": 7, "conv2d": 5, "dense_latency": 10, "dense_resource": 8, "rule4ml": 15}}`
- Seed: 42
- Device: cuda
- ll-hls4ml state: `{"commit": "e62d56e7c444aa35435fb0536b5c9b323ab42f0a", "dirty": true}` 
- hls4ml_pipeline state: `{"commit": "f07cc5fa390cea13ad4a026ef03ffdae9d97bca6", "dirty": true}` 

The synthetic test contains held-out samples from 2layer, 3layer, Conv1D, and
Conv2D. Exemplar is never used for training or validation and is treated as an
external inductive set, matching the paper's distinction in spirit. These are
locally generated splits, not the benchmark's official sample IDs.

## Synthetic test summary

| model | macro_r2 | macro_smape | median_abs_rpe |
| --- | ---: | ---: | ---: |
| rbf_svr | 0.317 | 33.953 | 0.963 |
| extra_trees | 0.509 | 37.659 | 0.592 |
| pooled_mlp | -46.311 | 46.343 | 15.698 |
| ridge | 0.339 | 49.474 | 3.791 |
| rgcn | 0.093 | 50.418 | 4.564 |
| median | -0.072 | 90.735 | 3.043 |

## Exemplar summary

| model | macro_r2 | macro_smape | median_abs_rpe |
| --- | ---: | ---: | ---: |
| ridge | -1.922 | 96.753 | 48.878 |
| pooled_mlp | -1.450 | 99.606 | 20.819 |
| extra_trees | -1.183 | 113.577 | 119.335 |
| rgcn | -10.156 | 115.241 | 112.117 |
| rbf_svr | -9.073 | 118.003 | 157.602 |
| median | -1.666 | 119.551 | 230.341 |

## Main findings

- `rbf_svr` is the strongest current synthetic-test baseline. The
  simple classical models outperform both neural architectures, so further
  model scaling is not justified yet.
- Every model fails on exemplar (`ridge` is merely the least bad).
  This is evidence of severe family/domain shift, not a useful wa-hls4ml
  headline comparison.
- Adding pragmas changes synthetic macro SMAPE by not run
  points for the pooled MLP and not run points for the R-GCN
  relative to their no-pragma ablations. Negative means pragmas help. The
  synthetic effect is modest and does not carry consistently to exemplar, so
  one seed cannot establish that the injected representation generalizes.
- The correct next investment is data/schema validation and group-aware
  evaluation, followed by coverage of the missing kernel families. It is not a
  larger GNN.

### Synthetic-test SMAPE by target

| model | lut | ff | dsp | bram | cycles_max | interval_max | macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rbf_svr | 29.59 | 30.75 | 51.15 | 34.80 | 26.46 | 30.97 | 33.95 |
| extra_trees | 33.57 | 32.90 | 47.10 | 37.49 | 32.79 | 42.11 | 37.66 |
| pooled_mlp | 38.00 | 40.56 | 61.41 | 40.81 | 42.16 | 55.12 | 46.34 |
| ridge | 38.54 | 44.29 | 77.04 | 43.91 | 42.62 | 50.45 | 49.47 |
| rgcn | 43.67 | 41.27 | 46.70 | 48.24 | 58.18 | 64.44 | 50.42 |
| median | 64.37 | 66.68 | 116.71 | 88.04 | 97.36 | 111.25 | 90.73 |

### Exemplar SMAPE by target

| model | lut | ff | dsp | bram | cycles_max | interval_max | macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ridge | 57.69 | 92.85 | 109.63 | 104.59 | 97.49 | 118.28 | 96.75 |
| pooled_mlp | 64.73 | 85.64 | 94.70 | 118.88 | 104.10 | 129.59 | 99.61 |
| extra_trees | 89.55 | 84.09 | 132.83 | 125.01 | 111.79 | 138.19 | 113.58 |
| rgcn | 97.52 | 99.49 | 110.15 | 153.60 | 104.65 | 126.03 | 115.24 |
| rbf_svr | 106.87 | 92.82 | 120.61 | 151.39 | 106.91 | 129.42 | 118.00 |
| median | 105.54 | 114.31 | 114.29 | 79.80 | 148.83 | 154.53 | 119.55 |

Full per-target R², SMAPE, RMSE, RPE quartiles, sample counts, and inference
timings are in `metrics.csv`. Per-sample predictions are in `predictions/`.
Paper-style RPE box plots and log-log prediction scatter plots are in `figures/`.

## Pragma audit

- Graphs checked: 702
- Graphs without injected pragma nodes: 0
- Injection anchors: `{"llvm.sideeffect": 19922, "function_entry": 17548, "named_target": 5202}`
- Directives not represented distinctly by the current tensor vocabulary:
  `{}`

The MLP consumes pragma IDs through a pooled pragma embedding. The R-GCN uses the
same embedding, sends pragma-node messages to anchored instruction/variable
nodes, and also pools the resulting pragma representation. The `*_no_pragmas`
runs zero pragma features and remove pragma edges; their difference from the
normal runs is the first empirical check of whether pragmas currently help.

This does not establish that injection is semantically correct. In particular,
function-entry fallback anchors are coarse, and compiler-carrier and diagnostic
records may overlap. Every directive observed in this snapshot now has a
distinct tensor ID; future unknown directives will still collapse to UNK.

## Interpretation rules

- Prefer per-target results over macro averages; RMSE is scale-dependent.
- RPE follows the paper: `(target - prediction) / (target + 1) * 100`.
  Positive values mean underprediction and negative values mean overprediction.
- DSP and BRAM contain genuine zeros, so RPE and SMAPE use the paper's `+1`
  denominator convention.
- Exemplar performance is the more relevant generalization warning, but the
  present sample is small and the training families are incomplete.
- Do not compare these numbers directly with the published headline results
  until official split membership, all intended families, graph provenance, and
  target definitions are aligned.

## Data-retention consequence

Graph JSON, pragma dumps, and manifests are the durable research artifacts.
Tensors are cheap derived files: rebuild an affected archive when the feature
schema changes rather than adding loader compatibility for old tensor layouts.
Keep one small representative source archive/project set as an end-to-end
compiler/injection canary. Process large missing families one archive at a time;
after graph/tensor counts and failures are verified, remove extracted projects.
Keep the compressed tarball only when its redownload cost is worth the disk
space. Never purge automatically after a partial compilation failure.
