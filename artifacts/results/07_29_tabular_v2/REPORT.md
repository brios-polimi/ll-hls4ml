# Tensor-v2 eight-family benchmark

Generated: 2026-07-29 11:00:29 CEST

## Scope

This is an engineering benchmark of the current tensor snapshot, not a final
comparison with wa-hls4ml. The graph compiler, static-initializer cleanup, type
encoding, and pragma injection are all research variables that may change.

- Tensor snapshot: `74d9060ab3116fa4`
- Families present: 2layer, 3layer, conv1d, conv2d, dense_resource, dense_latency, rule4ml, exemplar
- Split counts: `{"exemplar": {"exemplar": 400}, "synthetic_test": {"2layer": 52, "3layer": 55, "conv1d": 27, "conv2d": 12, "dense_latency": 54, "dense_resource": 33, "rule4ml": 61}, "train": {"2layer": 293, "3layer": 288, "conv1d": 124, "conv2d": 57, "dense_latency": 192, "dense_resource": 142, "rule4ml": 287}, "validation": {"2layer": 55, "3layer": 57, "conv1d": 22, "conv2d": 12, "dense_latency": 46, "dense_resource": 30, "rule4ml": 52}}`
- Seed: 42
- Device: cpu
- ll-hls4ml state: `{"commit": "0bdd5fff97ae28425ed36052c3bb76bf2e05074c", "dirty": true}` 
- hls4ml_pipeline state: `{"commit": "d0e8db8f0e614a3195f2cf3f161e05205b6996b5", "dirty": true}` 

The synthetic train, validation, and test memberships come from the dataset's
official split labels. Exemplar is never used for fitting or model selection and
is treated as a deliberately shifted external set.

## Synthetic test summary

| model | macro_r2 | macro_smape | median_abs_rpe |
| --- | ---: | ---: | ---: |
| logistic_hurdle_rbf_threshold_graph | 0.645 | 27.858 | 0.513 |
| logistic_hurdle_rbf_expected_graph | 0.645 | 29.505 | 0.582 |
| rbf_svr_graph_no_pragma_edges | 0.607 | 29.841 | 0.677 |
| rbf_svr_graph_no_edges | 0.617 | 29.847 | 0.434 |
| rbf_svr_graph | 0.595 | 29.992 | 0.582 |
| rbf_svr_all_context | 0.592 | 30.200 | 0.788 |
| rbf_svr_core_context | 0.592 | 30.204 | 0.848 |
| rbf_svr_graph_no_literal_features | 0.570 | 30.267 | 0.407 |
| extra_trees_core_context | 0.616 | 30.949 | 1.368 |
| extra_trees_all_context | 0.613 | 31.041 | 1.300 |
| tabular_mlp_core_context | 0.643 | 31.095 | 1.825 |
| extra_trees_graph | 0.614 | 31.376 | 1.467 |
| tabular_mlp_all_context | 0.737 | 32.614 | 1.670 |
| tabular_mlp_graph | 0.692 | 32.776 | 1.098 |
| rbf_svr_graph_no_pragma_arguments | 0.482 | 34.464 | 0.593 |
| rbf_svr_graph_no_pragmas | 0.502 | 34.803 | 0.699 |
| rbf_svr_graph_no_type_features | 0.503 | 41.359 | 1.718 |
| ridge_graph | 0.414 | 45.122 | 1.000 |
| ridge_core_context | 0.380 | 45.271 | 0.971 |
| ridge_all_context | 0.380 | 45.276 | 1.000 |
| rbf_svr_graph_opcodes_size | 0.387 | 50.160 | 1.259 |
| rbf_svr_graph_size_only | 0.261 | 59.586 | 2.219 |
| median | -0.095 | 102.758 | 6.414 |

## Exemplar summary

| model | macro_r2 | macro_smape | median_abs_rpe |
| --- | ---: | ---: | ---: |
| rbf_svr_graph_no_edges | 0.100 | 86.403 | 15.042 |
| logistic_hurdle_rbf_threshold_graph | 0.103 | 86.547 | 13.667 |
| rbf_svr_graph | 0.074 | 87.602 | 16.004 |
| rbf_svr_graph_no_pragma_edges | 0.074 | 87.669 | 16.668 |
| rbf_svr_all_context | 0.123 | 87.687 | 6.697 |
| rbf_svr_core_context | 0.119 | 87.707 | 8.985 |
| rbf_svr_graph_no_literal_features | 0.029 | 87.959 | 14.248 |
| rbf_svr_graph_no_pragma_arguments | -0.001 | 88.072 | 8.710 |
| logistic_hurdle_rbf_expected_graph | 0.095 | 88.428 | 16.591 |
| rbf_svr_graph_no_pragmas | -0.019 | 88.895 | 11.501 |
| tabular_mlp_graph | -0.808 | 90.306 | 39.786 |
| tabular_mlp_all_context | -0.183 | 90.451 | 12.483 |
| tabular_mlp_core_context | -0.028 | 90.672 | 70.001 |
| ridge_graph | -0.525 | 94.393 | 46.610 |
| rbf_svr_graph_opcodes_size | -1.139 | 95.045 | 39.362 |
| ridge_all_context | -0.103 | 95.846 | 59.672 |
| ridge_core_context | -0.206 | 95.973 | 52.785 |
| rbf_svr_graph_no_type_features | -0.220 | 100.481 | 28.998 |
| extra_trees_core_context | -0.326 | 107.151 | 94.044 |
| extra_trees_all_context | -0.559 | 109.367 | 110.665 |
| extra_trees_graph | -1.845 | 112.489 | 115.584 |
| rbf_svr_graph_size_only | -0.703 | 112.980 | 61.828 |
| median | -0.847 | 119.234 | 303.946 |

## Main findings

- `logistic_hurdle_rbf_threshold_graph` is the strongest synthetic-test model in this run;
  `rbf_svr_graph_no_edges` has the lowest exemplar macro SMAPE.
- Neural runs were intentionally omitted. These results establish a tabular reference and cannot by themselves support a verdict about the GNN.
- Exemplar also changes kernel family, tool versions, and synthesis-context
  combinations. Its score therefore measures compound domain shift rather than
  isolated structural generalization.
- Adding pragmas changes synthetic macro SMAPE by not run
  points for the pooled MLP and not run points for the heterogeneous GAT
  relative to their no-pragma ablations. Negative means pragmas help.

### Synthetic-test SMAPE by target

| model | lut | ff | dsp | bram | cycles_max | interval_max | macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_hurdle_rbf_threshold_graph | 23.65 | 28.13 | 24.05 | 31.06 | 27.90 | 32.35 | 27.86 |
| logistic_hurdle_rbf_expected_graph | 23.65 | 28.13 | 33.54 | 31.46 | 27.90 | 32.35 | 29.50 |
| rbf_svr_graph_no_pragma_edges | 23.51 | 27.95 | 35.00 | 32.73 | 27.70 | 32.15 | 29.84 |
| rbf_svr_graph_no_edges | 23.45 | 27.76 | 35.46 | 32.94 | 27.09 | 32.39 | 29.85 |
| rbf_svr_graph | 23.65 | 28.13 | 35.13 | 32.79 | 27.90 | 32.35 | 29.99 |
| rbf_svr_all_context | 24.26 | 26.06 | 36.29 | 32.56 | 28.04 | 33.99 | 30.20 |
| rbf_svr_core_context | 24.33 | 26.04 | 36.25 | 32.56 | 28.04 | 34.00 | 30.20 |
| rbf_svr_graph_no_literal_features | 23.89 | 28.63 | 36.04 | 32.17 | 28.42 | 32.47 | 30.27 |
| extra_trees_core_context | 30.06 | 29.49 | 34.13 | 32.13 | 27.04 | 32.85 | 30.95 |
| extra_trees_all_context | 29.80 | 29.11 | 34.64 | 32.21 | 27.09 | 33.40 | 31.04 |
| tabular_mlp_core_context | 24.81 | 25.42 | 33.62 | 34.91 | 31.38 | 36.44 | 31.10 |
| extra_trees_graph | 29.86 | 30.55 | 35.11 | 32.00 | 27.34 | 33.41 | 31.38 |
| tabular_mlp_all_context | 25.84 | 28.40 | 36.11 | 34.78 | 29.40 | 41.15 | 32.61 |
| tabular_mlp_graph | 25.48 | 27.82 | 34.71 | 36.59 | 31.61 | 40.45 | 32.78 |
| rbf_svr_graph_no_pragma_arguments | 26.34 | 31.69 | 40.65 | 33.20 | 34.15 | 40.74 | 34.46 |
| rbf_svr_graph_no_pragmas | 26.68 | 31.83 | 40.68 | 33.19 | 34.96 | 41.47 | 34.80 |
| rbf_svr_graph_no_type_features | 31.92 | 34.23 | 51.84 | 35.26 | 45.02 | 49.89 | 41.36 |
| ridge_graph | 32.43 | 43.33 | 68.66 | 34.53 | 41.18 | 50.61 | 45.12 |
| ridge_core_context | 32.96 | 43.53 | 68.47 | 34.56 | 41.19 | 50.92 | 45.27 |
| ridge_all_context | 32.99 | 43.54 | 68.47 | 34.54 | 41.19 | 50.93 | 45.28 |
| rbf_svr_graph_opcodes_size | 40.71 | 43.28 | 54.00 | 38.13 | 58.74 | 66.10 | 50.16 |
| rbf_svr_graph_size_only | 46.30 | 51.28 | 76.03 | 43.83 | 66.65 | 73.44 | 59.59 |
| median | 71.69 | 75.59 | 123.34 | 102.19 | 112.61 | 131.12 | 102.76 |

### Exemplar SMAPE by target

| model | lut | ff | dsp | bram | cycles_max | interval_max | macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rbf_svr_graph_no_edges | 61.55 | 106.83 | 73.31 | 65.09 | 102.78 | 108.85 | 86.40 |
| logistic_hurdle_rbf_threshold_graph | 61.21 | 111.35 | 68.87 | 64.77 | 102.08 | 111.00 | 86.55 |
| rbf_svr_graph | 61.21 | 111.35 | 75.29 | 64.68 | 102.08 | 111.00 | 87.60 |
| rbf_svr_graph_no_pragma_edges | 61.12 | 110.35 | 75.91 | 65.05 | 102.45 | 111.13 | 87.67 |
| rbf_svr_all_context | 63.32 | 93.76 | 79.32 | 65.61 | 103.04 | 121.07 | 87.69 |
| rbf_svr_core_context | 62.98 | 95.23 | 78.70 | 65.67 | 103.14 | 120.52 | 87.71 |
| rbf_svr_graph_no_literal_features | 64.17 | 107.33 | 77.88 | 67.63 | 101.81 | 108.92 | 87.96 |
| rbf_svr_graph_no_pragma_arguments | 61.52 | 117.15 | 74.61 | 60.07 | 101.33 | 113.76 | 88.07 |
| logistic_hurdle_rbf_expected_graph | 61.21 | 111.35 | 79.46 | 65.47 | 102.08 | 111.00 | 88.43 |
| rbf_svr_graph_no_pragmas | 61.94 | 121.51 | 76.68 | 59.96 | 101.47 | 111.81 | 88.89 |
| tabular_mlp_graph | 76.32 | 89.45 | 95.41 | 59.73 | 106.74 | 114.19 | 90.31 |
| tabular_mlp_all_context | 91.36 | 88.53 | 87.32 | 50.85 | 112.33 | 112.32 | 90.45 |
| tabular_mlp_core_context | 66.61 | 75.08 | 103.59 | 63.00 | 123.44 | 112.31 | 90.67 |
| ridge_graph | 64.94 | 108.93 | 96.93 | 68.10 | 101.22 | 126.25 | 94.39 |
| rbf_svr_graph_opcodes_size | 80.71 | 123.88 | 93.48 | 68.36 | 101.98 | 101.85 | 95.05 |
| ridge_all_context | 65.25 | 121.96 | 101.55 | 67.80 | 100.83 | 117.68 | 95.85 |
| ridge_core_context | 66.80 | 119.91 | 103.66 | 67.42 | 99.70 | 118.35 | 95.97 |
| rbf_svr_graph_no_type_features | 87.53 | 123.47 | 105.89 | 70.15 | 103.26 | 112.59 | 100.48 |
| extra_trees_core_context | 98.51 | 94.36 | 107.76 | 108.99 | 103.10 | 130.20 | 107.15 |
| extra_trees_all_context | 98.64 | 94.08 | 114.85 | 115.81 | 104.11 | 128.71 | 109.37 |
| extra_trees_graph | 93.00 | 85.46 | 122.31 | 133.77 | 106.48 | 133.92 | 112.49 |
| rbf_svr_graph_size_only | 96.09 | 113.33 | 118.91 | 86.32 | 132.49 | 130.73 | 112.98 |
| median | 107.61 | 120.75 | 107.10 | 87.48 | 141.28 | 151.18 | 119.23 |

Full per-target R², SMAPE, RMSE, RPE quartiles, sample counts, and inference
timings are in `metrics.csv`. Per-sample predictions are in `predictions/`.
Paper-style RPE box plots and log-log prediction scatter plots are in `figures/`.

## Pragma audit

- The graph-level pragma audit was skipped for this run.
- Graphs checked: 0
- Injection anchors: `{}`
- Directives not represented distinctly by the current tensor vocabulary:
  `{}`

The MLP consumes pragma IDs through a pooled pragma embedding. The heterogeneous GAT uses the
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
- Exemplar performance is a useful generalization warning, but it should be
  reported separately from in-distribution synthetic-test performance.
- Do not compare these numbers directly with the published headline results
  unless graph provenance, synthesis contexts, and target definitions are also
  aligned.

## Data-retention consequence

Graph JSON, pragma dumps, and manifests are the durable research artifacts.
Tensors are cheap derived files: rebuild an affected archive when the feature
schema changes rather than adding loader compatibility for old tensor layouts.
Keep one small representative source archive/project set as an end-to-end
compiler/injection canary. Process large missing families one archive at a time;
after graph/tensor counts and failures are verified, remove extracted projects.
Keep the compressed tarball only when its redownload cost is worth the disk
space. Never purge automatically after a partial compilation failure.
