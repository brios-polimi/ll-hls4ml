# hx_memory_dual_archives4_seed42

Single-model wa-hls4ml-style evaluation generated from persisted predictions.

- Model: `hierarchical_memory_dual`
- Device: `cuda`
- Tensor source revision: `not recorded`
- Seed: 42
- Current invocation wall time: 2235.6 seconds
- Cumulative training wall time: 2218.3 seconds
- Wall time to best validation: 2135.340830387984 seconds
- Parameters: 334994
- Peak allocated GPU memory: 2735.5 MiB
- GPU utilization trace: 52.9% mean, 22.0/56.0/78.0% p10/p50/p90, 4.3% zero samples, 12.5s longest low-utilization streak
- Host utilization trace: 14.3% system CPU, 100.0% training-tree CPU, 0.08 MiB/s disk reads
- Mean train seconds/sample/epoch: 0.061156
- Stop reason: `epochs_complete`
- Validation cadence: every 1 epoch(s)
- Checkpoint cadence: every 5 epoch(s)
- Split SHA-256: `e3232c6e25a73d2d8cb6a00d440ffc644b06e620dd28044b7a9e27bdfa511fdb`
- Split sizes: `{"exemplar": 100, "test": 81, "train": 1383, "validation": 66}`
- ll-hls4ml state: `{"commit": "235016b62584c43c538420657c0b585971587184", "dirty": true}`

## Evaluation metrics

| split | scope | macro R² | macro SMAPE (%) | macro RMSE |
| --- | --- | ---: | ---: | ---: |
| exemplar | overall | -6.105 | 120.82 | 12115.14 |
| exemplar | resource | -8.977 | 116.98 | 17636.78 |
| exemplar | timing | -0.360 | 128.51 | 1071.85 |
| test | overall | 0.534 | 43.23 | 239247.61 |
| test | resource | 0.511 | 43.44 | 31346.11 |
| test | timing | 0.580 | 42.81 | 655050.59 |

| split | target | R² | SMAPE (%) | RMSE | median RPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| test | lut | 0.388 | 47.23 | 74620.65 | -20.55 |
| test | ff | 0.573 | 48.97 | 50268.53 | -17.04 |
| test | dsp | 0.749 | 39.83 | 461.70 | 0.00 |
| test | bram | 0.334 | 37.71 | 33.57 | 0.00 |
| test | cycles_max | 0.595 | 39.49 | 643033.00 | 0.75 |
| test | interval_max | 0.565 | 46.12 | 667068.19 | 3.17 |
| exemplar | lut | -1.764 | 113.23 | 43095.51 | -299.58 |
| exemplar | ff | -2.275 | 120.75 | 26645.83 | -365.06 |
| exemplar | dsp | 0.106 | 107.34 | 793.20 | 0.00 |
| exemplar | bram | -31.976 | 126.59 | 12.59 | -415.42 |
| exemplar | cycles_max | -0.573 | 123.92 | 1641.50 | -384.84 |
| exemplar | interval_max | -0.147 | 133.09 | 502.20 | -649.79 |

## Test metrics by kernel family

| test family | samples | macro R² | macro SMAPE (%) |
| --- | ---: | ---: | ---: |
| 2layer | 14 | 0.181 | 24.38 |
| 3layer | 14 | 0.231 | 25.03 |
| conv1d | 6 | 0.067 | 30.65 |
| conv2d | 4 | 0.121 | 58.21 |
| dense_latency | 15 | 0.704 | 37.88 |
| dense_resource | 8 | -13.279 | 34.83 |
| rule4ml | 20 | 0.443 | 77.30 |

## Cohort membership

| split | family | archive | samples |
| --- | --- | --- | ---: |
| train | 2layer | archive_1, archive_2, archive_3, archive_4 | 293 |
| train | 3layer | archive_1, archive_2, archive_3, archive_4 | 288 |
| train | conv1d | archive_1, archive_2, archive_3, archive_4 | 124 |
| train | conv2d | archive_1, archive_3, archive_4 | 57 |
| train | dense_latency | archive_1, archive_2, archive_3, archive_4 | 192 |
| train | dense_resource | archive_1, archive_2, archive_3, archive_4 | 142 |
| train | rule4ml | archive_1, archive_2, archive_3, archive_4 | 287 |
| validation | 2layer | archive_1 | 10 |
| validation | 3layer | archive_1 | 15 |
| validation | conv1d | archive_1 | 6 |
| validation | conv2d | archive_1 | 1 |
| validation | dense_latency | archive_1 | 7 |
| validation | dense_resource | archive_1 | 12 |
| validation | rule4ml | archive_1 | 15 |
| test | 2layer | archive_1 | 14 |
| test | 3layer | archive_1 | 14 |
| test | conv1d | archive_1 | 6 |
| test | conv2d | archive_1 | 4 |
| test | dense_latency | archive_1 | 15 |
| test | dense_resource | archive_1 | 8 |
| test | rule4ml | archive_1 | 20 |
| exemplar | exemplar | archive_1 | 100 |

## Hurdle confusion matrices

| split | target | TN | FP | FN | TP | accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| exemplar | DSP | 21 | 0 | 0 | 79 | 1.000 |
| exemplar | BRAM | 11 | 27 | 0 | 62 | 0.730 |
| test | DSP | 24 | 0 | 0 | 57 | 1.000 |
| test | BRAM | 18 | 1 | 0 | 62 | 0.988 |

Full per-family confusion matrices and per-archive membership are persisted in
`experiment_accounting.json` and `hurdle_confusion.csv`. Presence reliability
data and figures are in `hurdle_calibration.csv` and `figures/`.

## Efficiency and structural diagnostics

Learning curves are persisted in `learning_curves.csv` and plotted against both
epoch and cumulative training wall time. `macro_metrics.csv` separates resource
and timing quality; `structural_error_slices.csv` bins error by graph size, block
length, loop/SCC burden, call depth, and memory burden. If an H0 prediction path
was configured, exact per-sample paired deltas are in `paired_deltas_vs_h0.csv`.

Per-target test and exemplar metrics are in `metrics.csv`. Exact split
membership is in `split_manifest.json`, per-sample predictions are in
`predictions.csv`, and RPE/scatter figures are in `figures/`.

This is only directly comparable with wa-hls4ml when dataset membership,
compiler/graph provenance, targets, and evaluation splits are aligned.
