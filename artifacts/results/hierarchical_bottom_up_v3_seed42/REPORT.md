# hierarchical_bottom_up_v3_seed42

Single-model wa-hls4ml-style evaluation generated from persisted predictions.

- Model: `hierarchical`
- Device: `cuda`
- Tensor source revision: `not recorded`
- Seed: 42
- Current invocation wall time: 3005.0 seconds
- Cumulative training wall time: 20504.0 seconds
- Wall time to best validation: 18657.601458767 seconds
- Parameters: 239696
- Peak allocated GPU memory: 661.7 MiB
- GPU utilization trace: not sampled
- Host utilization trace: not sampled
- Mean train seconds/sample/epoch: 0.028369
- Stop reason: `early_stopping`
- Validation cadence: every 1 epoch(s)
- Checkpoint cadence: every 5 epoch(s)
- Split SHA-256: `fd78e658d20afc9d49a610bff9ab8f4c87ff1bde300d8e111d1bbce4ca61209a`
- Split sizes: `{"exemplar": 886, "test": 1133, "train": 5166, "validation": 1064}`
- ll-hls4ml state: `{"commit": "04698fd4a6bca473ce2d14b3b7b6760c39799bac", "dirty": true}`

## Evaluation metrics

| split | scope | macro R² | macro SMAPE (%) | macro RMSE |
| --- | --- | ---: | ---: | ---: |
| exemplar | overall | 0.374 | 74.72 | 4887.83 |
| exemplar | resource | 0.358 | 69.34 | 7008.21 |
| exemplar | timing | 0.404 | 85.50 | 647.07 |
| test | overall | 0.772 | 18.32 | 226323.69 |
| test | resource | 0.847 | 18.30 | 17695.64 |
| test | timing | 0.623 | 18.36 | 643579.78 |

| split | target | R² | SMAPE (%) | RMSE | median RPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| test | lut | 0.812 | 17.94 | 48522.74 | -0.53 |
| test | ff | 0.893 | 15.81 | 21987.93 | -1.71 |
| test | dsp | 0.957 | 14.55 | 251.58 | 0.00 |
| test | bram | 0.725 | 24.92 | 20.31 | -3.17 |
| test | cycles_max | 0.614 | 19.74 | 651321.12 | -2.71 |
| test | interval_max | 0.632 | 16.97 | 635838.44 | -2.81 |
| exemplar | lut | 0.606 | 59.27 | 17207.39 | 1.85 |
| exemplar | ff | 0.643 | 67.17 | 10120.44 | 15.25 |
| exemplar | dsp | 0.255 | 81.74 | 702.46 | 51.51 |
| exemplar | bram | -0.071 | 69.18 | 2.53 | -31.39 |
| exemplar | cycles_max | 0.531 | 86.48 | 861.71 | -69.67 |
| exemplar | interval_max | 0.278 | 84.51 | 432.42 | -33.41 |

## Test metrics by kernel family

| test family | samples | macro R² | macro SMAPE (%) |
| --- | ---: | ---: | ---: |
| 2layer | 246 | 0.665 | 14.83 |
| 3layer | 223 | 0.749 | 12.44 |
| conv1d | 96 | 0.906 | 17.58 |
| conv2d | 57 | 0.616 | 28.33 |
| dense_latency | 177 | -497923.618 | 10.01 |
| dense_resource | 115 | 0.944 | 11.89 |
| rule4ml | 219 | 0.736 | 36.05 |

## Cohort membership

| split | family | archive | samples |
| --- | --- | --- | ---: |
| train | 2layer | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 1061 |
| train | 3layer | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 1052 |
| train | conv1d | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 431 |
| train | conv2d | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 283 |
| train | dense_latency | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 728 |
| train | dense_resource | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 546 |
| train | rule4ml | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 1065 |
| validation | 2layer | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 193 |
| validation | 3layer | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 225 |
| validation | conv1d | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 91 |
| validation | conv2d | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 79 |
| validation | dense_latency | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 154 |
| validation | dense_resource | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 106 |
| validation | rule4ml | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 216 |
| test | 2layer | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 246 |
| test | 3layer | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 223 |
| test | conv1d | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 96 |
| test | conv2d | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 57 |
| test | dense_latency | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 177 |
| test | dense_resource | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 115 |
| test | rule4ml | archive_1, archive_10, archive_11, archive_12, archive_13, archive_14, archive_15, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 219 |
| exemplar | exemplar | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 886 |

## Hurdle confusion matrices

| split | target | TN | FP | FN | TP | accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| exemplar | DSP | 164 | 0 | 90 | 632 | 0.898 |
| exemplar | BRAM | 0 | 354 | 0 | 532 | 0.600 |
| test | DSP | 387 | 1 | 3 | 742 | 0.996 |
| test | BRAM | 0 | 202 | 0 | 931 | 0.822 |

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
