# hx_sequence_gru_scale200_seed42

Single-model wa-hls4ml-style evaluation generated from persisted predictions.

- Model: `hierarchical_sequence`
- Device: `cuda:0`
- Tensor source revision: `ac09cd77a916ab40f017733efe60e490990f0ff9`
- Seed: 42
- Current invocation wall time: 15381.4 seconds
- Cumulative training wall time: 15331.7 seconds
- Wall time to best validation: 12140.607930784 seconds
- Parameters: 289233
- Peak allocated GPU memory: 3114.3 MiB
- GPU utilization trace: 58.5% mean, 22.0/57.0/97.0% p10/p50/p90, 0.2% zero samples, 5.0s longest low-utilization streak
- Host utilization trace: 69.5% system CPU, 110.6% training-tree CPU, 0.01 MiB/s disk reads
- Mean train seconds/sample/epoch: 0.034562
- Stop reason: `early_stopping`
- Validation cadence: every 1 epoch(s)
- Checkpoint cadence: every 5 epoch(s)
- Split SHA-256: `767cd7ec83b6563ea7c1563e064bdf89eff034179e73601e054c2053d99adc12`
- Split sizes: `{"exemplar": 799, "test": 593, "train": 2742, "validation": 569}`
- ll-hls4ml state: `{"commit": "852845796fc6202d3208f3058aeb9b9412ac093c", "dirty": false}`

## Evaluation metrics

| split | scope | macro R² | macro SMAPE (%) | macro RMSE |
| --- | --- | ---: | ---: | ---: |
| exemplar | overall | -0.278 | 85.59 | 5024.40 |
| exemplar | resource | -0.588 | 81.45 | 7187.21 |
| exemplar | timing | 0.341 | 93.88 | 698.79 |
| test | overall | 0.599 | 24.24 | 268353.00 |
| test | resource | 0.713 | 23.55 | 27134.52 |
| test | timing | 0.371 | 25.64 | 750789.97 |

| split | target | R² | SMAPE (%) | RMSE | median RPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| test | lut | 0.663 | 23.39 | 74646.42 | 1.77 |
| test | ff | 0.770 | 22.87 | 33362.97 | 5.76 |
| test | dsp | 0.797 | 21.98 | 502.23 | 0.00 |
| test | bram | 0.620 | 25.95 | 26.46 | 0.00 |
| test | cycles_max | 0.389 | 25.25 | 740439.38 | 5.13 |
| test | interval_max | 0.354 | 26.02 | 761140.56 | 7.21 |
| exemplar | lut | 0.550 | 72.79 | 18316.19 | -85.11 |
| exemplar | ff | 0.676 | 75.23 | 9759.58 | -81.39 |
| exemplar | dsp | 0.256 | 79.47 | 667.60 | -19.93 |
| exemplar | bram | -3.833 | 98.29 | 5.47 | -150.28 |
| exemplar | cycles_max | 0.444 | 85.32 | 950.42 | -62.26 |
| exemplar | interval_max | 0.238 | 102.45 | 447.15 | -104.11 |

## Test metrics by kernel family

| test family | samples | macro R² | macro SMAPE (%) |
| --- | ---: | ---: | ---: |
| 2layer | 125 | 0.625 | 16.06 |
| 3layer | 116 | 0.740 | 14.39 |
| conv1d | 55 | 0.713 | 28.36 |
| conv2d | 24 | 0.321 | 43.33 |
| dense_latency | 90 | 0.922 | 13.93 |
| dense_resource | 59 | 0.757 | 19.10 |
| rule4ml | 124 | 0.646 | 46.13 |

## Cohort membership

| split | family | archive | samples |
| --- | --- | --- | ---: |
| train | 2layer | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 567 |
| train | 3layer | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 563 |
| train | conv1d | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 243 |
| train | conv2d | archive_1, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 136 |
| train | dense_latency | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 377 |
| train | dense_resource | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 298 |
| train | rule4ml | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 558 |
| validation | 2layer | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 108 |
| validation | 3layer | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 121 |
| validation | conv1d | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 48 |
| validation | conv2d | archive_1, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 34 |
| validation | dense_latency | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 83 |
| validation | dense_resource | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 57 |
| validation | rule4ml | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 118 |
| test | 2layer | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 125 |
| test | 3layer | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 116 |
| test | conv1d | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 55 |
| test | conv2d | archive_1, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 24 |
| test | dense_latency | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 90 |
| test | dense_resource | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 59 |
| test | rule4ml | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 124 |
| exemplar | exemplar | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8 | 799 |

## Hurdle confusion matrices

| split | target | TN | FP | FN | TP | accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| exemplar | DSP | 148 | 0 | 0 | 651 | 1.000 |
| exemplar | BRAM | 54 | 265 | 0 | 480 | 0.668 |
| test | DSP | 192 | 1 | 1 | 399 | 0.997 |
| test | BRAM | 104 | 2 | 0 | 487 | 0.997 |

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
