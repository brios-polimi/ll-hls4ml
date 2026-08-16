# hx_memory_dual_scale200_seed42

Single-model wa-hls4ml-style evaluation generated from persisted predictions.

- Model: `hierarchical_memory_dual`
- Device: `cuda:0`
- Tensor source revision: `ac09cd77a916ab40f017733efe60e490990f0ff9`
- Seed: 42
- Current invocation wall time: 11624.5 seconds
- Cumulative training wall time: 30638.5 seconds
- Wall time to best validation: 26196.398376105 seconds
- Parameters: 334994
- Peak allocated GPU memory: 3065.3 MiB
- GPU utilization trace: 59.2% mean, 16.0/60.0/100.0% p10/p50/p90, 0.2% zero samples, 4.0s longest low-utilization streak
- Host utilization trace: 65.0% system CPU, 107.2% training-tree CPU, 2.16 MiB/s disk reads
- Mean train seconds/sample/epoch: 0.049407
- Stop reason: `early_stopping`
- Validation cadence: every 1 epoch(s)
- Checkpoint cadence: every 5 epoch(s)
- Split SHA-256: `767cd7ec83b6563ea7c1563e064bdf89eff034179e73601e054c2053d99adc12`
- Split sizes: `{"exemplar": 799, "test": 593, "train": 2742, "validation": 569}`
- ll-hls4ml state: `{"commit": "852845796fc6202d3208f3058aeb9b9412ac093c", "dirty": false}`

## Evaluation metrics

| split | scope | macro R² | macro SMAPE (%) | macro RMSE |
| --- | --- | ---: | ---: | ---: |
| exemplar | overall | -0.785 | 81.86 | 6796.36 |
| exemplar | resource | -1.395 | 74.09 | 9852.23 |
| exemplar | timing | 0.434 | 97.40 | 684.64 |
| test | overall | 0.671 | 25.57 | 240673.92 |
| test | resource | 0.762 | 25.52 | 22739.59 |
| test | timing | 0.489 | 25.67 | 676542.59 |

| split | target | R² | SMAPE (%) | RMSE | median RPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| test | lut | 0.683 | 24.57 | 72433.22 | 3.66 |
| test | ff | 0.933 | 24.78 | 17993.76 | 2.28 |
| test | dsp | 0.795 | 23.65 | 505.46 | 0.00 |
| test | bram | 0.636 | 29.07 | 25.90 | 0.00 |
| test | cycles_max | 0.510 | 27.34 | 662526.12 | 2.04 |
| test | interval_max | 0.468 | 24.00 | 690559.06 | 1.47 |
| exemplar | lut | 0.252 | 59.71 | 23600.48 | -20.32 |
| exemplar | ff | 0.217 | 72.96 | 15173.80 | -12.32 |
| exemplar | dsp | 0.342 | 78.66 | 627.87 | 0.00 |
| exemplar | bram | -6.389 | 85.03 | 6.76 | -81.17 |
| exemplar | cycles_max | 0.380 | 99.87 | 1003.02 | -164.13 |
| exemplar | interval_max | 0.489 | 94.92 | 366.26 | -117.85 |

## Test metrics by kernel family

| test family | samples | macro R² | macro SMAPE (%) |
| --- | ---: | ---: | ---: |
| 2layer | 125 | 0.517 | 15.10 |
| 3layer | 116 | 0.635 | 15.31 |
| conv1d | 55 | 0.663 | 28.67 |
| conv2d | 24 | 0.673 | 35.86 |
| dense_latency | 90 | 0.897 | 13.68 |
| dense_resource | 59 | 0.829 | 17.29 |
| rule4ml | 124 | 0.415 | 54.92 |

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
| exemplar | DSP | 148 | 0 | 96 | 555 | 0.880 |
| exemplar | BRAM | 104 | 215 | 12 | 468 | 0.716 |
| test | DSP | 191 | 2 | 2 | 398 | 0.993 |
| test | BRAM | 105 | 1 | 0 | 487 | 0.998 |

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
