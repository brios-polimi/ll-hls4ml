# hx_block_attention_scale200_seed42

Single-model wa-hls4ml-style evaluation generated from persisted predictions.

- Model: `hierarchical_block_attention`
- Device: `cuda:0`
- Tensor source revision: `ac09cd77a916ab40f017733efe60e490990f0ff9`
- Seed: 42
- Current invocation wall time: 9551.8 seconds
- Cumulative training wall time: 9496.3 seconds
- Wall time to best validation: 6030.790659542 seconds
- Parameters: 289744
- Peak allocated GPU memory: 1977.3 MiB
- GPU utilization trace: 48.6% mean, 13.0/43.0/93.0% p10/p50/p90, 0.3% zero samples, 7.0s longest low-utilization streak
- Host utilization trace: 70.5% system CPU, 111.1% training-tree CPU, 2.61 MiB/s disk reads
- Mean train seconds/sample/epoch: 0.036768
- Stop reason: `early_stopping`
- Validation cadence: every 1 epoch(s)
- Checkpoint cadence: every 5 epoch(s)
- Split SHA-256: `767cd7ec83b6563ea7c1563e064bdf89eff034179e73601e054c2053d99adc12`
- Split sizes: `{"exemplar": 799, "test": 593, "train": 2742, "validation": 569}`
- ll-hls4ml state: `{"commit": "852845796fc6202d3208f3058aeb9b9412ac093c", "dirty": false}`

## Evaluation metrics

| split | scope | macro R² | macro SMAPE (%) | macro RMSE |
| --- | --- | ---: | ---: | ---: |
| exemplar | overall | -0.275 | 91.73 | 5236.90 |
| exemplar | resource | -0.190 | 80.03 | 7334.45 |
| exemplar | timing | -0.444 | 115.12 | 1041.79 |
| test | overall | 0.581 | 28.15 | 246466.92 |
| test | resource | 0.634 | 26.44 | 27185.12 |
| test | timing | 0.476 | 31.57 | 685030.53 |

| split | target | R² | SMAPE (%) | RMSE | median RPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| test | lut | 0.656 | 24.46 | 75436.09 | 0.68 |
| test | ff | 0.781 | 22.59 | 32584.03 | 1.25 |
| test | dsp | 0.618 | 26.14 | 689.42 | 0.00 |
| test | bram | 0.480 | 32.58 | 30.94 | 0.00 |
| test | cycles_max | 0.520 | 31.14 | 655931.25 | -7.15 |
| test | interval_max | 0.431 | 32.00 | 714129.81 | -4.68 |
| exemplar | lut | 0.675 | 70.51 | 15564.01 | -86.81 |
| exemplar | ff | 0.454 | 80.96 | 12677.42 | -76.89 |
| exemplar | dsp | -0.994 | 85.12 | 1092.96 | 0.00 |
| exemplar | bram | -0.896 | 83.55 | 3.42 | -99.47 |
| exemplar | cycles_max | -0.260 | 103.57 | 1430.13 | -163.27 |
| exemplar | interval_max | -0.628 | 126.68 | 653.45 | -351.75 |

## Test metrics by kernel family

| test family | samples | macro R² | macro SMAPE (%) |
| --- | ---: | ---: | ---: |
| 2layer | 125 | 0.587 | 16.66 |
| 3layer | 116 | 0.694 | 15.45 |
| conv1d | 55 | 0.587 | 32.64 |
| conv2d | 24 | 0.408 | 43.72 |
| dense_latency | 90 | 0.813 | 16.32 |
| dense_resource | 59 | 0.853 | 17.64 |
| rule4ml | 124 | 0.362 | 60.20 |

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
| exemplar | DSP | 148 | 0 | 39 | 612 | 0.951 |
| exemplar | BRAM | 66 | 253 | 4 | 476 | 0.678 |
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
