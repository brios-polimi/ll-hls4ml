# canonical_h0_scale200_seed42

Single-model wa-hls4ml-style evaluation generated from persisted predictions.

- Model: `hierarchical`
- Device: `cuda:0`
- Tensor source revision: `68f9e6285cdf131235bf9a7a0220caff23f1c297`
- Seed: 42
- Current invocation wall time: 15325.0 seconds
- Cumulative training wall time: 15285.3 seconds
- Wall time to best validation: 13270.040384365 seconds
- Parameters: 239696
- Peak allocated GPU memory: 604.3 MiB
- GPU utilization trace: 35.3% mean, 10.0/34.0/61.0% p10/p50/p90, 0.3% zero samples, 8.0s longest low-utilization streak
- Host utilization trace: 78.7% system CPU, 115.3% training-tree CPU, 0.02 MiB/s disk reads
- Mean train seconds/sample/epoch: 0.020907
- Stop reason: `early_stopping`
- Validation cadence: every 1 epoch(s)
- Checkpoint cadence: every 5 epoch(s)
- Split SHA-256: `076373d277dbbf0a4fefe6e221cc3392837c8fee4a7cab628c0e3b56593be335`
- Split sizes: `{"exemplar": 886, "test": 593, "train": 2742, "validation": 569}`
- ll-hls4ml state: `{"commit": "04698fd4a6bca473ce2d14b3b7b6760c39799bac", "dirty": false}`

## Evaluation metrics

| split | scope | macro R² | macro SMAPE (%) | macro RMSE |
| --- | --- | ---: | ---: | ---: |
| exemplar | overall | -0.156 | 91.98 | 6063.69 |
| exemplar | resource | -0.121 | 82.44 | 8600.03 |
| exemplar | timing | -0.226 | 111.08 | 991.01 |
| test | overall | 0.585 | 28.09 | 270223.45 |
| test | resource | 0.697 | 27.05 | 26923.33 |
| test | timing | 0.361 | 30.16 | 756823.69 |

| split | target | R² | SMAPE (%) | RMSE | median RPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| test | lut | 0.650 | 27.16 | 76115.05 | -1.82 |
| test | ff | 0.800 | 25.84 | 31158.81 | -1.31 |
| test | dsp | 0.879 | 23.62 | 387.97 | 0.00 |
| test | bram | 0.461 | 31.60 | 31.49 | 0.00 |
| test | cycles_max | 0.350 | 31.49 | 763404.44 | 0.03 |
| test | interval_max | 0.372 | 28.82 | 750242.94 | -1.26 |
| exemplar | lut | 0.412 | 84.26 | 21026.98 | -102.74 |
| exemplar | ff | 0.439 | 88.79 | 12684.44 | -81.31 |
| exemplar | dsp | 0.292 | 76.23 | 684.73 | 39.50 |
| exemplar | bram | -1.626 | 80.48 | 3.97 | -87.97 |
| exemplar | cycles_max | -0.303 | 104.97 | 1436.68 | -58.21 |
| exemplar | interval_max | -0.149 | 117.19 | 545.34 | -135.78 |

## Test metrics by kernel family

| test family | samples | macro R² | macro SMAPE (%) |
| --- | ---: | ---: | ---: |
| 2layer | 125 | 0.240 | 25.79 |
| 3layer | 116 | 0.335 | 25.46 |
| conv1d | 55 | 0.687 | 29.59 |
| conv2d | 24 | 0.344 | 47.20 |
| dense_latency | 90 | 0.922 | 12.81 |
| dense_resource | 59 | 0.907 | 14.89 |
| rule4ml | 124 | 0.592 | 45.87 |

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
| exemplar | exemplar | archive_1, archive_2, archive_3, archive_4, archive_5, archive_6, archive_7, archive_8, archive_9 | 886 |

## Hurdle confusion matrices

| split | target | TN | FP | FN | TP | accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| exemplar | DSP | 164 | 0 | 104 | 618 | 0.883 |
| exemplar | BRAM | 72 | 282 | 3 | 529 | 0.678 |
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
