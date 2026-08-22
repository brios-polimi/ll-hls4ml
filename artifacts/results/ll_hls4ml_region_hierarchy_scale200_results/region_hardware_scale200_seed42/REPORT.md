# region_hardware_scale200_seed42

Single-model wa-hls4ml-style evaluation generated from persisted predictions.

- Model: `hierarchical_region`
- Device: `cuda:0`
- Tensor source revision: `68f9e6285cdf131235bf9a7a0220caff23f1c297`
- Seed: 42
- Current invocation wall time: 3411.2 seconds
- Cumulative training wall time: 22704.6 seconds
- Wall time to best validation: 19232.185054847996 seconds
- Parameters: 323604
- Peak allocated GPU memory: 771.0 MiB
- GPU utilization trace: 31.4% mean, 8.0/25.5/65.0% p10/p50/p90, 0.4% zero samples, 8.0s longest low-utilization streak
- Host utilization trace: 67.8% system CPU, 109.4% training-tree CPU, 0.08 MiB/s disk reads
- Mean train seconds/sample/epoch: 0.036446
- Stop reason: `early_stopping`
- Validation cadence: every 1 epoch(s)
- Checkpoint cadence: every 5 epoch(s)
- Split SHA-256: `076373d277dbbf0a4fefe6e221cc3392837c8fee4a7cab628c0e3b56593be335`
- Split sizes: `{"exemplar": 886, "test": 593, "train": 2742, "validation": 569}`
- ll-hls4ml state: `{"commit": "04698fd4a6bca473ce2d14b3b7b6760c39799bac", "dirty": false}`

## Evaluation metrics

| split | scope | macro R² | macro SMAPE (%) | macro RMSE |
| --- | --- | ---: | ---: | ---: |
| exemplar | overall | -1.529 | 99.87 | 5969.37 |
| exemplar | resource | -1.978 | 88.87 | 8368.37 |
| exemplar | timing | -0.631 | 121.86 | 1171.35 |
| test | overall | 0.522 | 29.07 | 270459.56 |
| test | resource | 0.598 | 27.64 | 29599.61 |
| test | timing | 0.369 | 31.93 | 752179.44 |

| split | target | R² | SMAPE (%) | RMSE | median RPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| test | lut | 0.608 | 28.60 | 80559.43 | -1.17 |
| test | ff | 0.714 | 27.07 | 37248.18 | -0.61 |
| test | dsp | 0.752 | 22.75 | 555.46 | 0.00 |
| test | bram | 0.320 | 32.14 | 35.39 | 0.00 |
| test | cycles_max | 0.367 | 32.15 | 753485.81 | -0.33 |
| test | interval_max | 0.371 | 31.71 | 750873.06 | -0.30 |
| exemplar | lut | 0.539 | 78.79 | 18626.53 | -68.02 |
| exemplar | ff | 0.316 | 92.29 | 14009.36 | -91.40 |
| exemplar | dsp | -0.041 | 88.65 | 829.98 | 52.64 |
| exemplar | bram | -8.728 | 95.74 | 7.63 | -118.61 |
| exemplar | cycles_max | -0.960 | 113.08 | 1761.92 | -218.41 |
| exemplar | interval_max | -0.303 | 130.65 | 580.78 | -445.09 |

## Test metrics by kernel family

| test family | samples | macro R² | macro SMAPE (%) |
| --- | ---: | ---: | ---: |
| 2layer | 125 | 0.376 | 20.73 |
| 3layer | 116 | 0.471 | 20.32 |
| conv1d | 55 | 0.540 | 29.95 |
| conv2d | 24 | 0.065 | 54.92 |
| dense_latency | 90 | 0.897 | 17.01 |
| dense_resource | 59 | 0.780 | 20.40 |
| rule4ml | 124 | 0.363 | 53.14 |

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
| exemplar | DSP | 164 | 0 | 149 | 573 | 0.832 |
| exemplar | BRAM | 70 | 284 | 11 | 521 | 0.667 |
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
