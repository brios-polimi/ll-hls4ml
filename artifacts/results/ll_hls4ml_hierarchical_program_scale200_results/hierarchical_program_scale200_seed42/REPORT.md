# hierarchical_program_scale200_seed42

Single-model wa-hls4ml-style evaluation generated from persisted predictions.

- Model: `hierarchical_program`
- Device: `cuda:0`
- Tensor source revision: `15ba05b8bc32cc011f052b67b9eb74221230614b`
- Seed: 42
- Current invocation wall time: 26166.6 seconds
- Cumulative training wall time: 26046.3 seconds
- Wall time to best validation: 19993.925637264998 seconds
- Parameters: 344456
- Peak allocated GPU memory: 2688.4 MiB
- GPU utilization trace: 51.8% mean, 22.0/46.0/89.0% p10/p50/p90, 0.3% zero samples, 18.0s longest low-utilization streak
- Host utilization trace: 63.0% system CPU, 107.4% training-tree CPU, 5.36 MiB/s disk reads
- Mean train seconds/sample/epoch: 0.065726
- Stop reason: `early_stopping`
- Validation cadence: every 1 epoch(s)
- Checkpoint cadence: every 5 epoch(s)
- Split SHA-256: `076373d277dbbf0a4fefe6e221cc3392837c8fee4a7cab628c0e3b56593be335`
- Split sizes: `{"exemplar": 886, "test": 593, "train": 2742, "validation": 569}`
- ll-hls4ml state: `{"commit": "921b8b1271a9acb6a3bc3032c1f86a384528a9bf", "dirty": false}`

## Evaluation metrics

| split | scope | macro R² | macro SMAPE (%) | macro RMSE |
| --- | --- | ---: | ---: | ---: |
| exemplar | overall | 0.116 | 79.07 | 5594.43 |
| exemplar | resource | -0.017 | 70.88 | 8065.48 |
| exemplar | timing | 0.383 | 95.45 | 652.32 |
| test | overall | 0.617 | 28.10 | 263537.62 |
| test | resource | 0.730 | 26.93 | 25731.35 |
| test | timing | 0.390 | 30.45 | 739150.16 |

| split | target | R² | SMAPE (%) | RMSE | median RPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| test | lut | 0.641 | 27.13 | 77015.48 | -3.24 |
| test | ff | 0.867 | 26.55 | 25399.12 | -3.63 |
| test | dsp | 0.812 | 24.39 | 483.69 | 0.00 |
| test | bram | 0.601 | 29.65 | 27.10 | 0.00 |
| test | cycles_max | 0.428 | 32.20 | 715852.00 | -11.38 |
| test | interval_max | 0.352 | 28.70 | 762448.31 | -6.82 |
| exemplar | lut | 0.508 | 60.49 | 19236.38 | -43.79 |
| exemplar | ff | 0.458 | 73.83 | 12465.29 | -43.02 |
| exemplar | dsp | 0.532 | 63.50 | 556.34 | 0.00 |
| exemplar | bram | -1.566 | 85.70 | 3.92 | -108.14 |
| exemplar | cycles_max | 0.535 | 100.19 | 858.59 | -122.63 |
| exemplar | interval_max | 0.231 | 90.72 | 446.06 | -48.01 |

## Test metrics by kernel family

| test family | samples | macro R² | macro SMAPE (%) |
| --- | ---: | ---: | ---: |
| 2layer | 125 | 0.384 | 21.08 |
| 3layer | 116 | 0.464 | 20.55 |
| conv1d | 55 | 0.524 | 33.22 |
| conv2d | 24 | 0.579 | 39.56 |
| dense_latency | 90 | 0.894 | 16.24 |
| dense_resource | 59 | 0.850 | 19.16 |
| rule4ml | 124 | 0.456 | 50.63 |

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
| exemplar | DSP | 164 | 0 | 46 | 676 | 0.948 |
| exemplar | BRAM | 74 | 280 | 2 | 530 | 0.682 |
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
