# high_level_gatv2_scale200_seed42

Single-model wa-hls4ml-style evaluation generated from persisted predictions.

- Model: `high_level_gatv2`
- Device: `cuda`
- Tensor source revision: `not recorded`
- Seed: 42
- Current invocation wall time: 317.5 seconds
- Cumulative training wall time: 317.3 seconds
- Validation cadence: every 1 epoch(s)
- Checkpoint cadence: every 5 epoch(s)
- Split SHA-256: `076373d277dbbf0a4fefe6e221cc3392837c8fee4a7cab628c0e3b56593be335`
- Split sizes: `{"exemplar": 886, "test": 593, "train": 2742, "validation": 569}`
- ll-hls4ml state: `{"commit": "235016b62584c43c538420657c0b585971587184", "dirty": true}`

## Evaluation metrics

| split | macro R² | macro SMAPE (%) |
| --- | ---: | ---: |
| test | 0.511 | 25.55 |
| exemplar | -0.225 | 84.28 |

| split | target | R² | SMAPE (%) | RMSE | median RPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| test | lut | 0.592 | 20.53 | 82124.05 | 2.34 |
| test | ff | 0.739 | 24.08 | 35574.98 | 1.02 |
| test | dsp | 0.314 | 24.32 | 923.70 | 0.00 |
| test | bram | 0.645 | 23.85 | 25.57 | 0.00 |
| test | cycles_max | 0.395 | 25.50 | 736670.75 | -1.79 |
| test | interval_max | 0.383 | 35.00 | 743776.12 | 2.92 |
| exemplar | lut | 0.097 | 53.33 | 26064.48 | -49.94 |
| exemplar | ff | -1.883 | 69.50 | 28761.17 | -55.71 |
| exemplar | dsp | 0.157 | 95.82 | 746.83 | -37.09 |
| exemplar | bram | -0.639 | 86.17 | 3.13 | -39.22 |
| exemplar | cycles_max | 0.538 | 88.63 | 855.12 | -39.73 |
| exemplar | interval_max | 0.381 | 112.20 | 400.26 | -36.60 |

## Test metrics by kernel family

| test family | samples | macro R² | macro SMAPE (%) |
| --- | ---: | ---: | ---: |
| 2layer | 125 | 0.792 | 11.70 |
| 3layer | 116 | 0.790 | 11.91 |
| conv1d | 55 | 0.805 | 20.71 |
| conv2d | 24 | 0.314 | 37.47 |
| dense_latency | 90 | -324222484.908 | 37.64 |
| dense_resource | 59 | 0.723 | 17.31 |
| rule4ml | 124 | 0.616 | 47.24 |

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
| exemplar | DSP | 112 | 52 | 44 | 678 | 0.892 |
| exemplar | BRAM | 106 | 248 | 179 | 353 | 0.518 |
| test | DSP | 189 | 4 | 3 | 397 | 0.988 |
| test | BRAM | 100 | 6 | 3 | 484 | 0.985 |

Full per-family confusion matrices and per-archive membership are persisted in
`experiment_accounting.json` and `hurdle_confusion.csv`.

Per-target test and exemplar metrics are in `metrics.csv`. Exact split
membership is in `split_manifest.json`, per-sample predictions are in
`predictions.csv`, and RPE/scatter figures are in `figures/`.

This is only directly comparable with wa-hls4ml when dataset membership,
compiler/graph provenance, targets, and evaluation splits are aligned.
