# hierarchical_scale200_seed42

Single-model wa-hls4ml-style evaluation generated from persisted predictions.

- Model: `hierarchical`
- Device: `cuda`
- Tensor source revision: `32bc0272f1a1f53fbf051371ca7e2728bba60c77`
- Seed: 42
- Current invocation wall time: 92.5 seconds
- Cumulative training wall time: not recoverable for this legacy resumed run
- Validation cadence: every 1 epoch(s)
- Checkpoint cadence: every 5 epoch(s)
- Split SHA-256: `076373d277dbbf0a4fefe6e221cc3392837c8fee4a7cab628c0e3b56593be335`
- Split sizes: `{"exemplar": 886, "test": 593, "train": 2742, "validation": 569}`
- ll-hls4ml state: `{"commit": "235016b62584c43c538420657c0b585971587184", "dirty": true}`

## Evaluation metrics

| split | macro R² | macro SMAPE (%) |
| --- | ---: | ---: |
| test | 0.635 | 24.55 |
| exemplar | 0.245 | 74.81 |

| split | target | R² | SMAPE (%) | RMSE | median RPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| test | lut | 0.699 | 24.51 | 70584.97 | -5.75 |
| test | ff | 0.868 | 23.17 | 25292.86 | -1.40 |
| test | dsp | 0.842 | 19.24 | 443.24 | 0.00 |
| test | bram | 0.667 | 29.09 | 24.77 | 0.00 |
| test | cycles_max | 0.403 | 27.58 | 731547.31 | -5.12 |
| test | interval_max | 0.331 | 23.72 | 774445.19 | -2.12 |
| exemplar | lut | 0.182 | 61.02 | 24802.51 | -52.96 |
| exemplar | ff | 0.448 | 68.83 | 12587.88 | -14.73 |
| exemplar | dsp | 0.502 | 63.47 | 574.05 | 0.00 |
| exemplar | bram | -0.688 | 75.63 | 3.18 | -70.10 |
| exemplar | cycles_max | 0.385 | 101.50 | 987.25 | -184.02 |
| exemplar | interval_max | 0.643 | 78.43 | 304.16 | -61.64 |

## Test metrics by kernel family

| test family | samples | macro R² | macro SMAPE (%) |
| --- | ---: | ---: | ---: |
| 2layer | 125 | 0.486 | 16.24 |
| 3layer | 116 | 0.598 | 16.54 |
| conv1d | 55 | 0.617 | 29.55 |
| conv2d | 24 | 0.571 | 38.49 |
| dense_latency | 90 | 0.931 | 13.20 |
| dense_resource | 59 | 0.888 | 15.09 |
| rule4ml | 124 | 0.497 | 48.26 |

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
| exemplar | DSP | 164 | 0 | 64 | 658 | 0.928 |
| exemplar | BRAM | 72 | 282 | 0 | 532 | 0.682 |
| test | DSP | 192 | 1 | 1 | 399 | 0.997 |
| test | BRAM | 104 | 2 | 0 | 487 | 0.997 |

Full per-family confusion matrices and per-archive membership are persisted in
`experiment_accounting.json` and `hurdle_confusion.csv`.

Per-target test and exemplar metrics are in `metrics.csv`. Exact split
membership is in `split_manifest.json`, per-sample predictions are in
`predictions.csv`, and RPE/scatter figures are in `figures/`.

This is only directly comparable with wa-hls4ml when dataset membership,
compiler/graph provenance, targets, and evaluation splits are aligned.
