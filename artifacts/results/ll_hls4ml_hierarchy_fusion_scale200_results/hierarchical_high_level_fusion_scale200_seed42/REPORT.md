# hierarchical_high_level_fusion_scale200_seed42

Single-model wa-hls4ml-style evaluation generated from persisted predictions.

- Model: `hierarchical_high_level_fusion`
- Device: `cuda`
- Tensor source revision: `32bc0272f1a1f53fbf051371ca7e2728bba60c77`
- Seed: 42
- Current invocation wall time: 114.9 seconds
- Cumulative training wall time: not recoverable for this legacy resumed run
- Validation cadence: every 1 epoch(s)
- Checkpoint cadence: every 5 epoch(s)
- Split SHA-256: `076373d277dbbf0a4fefe6e221cc3392837c8fee4a7cab628c0e3b56593be335`
- Split sizes: `{"exemplar": 886, "test": 593, "train": 2742, "validation": 569}`
- ll-hls4ml state: `{"commit": "235016b62584c43c538420657c0b585971587184", "dirty": true}`

## Evaluation metrics

| split | macro R² | macro SMAPE (%) |
| --- | ---: | ---: |
| test | 0.835 | 16.10 |
| exemplar | 0.135 | 66.42 |

| split | target | R² | SMAPE (%) | RMSE | median RPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| test | lut | 0.856 | 14.93 | 48756.98 | -4.85 |
| test | ff | 0.969 | 13.49 | 12196.37 | 0.42 |
| test | dsp | 0.887 | 13.14 | 375.33 | 0.00 |
| test | bram | 0.792 | 19.38 | 19.57 | 0.00 |
| test | cycles_max | 0.724 | 16.31 | 497495.22 | -0.19 |
| test | interval_max | 0.779 | 19.33 | 444814.22 | 0.58 |
| exemplar | lut | 0.654 | 42.35 | 16124.20 | -25.48 |
| exemplar | ff | 0.649 | 56.35 | 10031.55 | -18.13 |
| exemplar | dsp | 0.394 | 68.73 | 633.28 | -3.90 |
| exemplar | bram | -1.664 | 75.57 | 3.99 | -90.66 |
| exemplar | cycles_max | 0.582 | 72.37 | 813.73 | -42.54 |
| exemplar | interval_max | 0.195 | 83.15 | 456.41 | -40.13 |

## Test metrics by kernel family

| test family | samples | macro R² | macro SMAPE (%) |
| --- | ---: | ---: | ---: |
| 2layer | 125 | 0.826 | 9.61 |
| 3layer | 116 | 0.918 | 9.03 |
| conv1d | 55 | 0.898 | 16.47 |
| conv2d | 24 | 0.776 | 24.85 |
| dense_latency | 90 | 0.939 | 13.33 |
| dense_resource | 59 | 0.925 | 12.26 |
| rule4ml | 124 | 0.826 | 31.22 |

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
| exemplar | DSP | 112 | 52 | 76 | 646 | 0.856 |
| exemplar | BRAM | 73 | 281 | 0 | 532 | 0.683 |
| test | DSP | 192 | 1 | 1 | 399 | 0.997 |
| test | BRAM | 104 | 2 | 0 | 487 | 0.997 |

Full per-family confusion matrices and per-archive membership are persisted in
`experiment_accounting.json` and `hurdle_confusion.csv`.

Per-target test and exemplar metrics are in `metrics.csv`. Exact split
membership is in `split_manifest.json`, per-sample predictions are in
`predictions.csv`, and RPE/scatter figures are in `figures/`.

This is only directly comparable with wa-hls4ml when dataset membership,
compiler/graph provenance, targets, and evaluation splits are aligned.
