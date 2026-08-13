# hx_sequence_gru_archives4_seed42

Single-model wa-hls4ml-style evaluation generated from persisted predictions.

- Model: `hierarchical_sequence`
- Device: `cuda`
- Tensor source revision: `not recorded`
- Seed: 42
- Current invocation wall time: 20.3 seconds
- Cumulative training wall time: 2016.4 seconds
- Wall time to best validation: 2016.3146337289945 seconds
- Parameters: 289233
- Peak allocated GPU memory: 2880.3 MiB
- GPU utilization trace: not sampled
- Mean train seconds/sample/epoch: 0.062743
- Stop reason: `checkpoint_evaluation`
- Validation cadence: every 1 epoch(s)
- Checkpoint cadence: every 5 epoch(s)
- Split SHA-256: `e3232c6e25a73d2d8cb6a00d440ffc644b06e620dd28044b7a9e27bdfa511fdb`
- Split sizes: `{"exemplar": 100, "test": 81, "train": 1383, "validation": 66}`
- ll-hls4ml state: `{"commit": "235016b62584c43c538420657c0b585971587184", "dirty": true}`

## Evaluation metrics

| split | scope | macro R² | macro SMAPE (%) | macro RMSE |
| --- | --- | ---: | ---: | ---: |
| exemplar | overall | -3.667 | 117.23 | 10123.18 |
| exemplar | resource | -3.713 | 108.80 | 14195.19 |
| exemplar | timing | -3.574 | 134.07 | 1979.16 |
| test | overall | 0.626 | 44.16 | 183179.74 |
| test | resource | 0.557 | 44.83 | 28519.67 |
| test | timing | 0.763 | 42.83 | 492499.88 |

| split | target | R² | SMAPE (%) | RMSE | median RPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| test | lut | 0.492 | 46.24 | 67991.66 | -23.09 |
| test | ff | 0.648 | 48.68 | 45616.43 | -36.46 |
| test | dsp | 0.776 | 40.49 | 436.48 | 0.00 |
| test | bram | 0.312 | 43.92 | 34.10 | -8.28 |
| test | cycles_max | 0.760 | 39.71 | 495607.69 | 8.20 |
| test | interval_max | 0.766 | 45.95 | 489392.06 | 17.09 |
| exemplar | lut | -0.441 | 102.57 | 31112.70 | -260.36 |
| exemplar | ff | -1.850 | 113.00 | 24858.18 | -285.26 |
| exemplar | dsp | 0.087 | 100.93 | 801.77 | 0.00 |
| exemplar | bram | -12.646 | 118.72 | 8.10 | -319.69 |
| exemplar | cycles_max | -4.454 | 130.08 | 3056.88 | -506.61 |
| exemplar | interval_max | -2.694 | 138.06 | 901.45 | -769.16 |

## Test metrics by kernel family

| test family | samples | macro R² | macro SMAPE (%) |
| --- | ---: | ---: | ---: |
| 2layer | 14 | -0.339 | 30.12 |
| 3layer | 14 | 0.214 | 29.46 |
| conv1d | 6 | -1.755 | 37.84 |
| conv2d | 4 | 0.386 | 52.95 |
| dense_latency | 15 | 0.680 | 33.15 |
| dense_resource | 8 | -8.446 | 35.96 |
| rule4ml | 20 | 0.351 | 75.97 |

## Cohort membership

| split | family | archive | samples |
| --- | --- | --- | ---: |
| train | 2layer | archive_1, archive_2, archive_3, archive_4 | 293 |
| train | 3layer | archive_1, archive_2, archive_3, archive_4 | 288 |
| train | conv1d | archive_1, archive_2, archive_3, archive_4 | 124 |
| train | conv2d | archive_1, archive_3, archive_4 | 57 |
| train | dense_latency | archive_1, archive_2, archive_3, archive_4 | 192 |
| train | dense_resource | archive_1, archive_2, archive_3, archive_4 | 142 |
| train | rule4ml | archive_1, archive_2, archive_3, archive_4 | 287 |
| validation | 2layer | archive_1 | 10 |
| validation | 3layer | archive_1 | 15 |
| validation | conv1d | archive_1 | 6 |
| validation | conv2d | archive_1 | 1 |
| validation | dense_latency | archive_1 | 7 |
| validation | dense_resource | archive_1 | 12 |
| validation | rule4ml | archive_1 | 15 |
| test | 2layer | archive_1 | 14 |
| test | 3layer | archive_1 | 14 |
| test | conv1d | archive_1 | 6 |
| test | conv2d | archive_1 | 4 |
| test | dense_latency | archive_1 | 15 |
| test | dense_resource | archive_1 | 8 |
| test | rule4ml | archive_1 | 20 |
| exemplar | exemplar | archive_1 | 100 |

## Hurdle confusion matrices

| split | target | TN | FP | FN | TP | accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| exemplar | DSP | 21 | 0 | 0 | 79 | 1.000 |
| exemplar | BRAM | 11 | 27 | 1 | 61 | 0.720 |
| test | DSP | 24 | 0 | 0 | 57 | 1.000 |
| test | BRAM | 18 | 1 | 0 | 62 | 0.988 |

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
