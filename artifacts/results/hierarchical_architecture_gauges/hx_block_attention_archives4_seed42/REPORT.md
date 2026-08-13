# hx_block_attention_archives4_seed42

Single-model wa-hls4ml-style evaluation generated from persisted predictions.

- Model: `hierarchical_block_attention`
- Device: `cuda`
- Tensor source revision: `not recorded`
- Seed: 42
- Current invocation wall time: 2195.4 seconds
- Cumulative training wall time: 2176.6 seconds
- Wall time to best validation: 2093.7166344800207 seconds
- Parameters: 289744
- Peak allocated GPU memory: 1611.3 MiB
- GPU utilization trace: 37.9% mean, 0.0/42.0/66.0% p10/p50/p90, 11.3% zero samples, 9.5s longest low-utilization streak
- Host utilization trace: 16.5% system CPU, 87.5% training-tree CPU, 0.37 MiB/s disk reads
- Mean train seconds/sample/epoch: 0.058601
- Stop reason: `epochs_complete`
- Validation cadence: every 1 epoch(s)
- Checkpoint cadence: every 5 epoch(s)
- Split SHA-256: `e3232c6e25a73d2d8cb6a00d440ffc644b06e620dd28044b7a9e27bdfa511fdb`
- Split sizes: `{"exemplar": 100, "test": 81, "train": 1383, "validation": 66}`
- ll-hls4ml state: `{"commit": "235016b62584c43c538420657c0b585971587184", "dirty": true}`

## Evaluation metrics

| split | scope | macro R² | macro SMAPE (%) | macro RMSE |
| --- | --- | ---: | ---: | ---: |
| exemplar | overall | -2.430 | 116.33 | 8233.53 |
| exemplar | resource | -1.320 | 104.14 | 11350.01 |
| exemplar | timing | -4.651 | 140.71 | 2000.55 |
| test | overall | 0.347 | 42.48 | 313535.42 |
| test | resource | 0.393 | 41.99 | 33704.41 |
| test | timing | 0.254 | 43.48 | 873197.44 |

| split | target | R² | SMAPE (%) | RMSE | median RPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| test | lut | 0.206 | 44.36 | 85017.38 | -24.70 |
| test | ff | 0.592 | 44.81 | 49096.48 | -9.50 |
| test | dsp | 0.473 | 40.94 | 669.41 | 0.00 |
| test | bram | 0.301 | 37.84 | 34.38 | 0.00 |
| test | cycles_max | 0.247 | 42.01 | 877493.88 | 7.67 |
| test | interval_max | 0.262 | 44.95 | 868901.00 | 10.98 |
| exemplar | lut | 0.046 | 104.90 | 25315.96 | -308.21 |
| exemplar | ff | -0.720 | 110.46 | 19314.11 | -260.76 |
| exemplar | dsp | 0.169 | 101.99 | 764.71 | 0.00 |
| exemplar | bram | -4.774 | 99.20 | 5.27 | -207.62 |
| exemplar | cycles_max | -3.506 | 133.41 | 2778.51 | -458.20 |
| exemplar | interval_max | -5.796 | 148.02 | 1222.60 | -1142.79 |

## Test metrics by kernel family

| test family | samples | macro R² | macro SMAPE (%) |
| --- | ---: | ---: | ---: |
| 2layer | 14 | 0.209 | 24.29 |
| 3layer | 14 | 0.272 | 24.39 |
| conv1d | 6 | -0.938 | 37.26 |
| conv2d | 4 | -0.269 | 59.59 |
| dense_latency | 15 | 0.693 | 36.16 |
| dense_resource | 8 | -28.092 | 37.35 |
| rule4ml | 20 | 0.377 | 72.82 |

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
| exemplar | DSP | 21 | 0 | 13 | 66 | 0.870 |
| exemplar | BRAM | 11 | 27 | 0 | 62 | 0.730 |
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
