# Natural-loop CPU information probe

This is an information check, not a proposed high-level fusion model. Hyperparameters are selected by archive-grouped CV within train; the official validation split is reported once. Test and exemplar are not read.

Retained 554 samples; missing LLVM files: 0.

| split | family | N |
| --- | --- | --- |
| train | 2layer | 40 |
| train | 3layer | 40 |
| train | conv1d | 40 |
| train | conv2d | 40 |
| train | dense_latency | 40 |
| train | dense_resource | 40 |
| train | rule4ml | 40 |
| validation | 2layer | 40 |
| validation | 3layer | 40 |
| validation | conv1d | 40 |
| validation | conv2d | 34 |
| validation | dense_latency | 40 |
| validation | dense_resource | 40 |
| validation | rule4ml | 40 |

## Training CV

| features | min leaf | OOF SMAPE |
| --- | --- | --- |
| control | 1 | 35.88 |
| control | 2 | 36.96 |
| control | 4 | 40.88 |
| control | 8 | 47.54 |
| control+region | 1 | 36.12 |
| control+region | 2 | 36.62 |
| control+region | 4 | 40.65 |
| control+region | 8 | 47.52 |

## Official validation

| target | control | +region | delta |
| --- | --- | --- | --- |
| macro | 34.51 | 34.29 | -0.22 |
| lut | 29.99 | 29.37 | -0.62 |
| ff | 32.19 | 31.13 | -1.06 |
| dsp | 31.67 | 31.20 | -0.46 |
| bram | 33.59 | 33.51 | -0.08 |
| cycles_max | 36.94 | 36.98 | +0.04 |
| interval_max | 42.69 | 43.56 | +0.86 |

Paired sample-bootstrap 95% CI for the macro delta: [-0.71, +0.25] SMAPE.

| family | N | control | +region | delta |
| --- | --- | --- | --- | --- |
| 2layer | 40 | 18.42 | 17.71 | -0.72 |
| 3layer | 40 | 15.83 | 15.40 | -0.43 |
| conv1d | 40 | 38.62 | 39.02 | +0.41 |
| conv2d | 34 | 42.80 | 43.49 | +0.69 |
| dense_latency | 40 | 23.50 | 23.50 | +0.01 |
| dense_resource | 40 | 27.65 | 27.42 | -0.24 |
| rule4ml | 40 | 76.01 | 74.89 | -1.12 |

## Region-structure diversity

| family | loops min/med/max | nested min/med/max | unique summaries |
| --- | --- | --- | --- |
| 2layer | 10/10/10 | 2/2/2 | 1 |
| 3layer | 15/15/15 | 3/3/3 | 1 |
| conv1d | 38/69/140 | 10/23/54 | 80 |
| conv2d | 43/80/163 | 15/32/75 | 73 |
| dense_latency | 29/47/64 | 8/12/16 | 79 |
| dense_resource | 21/39/55 | 4/6/8 | 77 |
| rule4ml | 11/70/162 | 2/12/38 | 78 |

Negative deltas favor the explicit region features. A small or mixed delta is evidence against spending scarce GPU time on this path as-is; it is not rescued by seed replication.
