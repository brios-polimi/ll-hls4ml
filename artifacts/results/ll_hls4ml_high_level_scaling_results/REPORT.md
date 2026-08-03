# Matched high-level layer/config scaling benchmark

## Result

The 54k-parameter high-level GATv2 becomes substantially stronger than the
575k-parameter LLVM CDFG GAT as training data grows. At 200%, its three-seed mean
is **24.14 ± 0.39 test SMAPE** and **80.17 ± 2.44 exemplar SMAPE**, compared with
29.42 and 86.39 for the single available CDFG seed.

| Train scale | High-level test SMAPE | High-level exemplar SMAPE | CDFG test | Hurdle RBF test |
| ---: | ---: | ---: | ---: | ---: |
| 25% | 39.92 ± 1.23 | 96.57 ± 2.23 | 36.60 | 34.81 |
| 50% | 34.10 ± 0.65 | 89.74 ± 2.42 | 34.92 | 30.77 |
| 100% | 29.17 ± 0.90 | 89.86 ± 5.23 | 32.39 | 26.75 |
| 200% | 24.14 ± 0.39 | 80.17 ± 2.44 | 29.42 | 24.67 |

Each scaling step improves test SMAPE by approximately five points. A hierarchical
bootstrap over three training seeds and the 294 fixed test examples gives intervals
of -7.35 to -4.31 for 25→50%, -6.41 to -3.52 for 50→100%, and -6.64 to -3.44 for
100→200%. There is no observed plateau.

At 200%, high-level minus CDFG test SMAPE is -5.28 points (hierarchical interval
-7.63 to -2.90), conditional on the single available CDFG training seed. High-level
minus RBF is -0.53 points (-2.54 to +1.52), so the small apparent high-level advantage is
not statistically resolved even though all three high-level seeds score lower.

## 200% error structure

| Target | High-level SMAPE (mean ± seed SD) |
| --- | ---: |
| LUT | 18.86 ± 0.24 |
| FF | 24.10 ± 0.58 |
| DSP | 21.25 ± 0.24 |
| BRAM | 21.49 ± 1.46 |
| Cycles | 25.52 ± 0.80 |
| II | 33.62 ± 0.39 |

| Test family | High-level | CDFG | Hurdle RBF |
| --- | ---: | ---: | ---: |
| 2layer | 11.06 ± 0.62 | 22.0 | 12.6 |
| 3layer | 11.40 ± 0.19 | 22.1 | 13.0 |
| conv1d | 22.25 ± 1.80 | 27.1 | 22.0 |
| conv2d | 32.69 ± 2.92 | 31.0 | 35.5 |
| dense_latency | 28.99 ± 3.65 | 16.0 | 16.3 |
| dense_resource | 15.20 ± 0.80 | 16.9 | 12.4 |
| rule4ml | 51.44 ± 2.26 | 62.1 | 61.0 |

The high-level representation is especially effective for regular 2/3-layer
networks and Rule4ML. It does not dominate everywhere: CDFG and RBF are dramatically
better on `dense_latency`, and CDFG remains slightly better on conv2d. This is strong
evidence for complementary representations rather than replacing LLVM CDFGs.

## Interpretation

The principal limitation of the current CDFG model is not simply training-set size
or attention capacity. A much smaller model learns faster from explicit layer
dimensions, layer types, precision, reuse factor, strategy, and I/O configuration.
Those high-level variables are difficult to reconstruct through local LLVM message
passing and global pooling.

The next justified model is a controlled hybrid: encode the high-level layer graph
and LLVM CDFG separately, concatenate their graph embeddings and synthesis context,
then reuse the existing split hurdle heads. The `dense_latency` and conv2d results
provide a concrete falsifiable reason for retaining the CDFG branch. A large
hierarchical CDFG-only redesign is lower priority than this late-fusion test.

The high-level versus CDFG uncertainty interval does not include CDFG training-seed
variance. Two additional 200% CDFG seeds would be needed for a fully symmetric model
comparison, but they are not required to establish the high-level model's scaling
curve or its representation value.

## Reproducibility

- Pipeline departures: `docs/HIGH_LEVEL_BASELINE.md`
- Runner: `scripts/train_high_level.py`
- Adapter: `src/ll_hls4ml/data/high_level.py`
- Model: `src/ll_hls4ml/models/high_level.py`
- Raw runs: `high_level_gatv2_scale{025,050,100,200}_seed{42,43,44}`
- Aggregate outputs: `aggregate_summary.csv`, `target_summary.csv`,
  `family_summary.csv`, and `uncertainty_summary.csv`
- All 3,737 required label records were found. The explicit BiPC fallback affected
  57 of 400 exemplar records and no synthetic record.
