# Seed-42 architectural gauge

All three probes use the same manifest hash (`e3232c6e25a73d2d8cb6a00d440ffc644b06e620dd28044b7a9e27bdfa511fdb`), 1,383 training graphs, 66 validation graphs, 81 archive-1 test graphs, and 100 archive-1 exemplars. These are compute-allocation probes, not final estimates: there is one seed, only 57 Conv2D training graphs, one Conv2D validation graph, and four Conv2D test graphs.

| architecture | best validation SMAPE | epoch / training wall | test SMAPE | test R2 | resource / timing SMAPE | parameters | peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Block attention | 37.46% | 24 / 2,094 s | **42.48%** | 0.347 | **41.99%** / 43.48% | 289,744 | **1,611 MiB** |
| Memory-first dual path | 38.76% | 24 / 2,135 s | 43.23% | 0.534 | 43.44% / **42.81%** | 334,994 | 2,735 MiB |
| Block-sequence GRU | 38.76% | 22 / 2,016 s | 44.16% | **0.626** | 44.83% / 42.83% | **289,233** | 2,880 MiB |

At the common epoch 22, validation SMAPE was 38.76% for sequence GRU, 39.65% for memory-first, and 40.08% for block attention. Block attention then improved sharply at epoch 24. The complete epoch and wall-time curves are in `architecture_gauge_comparison.csv` and `architecture_gauge_learning_curves.png`.

Block attention is the best next compute investment under the SMAPE objective. It has the best final test SMAPE, the best resource SMAPE, the lowest peak memory, and the shortest full-probe training time. Memory-first is worth retaining as the timing/resource-specialization candidate: it has better macro R2 and timing SMAPE, and its GPU trace is substantially steadier (52.9% mean utilization, 4.3% zero samples) than block attention (37.9% mean, 11.3% zero samples). Disk reads were negligible for both, so remaining idle gaps are dominated by CPU dispatch and many small irregular graph operations rather than tensor loading.

None of the three is competitive with the fully trained H0 yet, and the comparison to H0 is not compute-matched: H0 used all eight training archives and hundreds of epochs, while these probes used four archives and 22-25 epochs. Exemplar SMAPE remains 116-121%, so no OOD claim is supported. The rational next step is three seeds of block attention before a longer run; use memory-first as the second choice if timing accuracy or utilization efficiency is prioritized.
