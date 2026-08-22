# Region representation audit

Read-only audit of tensors/graphs selected from the schema-v3 cohort named in the split manifest.

Selection: first 10 records per split/family.

## Core coverage

| split | samples | loops | loops/sample | known trip counts | source-anchored loops | samples with loop pragma | loops with loop pragma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 70 | 3540 | 50.57 | 0 (0.0%) | 1986 (56.1%) | 53 (75.7%) | 1428 (40.3%) |
| validation | 70 | 3345 | 47.79 | 0 (0.0%) | 2009 (60.1%) | 57 (81.4%) | 1514 (45.3%) |
| test | 70 | 3190 | 45.57 | 0 (0.0%) | 1808 (56.7%) | 54 (77.1%) | 1271 (39.8%) |
| exemplar | 10 | 304 | 30.40 | 0 (0.0%) | 126 (41.4%) | 7 (70.0%) | 112 (36.8%) |

## Interpretation

`trip_count_known` is the tensor feature consumed by the region model. A zero count means the current experiment did not test trip-count-conditioned composition.
The schema has no memory node type; `memory_like_variables` is only the existing reporting proxy, not a memory representation.

## Symbolic pragma values

| split | resolved numeric arguments | injected unresolved numeric arguments | unmatched symbolic numeric arguments | graphs with unmatched symbolic values |
| --- | ---: | ---: | ---: | ---: |
| train | 923 | 0 | 1024 | 70 |
| validation | 831 | 0 | 993 | 70 |
| test | 820 | 0 | 1022 | 70 |
| exemplar | 56 | 0 | 108 | 10 |

Injected unresolved numeric arguments would make tensorization fail; unmatched values are dump records whose source function was not represented in the final LLVM graph.

## Loop-scoped directives (all splits)

| directive | loop-scoped pragma nodes | distinct stored argument vectors |
| --- | ---: | ---: |
| unroll | 2943 | 1 |
| pipeline | 1333 | 107 |
| loop_flatten | 67 | 1 |

## Family detail (all splits)

| family | samples | loops | known trip counts | loops with loop pragma |
| --- | ---: | ---: | ---: | ---: |
| 2layer | 30 | 300 | 0 | 240 |
| 3layer | 30 | 450 | 0 | 360 |
| conv1d | 30 | 2287 | 0 | 1232 |
| conv2d | 30 | 2454 | 0 | 1193 |
| dense_latency | 30 | 1382 | 0 | 0 |
| dense_resource | 30 | 1134 | 0 | 684 |
| exemplar | 10 | 304 | 0 | 112 |
| rule4ml | 30 | 2068 | 0 | 504 |
