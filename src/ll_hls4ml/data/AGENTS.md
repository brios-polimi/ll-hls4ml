# Data and tensorization

This subtree owns the graph-JSON to tensor contract, vocabularies, splits,
high-level hls4ml modalities, and learning-time augmentations. Keep compiler/LLVM
rewrites in `hls-ir-graph`; transforms here must operate on loaded graph/tensor
representations and must never silently alter labels or split membership.

Augmentations belong under `augmentations/`, need stable configuration names,
and should be deterministic when given an explicit seed. Test schema changes on
small synthetic `HeteroData` objects rather than the stateful tensor database.
