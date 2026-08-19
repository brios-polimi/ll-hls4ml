# Region-hierarchy implementation journal

Last updated: 2026-08-19

## Objective

Build a standalone, general HLS surrogate over hierarchical LLVM CDFGs. Fusion
is outside the core implementation and remains only a possible final addition.
The working hypothesis is that H0 is missing semantic hierarchy and
hardware-aligned composition, not another generic attention/message operator.

## Repository state

- `ll-hls4ml` branch: `codex/region-hierarchy`
- `hls4ml_pipeline` branch: `codex/region-hierarchy`
- Python environment: always use
  `/home/brend/anaconda3/bin/conda run -n pipeline-env --no-capture-output ...`
- Tests: use `python -m unittest`; do not require pytest.
- Existing H0 control: 24.55 test macro SMAPE on 593 deduplicated synthetic
  projects, 239,568 parameters.
- Material-gain rule: approximately three validation points to promote a
  candidate; at least five test points for the intended architectural result.

The detailed evidence and design rationale are in
`docs/THESIS_WALL_REVIEW_2026-08-19.md`. Quantitative reproduction scripts are
`scripts/thesis_wall_audit.py`, `scripts/high_level_tabular_probe.py`, and
`scripts/ir_canonicalization_probe.py`. Region-specific diagnostics are
`scripts/region_feature_probe.py` and `scripts/region_model_cpu_benchmark.py`.

## Implementation strategy and ablations

The work is staged at durable handoff boundaries. Each stage must remain
independently switchable so unchanged H0 is always available.

| Stage | Mechanism | Ablation/control | Status |
| --- | --- | --- | --- |
| 0 | Reproducible review and CPU probes | Existing committed result blobs | complete |
| 1 | Conservative LLVM canonicalization | Pipeline config disabled/empty | implemented, sample-verified |
| 2 | Explicit natural-loop nodes and nesting | `enrich_natural_loops = false` | implemented, sample-verified |
| 3 | Loop tensor schema and region-HLS model | Model name `hierarchical` remains H0 | implemented, CPU-verified |
| 4 | Resource-additive/timing-path composition | `composition: generic` | implemented, CPU-verified |
| 5 | CPU information screen and GPU config | Existing H0 config | complete; dataset rebuilt and verified |
| 6 | Data expansion and structural-OOD evaluation | Deferred by scope | deferred |

## Stage 1: conservative LLVM canonicalization

### Why

The current compiler command does not run an explicit LLVM optimization
pipeline. A representative three-layer IR contained 4,159 instruction lines
and 568 allocas. The CPU probe found:

- `mem2reg`: 1,594 instructions;
- `sroa,mem2reg`: 1,132;
- `sroa,mem2reg,instcombine`: 710 on the initial three-layer probe;
- CFG simplification and O1 destroyed important anchors/structure.

### Implemented

- Added opt-in `llvm_canonicalization_passes`, `llvm_opt_binary`, and
  `llvm_analysis_triple` fields; the default pass string is empty.
- The compiler runs only the explicitly supplied pass pipeline.
- It uses a temporary analysis triple because stock LLVM 16 does not recognize the
  `fpga64` architecture, then restores the original target triple in durable IR.
- It validates the exact function/block CFG adjacency, ignoring only harmless
  true/false successor ordering, plus Vitis intrinsic and pragma-carrier counts.
  A failed invariant refuses the rewrite.
- Eight compiler-helper `unittest` cases pass.
- An initial integration check on
  `3layer/archive_6/b0051e09-ffb4-48e1-8991-9af53f5ee0e8.ll` produced a 80,063-byte
  canonical IR, restored `target triple = "fpga64-xilinx-none"`, and passed all
  sampled invariants with `sroa,mem2reg,instcombine`. The later labeled conv1d
  audit below found that `instcombine` is not safe corpus-wide, so the production
  dataset uses only `sroa,mem2reg`.

### Restart point

Do not enable it for the existing dataset yet. The remaining Stage 1 work is a
stratified 5–10-project-per-family probe and small graph-regeneration comparison;
that bulk data operation is deliberately deferred until the graph/model schema
is ready.

## Stage 2: explicit natural-loop graph hierarchy

### Implemented

- Detects natural loops per function from the already parsed LLVM block CFG:
  compute dominators, identify backedges, form natural-loop block sets, merge
  latches with a common header, and derive the nesting tree by set inclusion.
- Injects loop nodes into durable graph JSON with header/latch/exit counts,
  nesting depth, block count, and masked trip-count fields.
- Adds loop→block direct containment, loop→loop child containment, and
  function→loop top-level containment.
- Preserves current block/function nodes and relations for exact H0 compatibility.
- Adds pragma→loop edges when a source-loop anchor maps unambiguously to the
  natural loop headed by its CFG successor. Preserve the old pragma→block edge
  as provenance/backward compatibility.
- Versions the attachment contract as schema 5. Disabling loop enrichment
  retains the previous graph topology.

The representative three-layer sample produced 15 loop nodes, 207 direct
loop→block edges, three nested-loop edges, and 12 function→top-loop edges. All
12 source-loop labels resolved to a unique natural loop. The mappings include
`InitAccum→for.cond`, `ReuseLoop→for.cond16`, `MultLoop→for.cond22`, and
`Result→for.cond48` in each concrete function specialization.

Twenty targeted `unittest` cases pass across compiler canonicalization, pragma
attachment, loop analysis, canonical relations, and finalized constant
ownership/pruning. The pruning compatibility fix retains a constant when any
surviving semantic edge—such as `pragma→constant`—still references it.

### Ablation

Set `enrich_natural_loops = false` (the default) for the prior topology. Old
graph/tensor schema and H0 remain readable. Region models will require the new
schema rather than silently synthesizing fake loops.

## Stage 3: tensor schema and model

### Implemented

- Tensorizes loop nodes, loop containment, top-level ownership, and pragma scope
  as hierarchy schema 3. Schema-2 tensors remain readable.
- Adds the separate model name `hierarchical_region`; `hierarchical` and its six
  node-type contract are unchanged.
- Retains H0 instruction→block processing and leaf-to-root call schedule.
- Processes loop nesting inside-out with shared weights across depths; combines
  direct blocks and completed child-loop states.
- Adds optional bidirectional typed, cardinality-aware local messages using
  signed-sum/mean/max/std/log-degree statistics. `cardinality_messages: false`
  retains the H0 mean-message layers and cleanly isolates the loop hierarchy.
- Validates exactly one direct parent per loop, no duplicate direct loop owner
  per block, same-function nesting, and consistent nesting depths.

### Verification and remaining work

- Synthetic nested-loop tensorization, ownership rejection, two-graph batching,
  H0 schema-2 forward, and region forward/backward pass for both message modes.
- A real validation graph tensorized in memory with 3,551 instructions, 253
  blocks, and 15 loops. H0 still has exactly 239,568 parameters.
- A top-down context pass is intentionally not implemented yet. It is not
  justified before bottom-up region composition shows a material validation
  gain.

## Stage 4: hardware-aligned composition

### Implemented

- `composition: generic` retains the standard sum/mean/max/std/count pool.
- `composition: hardware_aligned` uses separate nonnegative additive resource
  contributions, smooth learned critical-element pooling, sharing mean/std, and
  cardinality at every hierarchy boundary.
- Damped max-plus relaxation over directed instruction control/def-use and block
  CFG edges gives local delay, arrival, and recurrence summaries without an
  unbounded cyclic-CFG recurrence.
- Loop structural features and exact loop-scoped pragma features gate the
  composed child/block state, so trip-count/II fields can condition a region
  transformation instead of only acting as averaged neighbor messages.

### Ablation

Use `composition: generic` versus `hardware_aligned`. `critical_path_steps`
controls the bounded directed relaxation. No broad operator matrix was added.

## Stage 5: CPU screens and experiment decision

### Aggregate information probe

`scripts/region_feature_probe.py` compared an identical high-level+CFG control
with and without natural-loop summary features. This diagnostic is not a fusion
proposal: it asks whether loop counts/sizes alone contain marginal information.
It used 280 train and 274 official-validation projects (up to 40 per family and
split), archive-grouped training CV, fixed 0.5 hurdle thresholds, and never read
test or exemplar.

- Control validation macro SMAPE: 34.51.
- Control+region-summary validation macro SMAPE: 34.29 (delta -0.22).
- Paired 95% bootstrap CI: [-0.71, +0.25].
- Timing was not improved: cycles +0.04, interval +0.86.
- Two/three-layer loop summaries were exactly invariant in the sample; other
  families had 73–80 distinct summaries but still no material marginal gain.

Conclusion: loop counts are not a plausible 5–10 point answer. The GNN is worth
one staged test only because it preserves operation placement, nesting, exact
pragma scope, and region composition—information deliberately destroyed by the
tabular summary. Do not reinterpret this null result as support for adding loop
statistics to a global feature vector.

### Feasibility benchmark

One-thread CPU medians on the real three-layer validation graph:

| Ablation | Parameters | Median forward |
| --- | ---: | ---: |
| H0 | 239,568 | 13.6 ms |
| region, mean messages, generic composition | 301,968 | 15.6 ms |
| region, mean messages, hardware composition | 323,476 | 22.1 ms |
| region, cardinality messages, generic composition | 634,896 | 75.0 ms |
| region, cardinality messages, hardware composition | 656,404 | 79.2 ms |

This makes cardinality messaging a later ablation, not the first GPU run.

### GPU promotion order

The user explicitly chose an in-place combined-representation test after
finalizing the old tensorization changes. The high-signal design is now:

1. Retrain unchanged H0 on the canonicalized schema-3 tensors. H0 accepts schema
   2 or 3 and ignores the loop stores, so this is the matched representation
   control. Its paired delta from the historical H0 estimates the
   canonicalization contribution.
2. Train the mean-message region model with hardware-aligned composition on the
   identical tensors and cohort. This is the primary architecture bundle.
3. Only if the bundle materially improves H0, train generic composition as the
   ablation that retains explicit loops but removes the hardware-aligned
   composition prior.
4. Cardinality messaging remains deferred because it roughly doubles parameters
   and was much slower in the CPU feasibility benchmark.

This prioritizes finding a genuinely large result within the 21-hour GPU budget,
while the matched H0 and conditional generic run preserve causal interpretability.

### Restart point (full rebuild authorized 2026-08-19)

The user authorized replacing `data/ll`, `data/graphs`, and `data/tensors`, while
strictly preserving `data/source` and forbidding web downloads. Preflight found
65 registered archives, 6,487 retained project source trees, and no archive
missing retained source. Before deletion, a deterministic tar-stream hash of the
entire 9.8 GiB source tree was recorded:

`ee6387acae725e93ad19e758404b858a06062ea9db181a35defd83d824db3a5c`

After graph regeneration and tensorization, the same deterministic full-tree
hash was recomputed and matched exactly. `data/source` was byte-for-byte
preserved.

The production config now has `allow_downloads: false`,
`delete_source_after_compile: false`, canonicalization
`sroa,mem2reg`, and natural-loop enrichment enabled. The processor
prefers retained sources even if disposable extracted archives remain and fails
closed before the downloader if a retained archive is absent.

The completed rebuild command, run from `/home/brend/projects`, was:

```bash
/home/brend/anaconda3/bin/conda run -n pipeline-env --no-capture-output \
  python -m hls4ml_pipeline --config hls4ml_pipeline/config.json fetch \
  --type all --archives all --labels data/labels --force-recompile
```

It initially exposed a stale broken `bazel-bin` symlink. A later broad pass also
showed that `instcombine` removed Vitis FIFO pops in labeled conv1d IR. A retained
conv1d audit measured 34,572 raw instructions; `sroa,mem2reg` reduced this to
10,957 (-68.3%) while preserving all 2,110 blocks, 69 functions, 302 Vitis
intrinsics, and 82 pragma carriers. Adding `instcombine` reduced to 7,486 but
removed 25 Vitis operations, so it was rejected. The final strict pipeline is
therefore `sroa,mem2reg`. Local label JSONs are passed to skip unlabelled retained
projects that are outside the training cohort; this is not a download. Partial
outputs were removed before the final restart. The final run considered 6,487
retained projects and produced 4,816 labeled LLVM/graph pairs, skipped 1,671
projects without local synthesis labels, and had zero failures.

After graph completion, tensorize all labeled graphs with:

```bash
/home/brend/anaconda3/bin/conda run -n pipeline-env --no-capture-output \
  python scripts/build_tensors.py --config configs/default.yaml --workers 8
```

The first tensorization attempt exposed a stale positional vocabulary bound of
4: a valid canonical conv1d operand used position 5. A complete vocabulary scan
over all 4,816 graphs established the actual maximum as 6 and regenerated the
instruction vocabulary. The successful rerun produced 4,816 tensors plus
`labels.json` and automatically bundled `vocab.json`. Tensorization still reports
`ptr` and `struct` as unparsed type categories; this pre-existing limitation is
uniform across the cohort and is recorded for later representation work rather
than changed inside this experiment.

Filesystem and cohort validation:

- 4,816 `.ll`, 4,816 graph JSON, and 4,816 `.pt` files;
- 26 duplicate main-cohort UUID paths, yielding exactly 3,904 unique main
  samples and 886 unique exemplar samples;
- exact saved membership: 2,742 train, 569 validation, 593 test, and 886
  exemplar, with no missing or unexpected unique tensor paths;
- one real tensor per archive (65 total) passed schema-3, `sroa,mem2reg`, and
  feature-width checks; every sample had natural loops and 51 had a nonempty
  pragma-to-loop relation;
- both matched H0 and hardware-region models produced finite outputs on a real
  schema-3 tensor using the rebuilt vocabulary (`max_pos = 6`, 37 instruction
  tokens);
- output sizes: 2.3 GiB LLVM IR, 13 GiB graphs, and 5.6 GiB tensors.

The final tensor root is ready to upload as the pinned Kaggle dataset snapshot.
Do not rebuild it in `/kaggle/working`.

The Kaggle driver is
`notebooks/04_06_kaggle_region_hierarchy_scale200.ipynb`. Tensor snapshots and
the Hugging Face/Xet cache live entirely under `/tmp`; `/kaggle/working` holds
only code, small configs, checkpoints, reports, and the resumable result zip.
Paste immutable code and tensor revisions before running.

## Validation protocol

- CPU/unit verification throughout; no large regeneration or training until a
  deliberately narrow sample passes semantic checks.
- Use validation, not the repeatedly inspected 593-project test, for design
  decisions.
- One seed gates GPU candidates. Additional seeds only confirm a material gain.
- Data expansion and structural robustness are explicitly deferred until the
  representation/model path is viable.

## Change log

### 2026-08-19

- Created implementation branches in both active repositories.
- Established staged/ablatable implementation plan and restart document.
- Implemented optional conservative LLVM canonicalization with structural/HLS
  invariants in `hls4ml_pipeline`.
- Verified the candidate pass pipeline on one retained three-layer LLVM file.
- Implemented optional natural-loop extraction, nesting, direct block
  ownership, and exact pragma→loop attachment in `hls4ml_pipeline`.
- Verified Stage 2 on one retained graph/IR pair and with targeted CPU tests.
- Implemented schema-3 tensorization and the separately registered
  `hierarchical_region` model with generic/hardware composition and optional
  cardinality messages.
- Added isolated graph/tensor paths and two first-run experiment configs.
- Ran the 554-project CPU information probe and real-graph model accounting;
  results are in `docs/REGION_FEATURE_PROBE_2026-08-19.md` and
  `docs/REGION_MODEL_CPU_BENCHMARK_2026-08-19.md`.
- Final verification: all 41 `hls4ml_pipeline` discovery tests pass, and 30
  focused ll-hls4ml dataset/hierarchy/tensorization/split tests pass.
- No GPU training was started.
- Completed the user-authorized offline in-place rebuild of `ll/`, `graphs/`,
  and `tensors/` with canonicalization plus loop hierarchy; `source/` remained
  guarded and fingerprinted.
- Added canonicalization provenance to graph JSON and PyG tensors, made H0 a
  valid matched control on schema 3, and made training tensorization skip
  unlabelled graphs.
- Added the full-scale Kaggle notebook with two 330-minute primary runs,
  conditional generic-composition ablation, exact cohort assertions, resume
  packaging, cache-only tensor downloads, and paired bootstrap intervals.
- Rebuilt the corpus vocabulary (`max_pos = 6`) and made tensorization bundle it
  with the dataset automatically.
