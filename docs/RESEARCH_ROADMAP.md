# HLS surrogate modeling: architecture and research roadmap

## Decision summary

Do not merge `hls4ml_pipeline` and `ll-hls4ml` now. Their dependency profiles
and execution environments are different: graph production requires a specific
Vitis/LLVM/ProGraML toolchain, while training must remain portable across
Colab, Kaggle, and local GPUs. Keep the repositories adjacent, but formalize the
graph handoff and remove the current reverse dependency in which the pipeline
imports pragma injection from the ML package.

The desired flow is:

```text
source archive + labels
  -> compiler/toolchain identity
  -> LLVM + pragma dump
  -> versioned CDFG JSON
  -> frozen dataset manifest
  -> versioned tensor feature schema
  -> immutable split manifest
  -> config-driven training run
  -> metrics + predictions + provenance
```

Notebooks should remain, but as thin research interfaces: inspect a sample,
plot saved results, compare runs, and prototype a feature. Once a notebook idea
is used twice, move the reusable computation into `src/` or a script.

## Immediate stabilization

### 1. Freeze the data contract

Add `schema_version`, `producer`, and `producer_version` to every graph. Document
required node, edge, pragma, label, and provenance fields in a small JSON Schema
or precise Markdown contract. Make pragma injection owned by
`hls4ml_pipeline`; `ll-hls4ml` should only consume pragma nodes.

Create a validation command that checks a manifest without rebuilding data:
file existence, JSON readability, contiguous node IDs, valid edge endpoints,
known flow values, required labels, finite/non-negative targets, and aggregate
counts by kernel/archive. This is the highest-value protection against costly
redownloads and silent data drift.

### 2. Make derived artifacts identifiable

Identify released tensor datasets by their immutable repository commit. Keep
the graph/feature schema and preprocessing code versioned alongside the release;
do not add a second content-hashing pass over every tensor.

Keep a dataset-release manifest containing project ID, kernel family, archive,
graph path, label availability, toolchain identity, and optional checksum.
Archive the manifest and registry separately from the large dataset.

### 3. Make experiments reproducible

Use one config-driven runner for MLP and GNN baselines. Each run directory
should contain:

- resolved config and git commits for both repositories;
- immutable tensor dataset revision;
- exact train/validation/test project IDs;
- seed, device, package versions, target normalization from training only;
- best checkpoint, per-target metrics, and per-sample predictions.

Avoid a general experiment framework. A timestamped directory, JSON config,
JSONL/CSV metrics, and TensorBoard are enough for this thesis.

### 4. Protect evaluation validity

Random project splits are useful only as an optimistic baseline. Projects from
the same architecture/archive may be near-duplicates, so the primary split
should be group-aware. Evaluate at least:

1. random stratified split within all families;
2. archive/group split to reduce near-duplicate leakage;
3. leave-one-kernel-family-out for inductive generalization;
4. optional low-data curves using 10%, 25%, 50%, and 100% of training data.

Persist split membership. Fit target normalization on training data only.
Report per-target R², RMSE, and SMAPE, plus macro summaries and bootstrap
confidence intervals across test projects.

## Experiment sequence

Run the cheapest experiments first and promote complexity only when it beats a
clear baseline.

### Phase A — validate signal and leakage

- Constant median predictor per target.
- Handcrafted graph statistics with linear/ridge regression and gradient
  boosting.
- Pooled node-feature MLP, with and without semantic type features and pragmas.
- Compare random, grouped, and leave-family-out splits.

This phase determines whether gains come from graph topology or from size/family
correlations. It also catches label and split mistakes before GPU-heavy work.

### Phase B — controlled graph ablations

- Message passing versus pooled MLP at matched parameter count.
- Instruction-only; + type semantics; + edge positions; + pragma nodes.
- Mean versus max pooling; use sum pooling only with explicit graph-size
  normalization because it can encode graph size too directly.
- Shared multi-target head versus resource/timing-specific heads.

Change one factor per run. Use a small representative frozen subset to reject
bad ideas, then rerun promising configurations on the full manifest.

### Phase C — robust models

- GraphSAGE/GAT-style heterogeneous message passing with residual connections.
- Log-target Huber loss versus per-target weighted losses.
- Family embeddings or mixture-of-experts only if leave-family-out results show
  strong family-specific behavior.
- Prediction uncertainty using a small deep ensemble or MC dropout for
  out-of-distribution warning.

Avoid broad architecture search until the split and ablation results justify
it. The thesis contribution is stronger with a disciplined evaluation pipeline
than with many weakly controlled models.

## Notebook policy

Keep a small set:

- `00_data_audit`: manifest/schema/label distributions and failed samples;
- `01_eda`: feature and target analysis;
- `02_baseline_comparison`: read saved predictions and compare models/splits;
- `03_error_analysis`: worst cases, residuals by family, graph size, and target;
- `04_thesis_figures`: deterministic publication plots from exported tables.

Preprocessing and training notebooks should become wrappers around scripts or be
retired once the scripts cover their use. Do not duplicate training loops in
Colab/Kaggle notebooks; clone/install the repository, point it at mounted data,
and launch the same config.

## Portable compute

Keep the ML package's core dependencies minimal and put EDA/Jupyter packages in
extras. Default to single-GPU, mixed precision once validated, conservative
loader worker counts, resumable checkpoints, and explicit paths. Treat DDP as
optional; on free single-GPU environments, gradient accumulation is more useful
than distributed complexity.

Sync only graphs/tensors/manifests/configs/results to compute providers. Do not
sync or redownload extracted projects for training. A small smoke tensor set can
live separately for import and end-to-end checks.

## Success criteria for “robust enough”

- A fresh environment can install the ML package and run a smoke experiment
  from one config.
- Every reported number maps to a saved config, split, immutable dataset revision,
  predictions file, and commit.
- Re-running preprocessing cannot silently mix old and new feature schemas.
- A missing cache fails clearly and never triggers an implicit broad download.
- Random, grouped, and leave-family-out results are all available for the best
  baseline and best GNN.
- Thesis plots are regenerated from saved result tables without retraining.
