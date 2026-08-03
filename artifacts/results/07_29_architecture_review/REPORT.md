# LLVM CDFG surrogate architecture review

Status: preliminary diagnostic study. Neural runs use short 20-epoch budgets
unless noted otherwise. They are intended to compare modeling choices, not to
estimate the converged accuracy of the architecture.

## Data and evaluation

- Tensor snapshot: `4b637ba749da56c5`
- Unique non-exemplar graphs: 1,951
- Official/stratified split: 1,383 train, 274 validation, 294 test
- Exemplar set: 400 graphs
- Targets: LUT, FF, DSP, BRAM, maximum cycles, maximum interval
- Selection metric: validation macro SMAPE across the six targets
- Primary evidence: synthetic test split. Exemplar is a separate, compound-OOD
  stress test and is not used for model selection.

Twenty-six repeated Conv2D paths are removed by graph UUID before splitting.
A rounded graph-summary audit found one direct train/test near-duplicate. Removing
it changed the RBF macro SMAPE from 29.99% to 30.06%, so it does not explain the
benchmark result. A stronger source-identity group is still desirable for the
final split audit.

## Current baseline

The revised baseline represents instruction, variable, constant, pragma, and
block nodes separately. It retains typed relations, explicit pragma arguments,
numeric type semantics, scalar constants, and optional graph-level synthesis
context.

Its graph readout combines signed-log sum, mean, maximum, standard deviation,
and node count for every node type. A deterministic shortcut supplies opcode
and pragma histograms, node/edge counts, type statistics, and pragma-argument
statistics. Resource and timing targets use separate prediction towers. DSP and
BRAM optionally use hurdle outputs for zero occurrence and positive magnitude.

The training objective is Huber loss in `log1p(target)` space, with binary
cross-entropy for DSP/BRAM occurrence. This is a reasonable baseline for
multiplicative target error and zero-inflated resources.

## High-signal results so far

### Cheap same-split baselines

| model/ablation | synthetic macro SMAPE (%) |
| --- | ---: |
| RBF + hard DSP/BRAM hurdle | 27.86 |
| RBF without pragma edges | 29.84 |
| RBF without all edge counts | 29.85 |
| RBF graph features | 29.99 |
| RBF graph + synthesis context | about 30.20 |
| RBF without constant literals | 30.27 |
| RBF without pragma arguments | 34.46 |
| RBF without pragmas | 34.80 |
| RBF without numeric type features | 41.36 |
| opcode and graph size only | 50.16 |
| graph size only | 59.59 |
| train median | 102.76 |

The cheap model establishes that the enriched representation contains a strong
same-distribution signal. Numeric type semantics and pragma arguments are the
most clearly supported additions. Constant literal slots help slightly. Coarse
edge counts do not help this tabular model, which does not imply that graph
topology is useless.

The hard DSP/BRAM hurdle improves macro SMAPE by about 2.1 points. Its largest
effect is DSP (35.13% to 24.05%). This supports retaining zero-inflated target
modeling.

### Short neural diagnostics

| model | context | sampling | test macro SMAPE (%) | exemplar macro SMAPE (%) |
| --- | --- | --- | ---: | ---: |
| pooled MLP | backend + part + clock | family-balanced | 45.91 | 116.09 |
| pooled MLP | none | family-balanced | 41.86 | 103.38 |
| pooled MLP | none | empirical/unbalanced | 44.36 | 109.39 |
| mean-pooled MLP | none | family-balanced | 44.72 | 106.03 |
| pooled MLP, no global shortcut | none | family-balanced | 43.70 | 114.46 |
| pooled MLP, no DSP/BRAM hurdle | none | family-balanced | 45.95 | 103.50 |

These neural models are undertrained relative to earlier 60–400 epoch work and
must not be compared as converged models against wa-hls4ml. The meaningful
observation is narrower: in this one short run, the context branch did not
improve the selected checkpoint.

Family-balanced sampling is retained only to hold the remaining diagnostics
consistent. In this seed it improves test macro SMAPE by 2.50 points over
empirical sampling. It also improves the unweighted mean of the seven per-family
macro SMAPEs
(39.42% versus 43.54%), so the result is not only caused by weighting the
aggregate test population differently. The effect is not large enough to select
the sampler without repeated seeds.

### Uncertainty boundary

The neural comparisons currently have one training seed each, noisy
validation-SMAPE checkpoint selection, and short schedules. Differences of
roughly 1–3 SMAPE points should be treated as unresolved; differences around
4–5 points are useful hypotheses but still not reliable architecture rankings.
For scale, the last five validation checkpoints span 6.46 SMAPE points for the
primary multi-pool model, 4.39 for mean pooling, and 5.25 for empirical
sampling; their within-run standard deviations are 2.53, 1.57, and 1.90 points.
Test-set resampling would measure finite-test-set uncertainty but would not
capture the more important training-seed and optimization variance.

The short neural runs are useful for detecting broken models, checking that
ablations move in plausible directions, and prioritizing longer experiments.
They do not provide statistical evidence that one close variant is superior.

The synthesis context stored in every tensor is:

- core: backend, exact target FPGA part, and target clock;
- all: core plus Vivado/Vitis version and hls4ml version.

Here, “no context” means only that this graph-level synthesis-context branch is
disabled. It does not remove graph topology, node/edge types, numeric data types,
constants, blocks, or pragma semantics.

Context should remain in the dataset. It is causally relevant, especially across
devices and tools, but the synthetic training data has almost no crossed
replication of the same graph under different contexts. Backend/version/part
therefore mostly identify kernel family or dataset cohort. Exemplar also contains
tool versions unseen during synthetic training. A context encoder cannot reliably
separate causal context effects from cohort effects in this design.

## Statistical and architectural conclusions

### Loss

The loss is not naive after the revision. Log-space Huber is aligned with
multiplicative errors and is less dominated by the largest designs than raw MSE.
The DSP/BRAM hurdle has the strongest target-specific ablation evidence in this
study. Removing it changes overall test macro SMAPE from 41.86% to 45.95%, DSP
from 35.84% to 50.78%, and BRAM from 39.91% to 46.26%. Family-macro SMAPE changes
from 39.42% to 43.99%. The effect occurs on the zero-inflated targets predicted
by the hypothesis, and the independent cheap SVR hurdle also improves DSP. This
is still a one-seed neural comparison, but is more credible than the smaller
readout deltas.

The paired test-graph bootstrap estimates the no-hurdle minus hurdle macro
difference as +4.09 points (95% interval +2.43 to +5.80), DSP as +14.94
(+9.67 to +20.13), and BRAM as +6.34 (+2.07 to +10.56). These intervals exclude
finite-test-sample chance but not training-seed variance.

The effect is specifically on exact-zero cases. DSP-zero SMAPE is 0.00% with the
hurdle and 43.30% without it, while positive-DSP SMAPE is similar (52.17% versus
54.19%). BRAM-zero SMAPE is 11.86% versus 40.67%, while positive-BRAM SMAPE is
effectively unchanged (47.72% versus 47.81%). The hurdle is therefore addressing
the intended statistical structure rather than improving positive-value
regression generally.

Hard occurrence decisions currently score better under SMAPE than multiplying
by an occurrence probability in the cheap baseline.

For long runs, occurrence thresholds should be selected on validation data rather
than fixed permanently at 0.5. Per-target loss weights should only be introduced
after checking gradient scales; arbitrary weights would make the objective less
interpretable.

### Pooling

Mean pooling alone is an incompatible prior for resource counting: duplicating
identical operations need not change a mean. Mean plus maximum is still
cardinality-blind. Sum/count information is essential.

The current multi-statistic readout avoids that basic bottleneck and is a sound
baseline. It is still a topology bottleneck for timing. Any permutation-invariant
global pool over shallow local embeddings struggles to recover longest paths,
recurrences, loop-carried dependencies, and loop nesting. More pooling statistics
cannot fully solve that limitation.

Holding the deterministic global shortcut fixed, multi-statistic pooling is 2.86
points better in this seed. Its family-macro SMAPE is also lower (39.42% versus
43.28%). This is directionally compatible with a count-aware readout, but is too
small and noisy to establish a practical advantage. The shortcut makes mean
pooling viable in either case.

Removing the deterministic graph-wide shortcut changes test macro SMAPE from
41.86% to 43.70% and family-macro SMAPE from 39.42% to 42.05%. These differences
are within the observed optimization noise. Retraining without the branch shows
that it is not indispensable for learning a comparable predictor, but does not
show whether the original checkpoint used it: the remaining branch can
compensate during retraining. Shortcut utility remains unresolved pending
repeated seeds or a controlled branch-reliance analysis.

A 10,000-resample paired bootstrap over the identical test graphs estimates the
no-shortcut minus shortcut macro-SMAPE difference as +1.84 points, with a 95%
interval of +0.35 to +3.35. This suggests the held-out sample effect is not driven
by a few test graphs, but it remains smaller than plausible training-seed
variance and is not sufficient for architecture selection.

### Message passing

GAT attention normalizes neighbor weights. That is useful for selecting important
dependencies, but it is not naturally additive: ten identical contributing
operations can resemble one identical operation after normalized aggregation.
The graph-wide count shortcut mitigates this for resources.

A one-layer GAT only communicates across one relation. A block pragma can update
its target instruction/block, but information cannot then propagate farther until
a second layer. One layer is therefore a mechanical smoke comparison, not the
intended final GNN.

Timing needs a more explicitly path-sensitive prior in a later pass: DAG
longest-path summaries, SCC/recurrence features, loop depth/tripcount, or a
hierarchical block/loop readout. This is deferred until the current baseline
ablations are complete.

### Heads

Separate resource and timing towers are statistically plausible: cycles and
interval are strongly correlated with one another, LUT and FF are strongly
correlated, while BRAM is comparatively weakly correlated with most targets.
The split/shared-head ablation remains necessary before treating this as settled.
A fully independent head per target is a reasonable later experiment if BRAM
continues to exhibit negative transfer.

### Pragmas and blocks

Pragma argument values provide substantially more cheap-baseline signal than the
mere presence of pragma nodes or attachment-edge counts. Encoding the directive
and its structured numeric/categorical arguments together is therefore
well-founded.

The current pragma-to-target edges preserve attachment information. A one-layer
GNN cannot exploit longer pragma-to-operation-to-block interactions, and the
pooled MLP cannot exploit attachment topology at all. This is a model-capacity
limitation rather than evidence that pragma injection is poor.

Current block features identify stable coarse roles but do not represent loop
hierarchy, dominance, trip counts, or block-level operation summaries. Richer
block/hierarchical work is intentionally deferred.

## Diagnostic status

The context, sampling, pooling, global-shortcut, and hurdle diagnostics are
complete at one short seed. Only the hurdle produces a large, target-localized,
independently replicated effect. The other neural differences remain hypotheses
or unresolved.

The split/shared-head comparison and one-layer GAT accuracy run are intentionally
deferred. At this budget they would likely create another noisy architecture
ranking rather than high-signal evidence. The GAT implementation has passed
correctness and end-to-end smoke tests.

## Recommended long-run basis

Subject to repeated-seed confirmation, a well-founded long-run candidate is:

- enriched typed CDFG tensors with structured pragma arguments;
- multi-statistic, count-aware readout, with the deterministic global shortcut
  retained as an optional repeated-seed ablation rather than an assumed gain;
- no context branch for the primary same-cohort benchmark, while retaining all
  context fields and reporting a separate context ablation;
- DSP/BRAM hurdle outputs;
- resource/timing split towers as a working default, pending a properly powered
  head comparison;
- log-space robust regression;
- family-balanced and empirical sampling as a repeated-seed comparison;
- 2–3 message-passing layers for the real GNN comparison;
- longer patience and a learning-rate schedule for 60+ epoch runs;
- synthetic test as the primary score and exemplar as a separately labeled OOD
  stress test.

The strongest justification for pursuing the method is not the short neural
score. It is that a cheap nonlinear model reaches 27.86% macro SMAPE from the
enriched graph-derived features, far ahead of size/opcode-only controls, and the
feature removals behave in semantically credible directions.
