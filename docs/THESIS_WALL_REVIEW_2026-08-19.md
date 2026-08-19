# Thesis wall review: a general hierarchical LLVM CDFG model

## Bottom line

The thesis should not center fusion. Fusion is useful evidence and may be a
final accuracy addition, but the core research problem is a **general and robust
GNN for hierarchical LLVM IR CDFGs**.

The current experiments do not show that this direction is exhausted. They show
that swapping message-passing machinery over the current representation is
exhausted. H0 already has a good instruction → block → function/call hierarchy
and count-aware level readouts. Sequence GRUs, block attention, a memory branch,
and block structural features did not add the missing 5–10 SMAPE points. The
reason is more fundamental: the graph and computation still omit the quantities
that connect LLVM execution to HLS hardware—loop regions and nesting, trip counts
and invocation multiplicity, recurrence constraints, memory banking/port
pressure, and directive-conditioned resource sharing and scheduling.

There is also a newly identified input-representation problem. LLVM emission has
no explicit optimization pipeline and retains a large amount of front-end
scaffolding. On one stratified three-layer example, a conservative
`sroa,mem2reg,instcombine` pass reduced the retained IR from 4,159 to 710
instructions (82.9%) while preserving all functions, basic blocks, known loop
anchor names, Vitis intrinsics, and pragma carriers. H0 and its variants have
been spending most of their capacity and GPU time on a much noisier graph than
necessary.

My primary recommendation is therefore a **pragma-conditioned loop/region GNN
with target-aligned resource and timing aggregation**, trained on conservatively
canonicalized LLVM. It should preserve the successful H0 call hierarchy, add an
explicit natural-loop hierarchy, and replace generic mean aggregation with
operations that match hardware composition. Fusion comes only after this model
works standalone.

## What the completed experiments actually establish

All headline architecture comparisons below use the same 593 synthetic test
projects. Tensor revision strings are provenance, not presumed confounders; most
revisions were minor. The program/BiGRU run is the material schema exception
because it added many block-structural features.

| model | test SMAPE | delta from H0 | fixed H0 blend |
| --- | ---: | ---: | ---: |
| H0 hierarchy | 24.55 | -- | -- |
| Sequence GRU | 24.24 | -0.31 | 22.08 |
| Memory dual | 25.57 | +1.02 | 23.88 |
| Block attention | 28.15 | +3.60 | 25.03 |
| Program BiGRU + structural attention | 28.10 | +3.55 | 25.30 |
| High-level GATv2 | 25.55 | +1.00 | 22.14 |
| Learned late fusion | 16.10 | -8.45 | n/a |

The sequence result is a null result for the thesis goal. A 0.31-point gain does
not become a plausible 5–10-point result through seed uncertainty. The program
model is worse for every family and its H0 ensemble also worsens. Stop that
implementation direction.

The late-fusion result has one legitimate role in the GNN thesis: it proves that
roughly eight points of predictable signal are absent from or inaccessible to
the current CDFG branch. It does not imply that fusion should be the contribution.
The scientific task is to make a general LLVM representation recover the same
kinds of workload, parallelism, and scheduling information without relying on
hls4ml layer metadata.

H0's per-target SMAPEs are 24.51 LUT, 23.17 FF, 19.24 DSP, 29.09 BRAM, 27.58
cycles, and 23.72 II. This is not a single broken output head. Resource scale,
memory organization, and scheduling are all insufficiently represented.

## Where H0 is strong

The successful prior should be retained:

- leaf-to-root call processing, with completed callee states injected at call
  sites;
- instruction → basic block → function composition rather than one flat graph;
- typed control and def-use relations;
- numeric type, constant, and structured pragma features;
- signed-log sum, mean, max, dispersion, and count pooling at hierarchy
  boundaries;
- entry-function and all-reachable-function readouts.

These choices explain why H0 is hard to beat with a generic GNN layer. H0 is
already count-aware at hierarchy boundaries and has a useful program prior.
The next model should extend that prior, not replace it with an attention block.

## The real bottlenecks

### 1. The hierarchy stops before the most important HLS level

The schema has instruction, block, and function nodes. A block's only learned
features in H0 are entry, named, and source-loop-anchor flags. A source-loop
anchor is still an ordinary basic block. It is not a natural-loop object and it
does not contain the loop body, nested subloops, latches, exits, backedges, or a
trip count.

The program/BiGRU revision added reverse-postorder, dominance,
post-dominance, loop depth, header/latch, backedge, exit, and immediate-dominator
features to blocks. That is useful graph annotation, but it is not loop
composition. The model still cannot summarize an inner loop, pass that state to
its parent loop, and apply different scheduling/resource rules at each level.
Its 28.10 result therefore rejects “more block features plus BiGRU/attention,”
not an explicit loop hierarchy.

This distinction is supported by the primary literature. The 2024
[hierarchical source-to-post-route model](https://arxiv.org/abs/2401.08696)
extracts inner loops, predicts them locally, collapses them to supernodes, and
then predicts outer hierarchies. It also models unrolling as replication and
array partitioning as memory-port changes. Its data and engineered profiling
features make its scores non-comparable to this project, but its representation
choice is directly relevant. HARP likewise used a hierarchical GNN and a much
larger 40k-program corpus
([paper](https://arxiv.org/abs/2201.06848)).

### 2. H0 uses mean aggregation where multiplicity is causal

Every local H0 message aggregation uses `reduce="mean"`: operand-to-instruction,
pragma-to-target, control/def-use instruction propagation, and block CFG
propagation. The later hierarchy pools restore global counts, but the local
state cannot distinguish one identical predecessor from twenty identical
predecessors. That loses fan-in, fan-out, reuse, contention, and replica count
at exactly the locations where they matter.

This is not a reason to try PNA as another isolated architecture. A
cardinality-aware typed aggregator is necessary but probably not sufficient for
five points. It should be part of the loop/region design: forward and reverse
typed channels with sum, mean, max, standard deviation, and log degree, followed
by a small learned mixer.

### 3. Pragmas are represented as messages, not transformations

The pragma feature schema is thoughtful: directive identity, structured numeric
arguments, categorical modes, and structural target edges are retained. Earlier
CPU ablations strongly support this work: removing pragma arguments changed the
RBF baseline from 27.86 to 34.46 SMAPE, removing pragmas reached 34.80, and
removing numeric type features reached 41.36.

The model then mean-aggregates pragma embeddings into the affected instruction,
variable, block, or function. That treats a directive like descriptive context.
In HLS, it is an operator:

- unroll changes active replica count and remaining serial iterations;
- pipeline changes overlap and makes recurrence/resource-constrained II central;
- array partition/reshape changes banks, ports, and access contention;
- allocation/bind directives change sharing and implementation choice;
- inline and dataflow change function/region composition.

These effects should condition the computation at the affected semantic scope.
Use directive embeddings and arguments to FiLM/gate region states and to modify
explicit multiplicity, sharing, and port features. Do not merely diffuse the
directive through several generic graph layers.

### 4. Resource and timing require different graph algebras

Splitting the final heads is not enough. Resource usage is predominantly an
additive/counting problem with sharing and binding corrections. Timing is a
directed path, recurrence, and overlap problem. H0 gives both targets essentially
the same graph computation.

The failed memory-dual model does not settle this question. It used separate
resource/timing pools and a memory exchange, but still lacked explicit loop
objects, trip counts, recurrence distances, bank/port state, and
pragma-conditioned composition. It also changed H0's successful final readout.

A better inductive bias is:

- resource state: learned nonnegative per-operation/type cost, sum aggregation,
  then learned sharing/allocation corrections at loop and function boundaries;
- timing state: learned operation delay plus smooth max/longest-path propagation
  over directed def-use/control edges, recurrence summaries inside loops, and
  loop-level composition using trip count and II;
- shared semantic encoder underneath, but distinct computations from the first
  region boundary onward.

The model need not be a rigid analytical estimator. These are differentiable
priors that constrain how information composes.

### 5. The raw IR contains avoidable compiler scaffolding

The Vitis compiler command emits LLVM without an explicit optimization pass.
The sampled three-layer program contains 568 `alloca` instructions and produces
a 3,551-instruction ProGraML tensor. The new
`scripts/ir_canonicalization_probe.py` produced this LLVM-level comparison:

| pipeline | instructions | reduction | blocks | functions | loop anchors | Vitis intrinsics | pragma carriers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 4,159 | -- | 253 | 10 | 4/4 | 29 | 11 |
| mem2reg | 1,594 | 61.7% | 253 | 10 | 4/4 | 29 | 11 |
| sroa + mem2reg | 1,132 | 72.8% | 253 | 10 | 4/4 | 29 | 11 |
| + instcombine | 710 | 82.9% | 253 | 10 | 4/4 | 29 | 11 |
| + CFG/loop simplification | 512 | 87.7% | 61 | 10 | 1/4 | 29 | 11 |
| O1 | 325 | 92.2% | 37 | 2 | 0/4 | 0 | 11 |

This is one program, so it is a feasibility finding rather than an accuracy
result. It is nevertheless too large to ignore. `sroa,mem2reg,instcombine` is
the candidate to audit on a stratified CPU sample. Full CFG simplification and
O1 are currently too destructive. Production use must restore/retain the FPGA
triple, validate node/pragma mappings, and compare semantic invariants before
regenerating any dataset.

Canonicalization has three plausible benefits at once: less nuisance variation,
much cheaper GPU batches, and shorter graph distances between hardware-relevant
operations. It may also make explicit loop and recurrence analysis more stable.

### 6. Execution multiplicity is still implicit

Specialized LLVM often contains concrete loop bounds and array sizes, and
constant literals are encoded. But a shallow GNN must infer trip counts,
invocation counts, recurrence distances, and effective parallel work from
scattered constants and control structure. That is an unnecessarily hard use of
2,742 training graphs.

Use standard compiler analyses where possible:

- `LoopInfo`/dominators for natural loops and nesting;
- `ScalarEvolution` for exact, bounded, or unknown trip-count features;
- backedge and loop-carried def-use analysis for recurrence summaries;
- memory-object identity plus load/store access counts and, where recoverable,
  affine access information;
- call graph and existing function containment.

Unknown values need explicit masks. A general model should not depend on every
trip count being statically solvable.

## Recommended standalone architecture

Working name: **pragma-conditioned region hierarchy with dual resource/timing
composition**.

The structural path is:

`typed instructions/memory → basic blocks → natural loops/regions → functions → call graph/module`

The design should be compact—roughly H0's parameter scale, not a large
transformer.

### Representation

Add explicit loop nodes and relations:

- block/instruction belongs to innermost loop;
- loop contains direct blocks and child loops;
- loop parent/child nesting;
- header, latch, exit, and recurrence relations;
- loop features: depth, known/unknown trip count, estimated min/max/average,
  number of exits, backedges, and pipeline/unroll state;
- memory-object features: type/shape if recoverable, access counts, reads/writes,
  candidate ports/banks, and partition/bind directives.

Keep basic blocks, functions, and calls. Avoid representing each region only by
a feature copied onto all member blocks; the region must have its own state and
postorder schedule.

### Computation

1. Run two lightweight bidirectional typed message layers over instructions and
   memory objects. Aggregate sum/mean/max/std/log-degree per relation.
2. Pool instructions to blocks with H0's count-aware pool.
3. Process the loop nesting tree inside-out. A loop state combines its direct
   blocks, child-loop states, recurrence summary, trip-count features, and local
   pragmas.
4. Send a small top-down context pass from function/outer loop to inner loops so
   enclosing dataflow, pipeline, inline, and allocation policies can condition
   children.
5. Compose top-level regions into functions and retain H0's leaf-to-root call
   schedule and entry/reachable-function readout.
6. Maintain a resource channel with additive pooling and a timing channel with
   smooth-max/path pooling. Use separate target heads only after those distinct
   computations.

Weight sharing across loop depths is important for generality. It also controls
parameter count and lets the architecture handle deeper unseen nests.

### Pragma conditioning

Map directives to semantic scopes and expose explicit derived quantities where
safe. For example, with trip count `TC` and unroll factor `UF`, give the loop
both `log(TC)`, `log(UF)`, and `log(ceil(TC/UF))` plus masks. Let the network
learn corrections rather than relearn integer arithmetic. For array partition,
expose factor/type/dimension and estimated bank count. For pipelining, expose
requested II and recurrence/resource-pressure summaries.

This is general HLS semantics, not hls4ml layer fusion.

## CPU-first evidence program

CPU work should decide whether the new representation contains a plausible
large gain before authorizing GPU training.

### A. Canonicalization audit

Run `ir_canonicalization_probe.py` on 5–10 projects per synthetic family,
stratified by graph size and strategy. The acceptance criterion for a pass
pipeline is not maximum shrinkage. It is substantial median instruction
reduction with complete retention of function count, HLS intrinsics, pragma
carriers/targets, and recoverable loop anchors. Then regenerate graphs for a
small sample and compare relation/node invariants.

### B. LLVM-only tabular baselines

Build one row per graph using only compiler-derived information:

- opcode/type histograms and precision-weighted operation counts;
- per-loop and per-depth operation counts, raw and trip-count weighted;
- unroll-adjusted parallel demand and residual serial work;
- additive resource-cost summaries using simple opcode/type lookup bases;
- block/loop DAG longest paths and recurrence summaries;
- per-memory-object reads, writes, width, size, partition factor, bank/port
  pressure;
- pragma histograms and concrete arguments;
- function/call-depth summaries.

Fit ExtraTrees and histogram gradient boosting to log targets, with the existing
DSP/BRAM occurrence treatment, using validation for every choice. Compare
nested feature groups: raw graph → canonical graph → + loop hierarchy → +
pragma-derived quantities → + memory/critical-path summaries. This is a test of
information and representation, not the final surrogate.

The old graph-derived RBF evidence already says semantic type and pragma
arguments are valuable while coarse edge counts are not. The new CPU probe must
ask the missing question: do explicit loop/work/path/memory features produce a
multi-point validation improvement? If they do not, a neural encoder over the
same information is unlikely to create 5–10 points.

### C. Acceptance thresholds

Do not promote one-point improvements. A representation family should improve
validation by about three points in the cheap probe or show a comparably large,
target-localized effect before receiving a full GPU run. The standalone GNN
must ultimately improve H0 by at least five test SMAPE points to count as the
desired architectural result. Extra seeds are for confirming such a result, not
rescuing close runs.

## Compute-efficient GPU plan

Use one primary architecture, not a ten-model search.

1. **Canonicalized H0 control.** Train H0 unchanged on the accepted conservative
   IR representation. This measures how much of the wall is input noise and
   should be cheaper because graphs are smaller.
2. **Region-HLS model.** Train the complete explicit-loop,
   pragma-conditioned, dual-composition model at 96–128 hidden dimensions and
   two local layers. Share weights across loop levels.
3. **One mechanistic ablation.** If the model wins materially, remove the loop
   hierarchy or replace target-aligned composition with the generic H0 pool.
   Choose the ablation that best supports the thesis claim; do not run a broad
   operator grid.

Use a single seed for gating. Stop candidates that cannot beat H0 validation by
roughly three points. Run two additional seeds only after a candidate shows a
credible path to a five-point test gain. Report three-seed results for the final
model.

Canonicalization can fund some of the new computation: the sample's conservative
pass has roughly one-sixth as many instruction lines. Cache all compiler
analyses and tensor relations; never recompute loops, dominators, SCEV, or
recurrences during training.

## Data and robustness

The compact cohort has 2,742 train, 569 validation, and 593 deduplicated test
graphs. H0 has 239,568 parameters. A strong prior helps, but a genuinely robust
program model also needs structural diversity. HARP used 40k programs, and the
wa-hls4ml paper used hundreds of thousands of examples. Architecture alone may
not overcome this sample-size gap.

After canonicalization makes graph processing cheaper, expand the labeled CDFG
cohort selectively to roughly 5k–10k rather than blindly ingesting everything.
Use a CPU structural fingerprint—loop-depth/trip-count bins, opcode/type
histograms, pragma combinations, memory pressure, call depth, family—and select
diverse projects by farthest-point or cluster sampling. This remains a general
LLVM training set; no high-level fusion is required.

The current IID split is necessary but insufficient for “robust.” Add one final
structural challenge split, such as held-out parameter/loop-depth ranges and one
held-out kernel family. Do not make exemplar the sole robustness claim: it
simultaneously changes architectures, boards, clocks, tool versions, and target
support.

Only seven prepared architecture groups (14 samples) cross official splits, so
fixing those groups is hygiene rather than an explanation of current scores.
The larger evaluation issue is adaptive reuse: the 593-project test has informed
several architecture decisions. Freeze it now and create a fresh, previously
uninspected lockbox from unused wa-hls4ml projects for the final model if
possible.

Backend, part, and hls4ml version are sparsely crossed with family. Retain the
metadata, but report a no-context primary result or context ablation so cohort
identity is not mistaken for toolchain robustness.

## Target/statistical implications

- DSP is zero in 35.1% of designs. BRAM is zero in 16.9%, but zero occurrence is
  heavily family-determined; all dense-latency examples are zero. Hurdle
  accuracy alone is therefore weak evidence. Preserve occurrence handling but
  judge positive magnitudes and families separately.
- Cycles and II have log correlation 0.94, but cycles/II ranges broadly. Share a
  timing encoder while predicting distinct quantities.
- LUT and FF have log correlation 0.76 and are natural consumers of an additive
  learned cost representation.
- Paired project bootstraps are useful for the final large gain. They should not
  be used to elevate one-point architecture differences that are irrelevant to
  the research objective.

Report aggregate, target, family, structural-OOD, and final lockbox results,
plus parameter count and training cost. A general model should not obtain its
gain from one family or only from zero classification.

## What not to spend the next GPU budget on

Do not run another isolated GAT/TransformerConv/PNA/GRU swap, more instruction
sequence modeling, a rescue of the program BiGRU, or extensive scheduler/head
tuning. None has a mechanism for the required gain. Cardinality-aware messages
are worth implementing only inside the semantic region model.

Do not treat fusion, hurdle thresholds, synthesis context, or minor
tensor-revision bookkeeping as the central thesis. They can improve a final
system or clean the evaluation, but they do not solve general LLVM CDFG
reasoning.

## Thesis claim

A defensible contribution is:

> Accurate and robust HLS prediction from LLVM requires program hierarchy at
> the level where hardware decisions act. Explicit loop/region composition,
> pragma-conditioned multiplicity and memory semantics, and distinct additive
> resource versus path-sensitive timing computations provide a stronger prior
> than generic message-passing over a flat or block-only CDFG.

H0 is the baseline evidence that hierarchy matters. The failed architecture
family shows that generic local expressivity is not enough. Conservative IR
canonicalization, explicit loops, and target-aligned composition are the next
coherent hypothesis. Fusion can be reported afterward as an optional ceiling
or deployment enhancement, not as the thesis centerpiece.
