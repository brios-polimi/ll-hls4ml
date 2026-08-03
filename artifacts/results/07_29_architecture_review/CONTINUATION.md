# Continuation handoff

## Resume point

The representation audit, tensor-v2 rebuild, cheap feature baselines, context
ablation, sampling ablation, and pooling ablation are complete. No
hierarchical/block-feature work has started.

Primary comparison baseline:

- config: `configs/mlp_global_no_context.yaml`
- tensor snapshot: `4b637ba749da56c5`
- test macro SMAPE: 41.86%
- exemplar macro SMAPE: 103.38%
- family-balanced sampling
- multi-statistic pooling
- graph-wide shortcut enabled
- synthesis-context branch disabled
- split resource/timing heads
- DSP/BRAM hurdle outputs

“No context” only disables backend, target part, target clock, Vivado/Vitis
version, and hls4ml version. It retains the full typed CDFG representation.

## Short diagnostics completed after the handoff

### Graph-wide shortcut

Config: `configs/mlp_multi_no_global.yaml`

Result: 43.70% test macro SMAPE without the shortcut versus 41.86% with it.
Family-macro is 42.05% versus 39.42%. These 1.8–2.6 point gaps are comparable
with within-run checkpoint fluctuations, so shortcut necessity is unresolved.
Retraining shows that the learned pooling branch can compensate; it does not
show that the original checkpoint ignored the shortcut.

### DSP/BRAM hurdle

Config: `configs/mlp_split_no_hurdle.yaml`

Result: 45.95% test macro SMAPE without the hurdle versus 41.86% with it.
DSP changes from 35.84% to 50.78% and BRAM from 39.91% to 46.26%. This
target-specific effect agrees with the independent cheap hurdle baseline and is
the strongest neural ablation signal, while still requiring multi-seed
confirmation. The paired test bootstrap gives a +4.09 macro difference
(95% interval +2.43 to +5.80). Positive-value regression is nearly unchanged;
the benefit occurs on exact-zero DSP/BRAM samples.

## Deferred short runs

Config: `configs/mlp_shared_no_hurdle.yaml`

The shared-versus-split-head run is deferred. At one short seed, its likely
effect is too close to the observed 1.6–2.5 point within-run validation standard
deviation to justify another architecture ranking.

Config: `configs/hetero_gat_v2_no_context.yaml`

The 10-epoch, one-layer GAT accuracy run is also deferred. Correctness and
end-to-end smoke tests pass, but that run would only show that the software
trains. It is too shallow and short to test whether topology helps resource or
critical-path prediction.

## Command

Use the same command for each config:

```bash
MPLCONFIGDIR=/tmp/ll_hls4ml_mpl \
LL_HLS4ML_TQDM=0 \
PYTHONPATH=src \
/home/brend/anaconda3/bin/conda run --no-capture-output \
  -n pipeline-env python -u scripts/train.py \
  --config <config-path>
```

## Interpretation rules

- Treat synthetic test macro SMAPE as the primary comparison.
- Report unweighted family-macro SMAPE as a second view.
- Treat exemplar as a separately labeled compound-OOD stress test.
- Compare directions and effect sizes; these models are not converged.
- Do not claim parity or inferiority to wa-hls4ml from 10–20 epoch neural runs.
- Promote an ablation choice only if it is semantically plausible and the effect
  is not isolated to one family/target.
- Treat 1–3 point neural differences as unresolved at one seed.
- Treat 4–5 point differences as hypotheses unless the effect is target-specific
  and independently replicated.

## Rented-GPU continuation

Use at least three seeds, 60+ epochs, larger patience, and a learning-rate
schedule. The minimum high-value matrix is:

1. pooled MLP with and without the DSP/BRAM hurdle;
2. corrected heterogeneous GAT with 2–3 layers versus the matched pooled MLP;
3. only then, shared versus split heads if the first two comparisons have stable
   seed variance.

Keep synthesis context available but compare it separately until crossed
graph-by-context data exists. The next representation pass should add
path/recurrence/loop features and only then revisit hierarchical block pooling.
