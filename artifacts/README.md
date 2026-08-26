# Local research artifacts

This directory is stateful and gitignored. It may contain caches, tensors,
checkpoints, run bundles, plots, and result exports. Durable experiment evidence
should be summarized in a small report or manifest deliberately added elsewhere;
large generated files should use external artifact storage.

Scripts should write each run to a new experiment directory and record its
resolved configuration, split manifest, code revision, metrics, and provenance.
