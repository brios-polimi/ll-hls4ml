# Surrogate models

Models consume the typed tensor contract from `../data` and register stable
experiment names in `registry.py`. Keep reusable encoders separate from heads;
multimodal hls4ml-specific inputs should remain optional so LLVM-only and future
backend datasets can use the same surrogate family.

Do not change recorded experiment behavior under an existing model name. Add a
new name when architecture or output semantics change materially.
