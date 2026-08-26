# Training and evaluation

This subtree owns loaders, target transforms, loops, telemetry, and distributed
helpers. A reproducible run records resolved configuration, frozen split
manifest, code revision, target order/normalization, metrics, and predictions.

Tests use `unittest` and tiny synthetic inputs. Do not launch full, distributed,
or GPU training as validation unless the task explicitly requests it.
