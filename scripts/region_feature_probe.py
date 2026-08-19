#!/usr/bin/env python3
"""CPU-only information probe for natural-loop hierarchy features.

This is a representation diagnostic, not a proposed fusion model. It compares
the same deterministic high-level/CFG control with and without loop summaries,
selects ExtraTrees regularization by grouped cross-validation inside the
official training split, and reports the official validation split only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SRC = REPO_ROOT.parent / "hls4ml_pipeline" / "src"
sys.path.insert(0, str(PIPELINE_SRC))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hls4ml_pipeline.pipeline.hierarchy import parse_llvm_hierarchy  # noqa: E402
from hls4ml_pipeline.pipeline.loops import natural_loops  # noqa: E402
from high_level_tabular_probe import _summary_features  # noqa: E402


TARGETS = ("lut", "ff", "dsp", "bram", "cycles_max", "interval_max")
HURDLE_INDICES = (2, 3)


def _smape(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return 200.0 * np.abs(target - prediction) / (
        np.abs(target) + np.abs(prediction) + 1.0
    )


def _aggregate(values: np.ndarray, width: int) -> np.ndarray:
    if not len(values):
        return np.zeros(5 * width, dtype=np.float64)
    return np.concatenate(
        [
            values.sum(axis=0),
            values.mean(axis=0),
            values.max(axis=0),
            values.min(axis=0),
            values.std(axis=0),
        ]
    )


def _ir_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    functions = parse_llvm_hierarchy(path)
    function_rows = []
    loop_rows = []
    loop_functions = 0
    for blocks in functions.values():
        cfg_edges = sum(len(block.successors) for block in blocks)
        function_rows.append([len(blocks), cfg_edges])
        loops = natural_loops(blocks)
        loop_functions += bool(loops)
        loop_rows.extend(
            [
                loop.depth,
                len(loop.blocks),
                len(loop.direct_blocks),
                len(loop.latches),
                len(loop.exits),
            ]
            for loop in loops
        )
    function_values = np.asarray(function_rows, dtype=np.float64).reshape(-1, 2)
    loop_values = np.asarray(loop_rows, dtype=np.float64).reshape(-1, 5)
    cfg = np.concatenate(
        [
            np.asarray([len(functions)], dtype=np.float64),
            _aggregate(function_values, 2),
        ]
    )
    loop = np.concatenate(
        [
            np.asarray(
                [
                    len(loop_rows),
                    loop_functions,
                    sum(row[0] == 0 for row in loop_rows),
                    sum(row[0] > 0 for row in loop_rows),
                ],
                dtype=np.float64,
            ),
            _aggregate(loop_values, 5),
        ]
    )
    return np.log1p(cfg), np.log1p(loop)


def _stable_subset(
    paths: list[str], split: np.ndarray, family: np.ndarray, limit: int, seed: int
) -> np.ndarray:
    selected = []
    for split_name in ("train", "validation"):
        for family_name in sorted(set(family)):
            candidates = np.flatnonzero(
                (split == split_name) & (family == family_name)
            ).tolist()
            candidates.sort(
                key=lambda index: hashlib.sha256(
                    f"{seed}:{paths[index]}".encode()
                ).digest()
            )
            selected.extend(candidates if limit <= 0 else candidates[:limit])
    return np.asarray(sorted(selected), dtype=np.int64)


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    leaf: int,
    trees: int,
    seed: int,
) -> np.ndarray:
    regressor = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=leaf,
        max_features=0.7,
        n_jobs=-1,
        random_state=seed,
    )
    regressor.fit(train_x, np.log1p(train_y))
    prediction = np.maximum(np.expm1(regressor.predict(predict_x)), 0.0)
    for target_index in HURDLE_INDICES:
        presence = train_y[:, target_index] > 0
        if np.unique(presence).size < 2:
            prediction[:, target_index] *= bool(presence[0])
            continue
        classifier = ExtraTreesClassifier(
            n_estimators=trees,
            min_samples_leaf=leaf,
            max_features=0.7,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed + target_index,
        )
        classifier.fit(train_x, presence)
        prediction[:, target_index] *= (
            classifier.predict_proba(predict_x)[:, 1] >= 0.5
        )
    return prediction


def _select_leaf(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    trees: int,
    seed: int,
) -> tuple[int, list[tuple[int, float]]]:
    candidates = []
    folds = min(4, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=folds)
    for leaf in (1, 2, 4, 8):
        prediction = np.zeros_like(target)
        for fold, (train_index, val_index) in enumerate(
            splitter.split(features, groups=groups)
        ):
            prediction[val_index] = _fit_predict(
                features[train_index],
                target[train_index],
                features[val_index],
                leaf,
                trees,
                seed + fold * 100,
            )
        candidates.append((leaf, float(_smape(target, prediction).mean())))
    return min(candidates, key=lambda item: item[1])[0], candidates


def _table(headers, rows) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("artifacts/cache/wa_high_level_archives1_8_exemplar1_9.pt"),
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("../data/tensors/labels.json")
    )
    parser.add_argument("--llvm-root", type=Path, default=Path("../data/ll"))
    parser.add_argument("--max-per-family-split", type=int, default=40)
    parser.add_argument("--trees", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    label_index = json.loads(args.labels.read_text())
    metadata = label_index["metadata"]
    all_paths = list(cache["samples"])
    family = np.asarray(
        [cache["samples"][path]["kernel_family"] for path in all_paths]
    )
    split = np.asarray(
        [
            "validation"
            if str(metadata[path].get("dataset_split")) in {"val", "validation"}
            else str(metadata[path].get("dataset_split"))
            for path in all_paths
        ]
    )
    subset = _stable_subset(
        all_paths, split, family, args.max_per_family_split, args.seed
    )

    records = []
    missing = []
    for index in subset:
        relative = Path(all_paths[index])
        llvm_path = args.llvm_root / relative.with_suffix(".ll")
        if not llvm_path.is_file():
            missing.append(relative.as_posix())
            continue
        cfg, loops = _ir_features(llvm_path)
        raw = cache["samples"][all_paths[index]]["features"].numpy()
        high_level = _summary_features(raw, include_cost_basis=True)
        records.append((index, high_level, cfg, loops))
    if not records:
        raise RuntimeError("No matching LLVM files were found")

    indices = np.asarray([record[0] for record in records])
    families = family[indices]
    family_names = sorted(set(families))
    family_features = np.asarray(
        [[name == value for name in family_names] for value in families],
        dtype=np.float64,
    )
    base = np.column_stack(
        [
            np.stack([record[1] for record in records]),
            np.stack([record[2] for record in records]),
            family_features,
        ]
    )
    region = np.column_stack(
        [base, np.stack([record[3] for record in records])]
    )
    target = np.stack(
        [cache["samples"][all_paths[index]]["target"].numpy() for index in indices]
    )
    selected_split = split[indices]
    train_mask = selected_split == "train"
    val_mask = selected_split == "validation"
    groups = np.asarray(
        ["/".join(Path(path).parts[:2]) for path in np.asarray(all_paths)[indices]]
    )

    results = {}
    cv_rows = []
    for name, features in (("control", base), ("control+region", region)):
        leaf, candidates = _select_leaf(
            features[train_mask],
            target[train_mask],
            groups[train_mask],
            args.trees,
            args.seed,
        )
        prediction = _fit_predict(
            features[train_mask],
            target[train_mask],
            features[val_mask],
            leaf,
            args.trees,
            args.seed,
        )
        per_target = _smape(target[val_mask], prediction).mean(axis=0)
        results[name] = (per_target, prediction)
        cv_rows.extend((name, candidate, f"{score:.2f}") for candidate, score in candidates)

    control, control_prediction = results["control"]
    region_result, region_prediction = results["control+region"]
    validation_target = target[val_mask]
    paired_delta = (
        _smape(validation_target, region_prediction).mean(axis=1)
        - _smape(validation_target, control_prediction).mean(axis=1)
    )
    rng = np.random.default_rng(args.seed)
    bootstrap = np.asarray(
        [
            paired_delta[
                rng.integers(0, len(paired_delta), len(paired_delta))
            ].mean()
            for _ in range(5000)
        ]
    )
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    metric_rows = [
        (
            "macro",
            f"{control.mean():.2f}",
            f"{region_result.mean():.2f}",
            f"{region_result.mean() - control.mean():+.2f}",
        )
    ]
    metric_rows.extend(
        (
            target_name,
            f"{before:.2f}",
            f"{after:.2f}",
            f"{after - before:+.2f}",
        )
        for target_name, before, after in zip(TARGETS, control, region_result)
    )
    family_rows = []
    for family_name in family_names:
        mask = families[val_mask] == family_name
        before = _smape(
            validation_target[mask], control_prediction[mask]
        ).mean()
        after = _smape(
            validation_target[mask], region_prediction[mask]
        ).mean()
        family_rows.append(
            (family_name, int(mask.sum()), f"{before:.2f}", f"{after:.2f}", f"{after - before:+.2f}")
        )
    loop_matrix = np.stack([record[3] for record in records])
    structure_rows = []
    for family_name in family_names:
        mask = families == family_name
        loop_counts = np.rint(np.expm1(loop_matrix[mask, 0])).astype(int)
        nested_counts = np.rint(np.expm1(loop_matrix[mask, 3])).astype(int)
        unique_signatures = np.unique(loop_matrix[mask].round(8), axis=0).shape[0]
        structure_rows.append(
            (
                family_name,
                f"{loop_counts.min()}/{np.median(loop_counts):.0f}/{loop_counts.max()}",
                f"{nested_counts.min()}/{np.median(nested_counts):.0f}/{nested_counts.max()}",
                unique_signatures,
            )
        )
    counts = defaultdict(int)
    for split_name, family_name in zip(selected_split, families):
        counts[(split_name, family_name)] += 1
    report = [
        "# Natural-loop CPU information probe",
        "",
        "This is an information check, not a proposed high-level fusion model. "
        "Hyperparameters are selected by archive-grouped CV within train; the "
        "official validation split is reported once. Test and exemplar are not read.",
        "",
        f"Retained {len(records)} samples; missing LLVM files: {len(missing)}.",
        "",
        _table(
            ("split", "family", "N"),
            [(key[0], key[1], value) for key, value in sorted(counts.items())],
        ),
        "",
        "## Training CV",
        "",
        _table(("features", "min leaf", "OOF SMAPE"), cv_rows),
        "",
        "## Official validation",
        "",
        _table(("target", "control", "+region", "delta"), metric_rows),
        "",
        f"Paired sample-bootstrap 95% CI for the macro delta: "
        f"[{lower:+.2f}, {upper:+.2f}] SMAPE.",
        "",
        _table(("family", "N", "control", "+region", "delta"), family_rows),
        "",
        "## Region-structure diversity",
        "",
        _table(
            ("family", "loops min/med/max", "nested min/med/max", "unique summaries"),
            structure_rows,
        ),
        "",
        "Negative deltas favor the explicit region features. A small or mixed "
        "delta is evidence against spending scarce GPU time on this path as-is; "
        "it is not rescued by seed replication.",
        "",
    ]
    text = "\n".join(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
