#!/usr/bin/env python3
"""Cheap nonlinear probe for information in the cached high-level features.

This is deliberately a diagnostic rather than a proposed final surrogate.  It
uses the official split, selects its representation/regularization on validation
SMAPE, and evaluates the untouched test and exemplar splits once.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


TARGETS = ("lut", "ff", "dsp", "bram", "cycles_max", "interval_max")
HURDLE_INDICES = (2, 3)


def _aggregate(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            values.sum(axis=0),
            values.mean(axis=0),
            values.max(axis=0),
            values.min(axis=0),
            values.std(axis=0),
            values[0],
            values[-1],
        ]
    )


def _positive_product(values: np.ndarray) -> np.ndarray:
    return np.prod(np.where(values > 0, values, 1.0), axis=1)


def _summary_features(raw: np.ndarray, include_cost_basis: bool) -> np.ndarray:
    raw = raw.astype(np.float64)
    logged = np.sign(raw) * np.log1p(np.abs(raw))
    input_size = _positive_product(raw[:, :3])
    output_size = _positive_product(raw[:, 3:6])
    precision = np.maximum(raw[:, 6], 1.0)
    reuse = np.maximum(raw[:, 7], 1.0)
    filters = np.maximum(raw[:, 11], 1.0)
    kernel = np.maximum(raw[:, 12], 1.0)
    workload = input_size * output_size
    cost_basis = np.column_stack(
        [
            input_size,
            output_size,
            workload,
            workload * precision,
            workload / reuse,
            output_size / reuse,
            output_size * filters * kernel,
            output_size * filters * np.square(kernel),
            reuse,
            precision,
        ]
    )
    categorical = []
    for column, classes in ((9, 12), (10, 6), (14, 3)):
        ids = raw[:, column].astype(int).clip(0, classes - 1)
        counts = np.bincount(ids, minlength=classes).astype(float)
        categorical.extend([counts, counts / len(raw)])
    parts = [
        np.asarray([len(raw), np.log1p(len(raw))]),
        _aggregate(raw),
        _aggregate(logged),
    ]
    if include_cost_basis:
        parts.append(_aggregate(np.log1p(cost_basis)))
    parts.extend(categorical)
    return np.concatenate(parts)


def _sequence_features(raw: np.ndarray, max_layers: int) -> np.ndarray:
    values = np.sign(raw) * np.log1p(np.abs(raw))
    padded = np.zeros((max_layers, values.shape[1]), dtype=np.float64)
    padded[: len(values)] = values
    return padded.ravel()


def _smape(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return 200.0 * np.abs(target - prediction) / (
        np.abs(target) + np.abs(prediction) + 1.0
    )


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    exemplar_x: np.ndarray,
    min_samples_leaf: int,
    trees: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    regressor = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=min_samples_leaf,
        max_features=0.7,
        n_jobs=-1,
        random_state=seed,
    )
    regressor.fit(train_x, np.log1p(train_y))
    predictions = [
        np.maximum(np.expm1(regressor.predict(features)), 0.0)
        for features in (val_x, test_x, exemplar_x)
    ]
    thresholds = []
    for target_index in HURDLE_INDICES:
        classifier = ExtraTreesClassifier(
            n_estimators=trees,
            min_samples_leaf=min_samples_leaf,
            max_features=0.7,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed + target_index,
        )
        classifier.fit(train_x, train_y[:, target_index] > 0)
        probabilities = [
            classifier.predict_proba(features)[:, 1]
            for features in (val_x, test_x, exemplar_x)
        ]
        candidates = np.linspace(0.05, 0.95, 19)
        threshold = min(
            candidates,
            key=lambda value: _smape(
                val_y[:, target_index],
                predictions[0][:, target_index]
                * (probabilities[0] >= value),
            ).mean(),
        )
        thresholds.append(float(threshold))
        for prediction, probability in zip(predictions, probabilities):
            prediction[:, target_index] *= probability >= threshold
    return predictions[0], predictions[1], predictions[2], thresholds


def _table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
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
        "--labels",
        type=Path,
        default=Path("../data/tensors/labels.json"),
    )
    parser.add_argument("--trees", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predictions-output", type=Path)
    args = parser.parse_args()

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    label_index = json.loads(args.labels.read_text())
    metadata = label_index["metadata"]
    paths = list(cache["samples"])
    raw = [cache["samples"][path]["features"].numpy() for path in paths]
    target = np.stack([cache["samples"][path]["target"].numpy() for path in paths])
    family = np.asarray([cache["samples"][path]["kernel_family"] for path in paths])
    split = np.asarray(
        [
            "exemplar"
            if family[index] == "exemplar"
            else (
                "validation"
                if str(metadata[path]["dataset_split"]) in {"val", "validation"}
                else str(metadata[path]["dataset_split"])
            )
            for index, path in enumerate(paths)
        ]
    )
    max_layers = max(map(len, raw))
    raw_summary = np.stack(
        [_summary_features(values, include_cost_basis=False) for values in raw]
    )
    cost_summary = np.stack(
        [_summary_features(values, include_cost_basis=True) for values in raw]
    )
    sequence = np.stack([_sequence_features(values, max_layers) for values in raw])
    feature_sets = {
        "raw-summary": raw_summary,
        "cost-summary": cost_summary,
        "cost-summary+sequence": np.column_stack([cost_summary, sequence]),
    }
    masks = {name: split == name for name in ("train", "validation", "test", "exemplar")}

    candidates = []
    fitted = {}
    for feature_name, features in feature_sets.items():
        for leaf in (1, 2, 4, 8):
            result = _fit_predict(
                features[masks["train"]],
                target[masks["train"]],
                features[masks["validation"]],
                target[masks["validation"]],
                features[masks["test"]],
                features[masks["exemplar"]],
                leaf,
                args.trees,
                args.seed,
            )
            val_prediction, test_prediction, exemplar_prediction, thresholds = result
            val_smape = float(_smape(target[masks["validation"]], val_prediction).mean())
            key = (feature_name, leaf)
            candidates.append((val_smape, feature_name, leaf, thresholds))
            fitted[key] = (val_prediction, test_prediction, exemplar_prediction)

    candidates.sort()
    _, feature_name, leaf, thresholds = candidates[0]
    val_prediction, test_prediction, exemplar_prediction = fitted[(feature_name, leaf)]
    predictions = {
        "validation": val_prediction,
        "test": test_prediction,
        "exemplar": exemplar_prediction,
    }
    report = [
        "# High-level tabular information probe",
        "",
        "Validation-selected ExtraTrees over deterministic high-level summaries. "
        "No test result was used for model selection.",
        "",
        f"Selected representation: `{feature_name}`; min_samples_leaf: {leaf}; "
        f"DSP/BRAM thresholds: {thresholds}.",
        "",
        "## Validation candidates",
        "",
        _table(
            ("features", "min leaf", "validation SMAPE"),
            [
                (name, candidate_leaf, f"{score:.2f}")
                for score, name, candidate_leaf, _ in candidates
            ],
        ),
        "",
        "## Selected-model metrics",
        "",
    ]
    metric_rows = []
    for name, prediction in predictions.items():
        split_target = target[masks[name]]
        per_target = _smape(split_target, prediction).mean(axis=0)
        metric_rows.append((name, "macro", f"{per_target.mean():.2f}"))
        metric_rows.extend(
            (name, target_name, f"{value:.2f}")
            for target_name, value in zip(TARGETS, per_target)
        )
    report.append(_table(("split", "scope", "SMAPE"), metric_rows))
    report.extend(["", "Test SMAPE by family:", ""])
    test_family = family[masks["test"]]
    test_target = target[masks["test"]]
    family_rows = []
    for name in sorted(set(test_family)):
        mask = test_family == name
        family_rows.append(
            (
                name,
                int(mask.sum()),
                f"{_smape(test_target[mask], test_prediction[mask]).mean():.2f}",
            )
        )
    report.append(_table(("family", "N", "SMAPE"), family_rows))
    text = "\n".join(report) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_output.open("w", newline="") as handle:
            fieldnames = [
                "tensor_path",
                "split",
                "kernel_family",
                *[f"target_{name}" for name in TARGETS],
                *[f"prediction_{name}" for name in TARGETS],
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for split_name, prediction in predictions.items():
                indices = np.flatnonzero(masks[split_name])
                for local_index, global_index in enumerate(indices):
                    row = {
                        "tensor_path": paths[global_index],
                        "split": split_name,
                        "kernel_family": family[global_index],
                    }
                    row.update(
                        {
                            f"target_{name}": target[global_index, column]
                            for column, name in enumerate(TARGETS)
                        }
                    )
                    row.update(
                        {
                            f"prediction_{name}": prediction[local_index, column]
                            for column, name in enumerate(TARGETS)
                        }
                    )
                    writer.writerow(row)
    print(text)


if __name__ == "__main__":
    main()
