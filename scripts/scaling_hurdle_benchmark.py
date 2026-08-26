#!/usr/bin/env python3
"""Fit the graph-feature hurdle RBF on saved neural scaling splits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hls-surrogate-lab-matplotlib")
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from ll_hls4ml.data.dataset import HeteroGraphDataset
from ll_hls4ml.data.vocab import load_vocab
from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.training.targets import wahls4ml_metrics_raw
from preliminary_benchmark import FAMILIES, _tensor_features


SCALES = ("025", "050", "100", "200")
HURDLE_TARGETS = (2, 3)


def _load_splits(
    results_root: Path,
    path_to_index: dict[str, int],
) -> tuple[dict[str, dict[str, list[int]]], dict[str, Path]]:
    splits = {}
    run_dirs = {}
    for scale in SCALES:
        matches = sorted(results_root.glob(f"*scale{scale}_seed42"))
        if len(matches) != 1:
            raise ValueError(f"Expected one scale{scale} run, found {matches}")
        run_dir = matches[0]
        manifest = json.loads((run_dir / "split_manifest.json").read_text())
        run_dirs[scale] = run_dir
        splits[scale] = {}
        for split_name in ("train", "validation", "test", "exemplar"):
            try:
                splits[scale][split_name] = [
                    path_to_index[item["tensor_path"]]
                    for item in manifest[split_name]
                ]
            except KeyError as error:
                raise KeyError(
                    f"A scale{scale} manifest path is absent from the tensor dataset: {error}"
                ) from error
    reference = splits["025"]
    for scale in SCALES[1:]:
        for split_name in ("validation", "test", "exemplar"):
            if splits[scale][split_name] != reference[split_name]:
                raise ValueError(f"scale{scale} has a different {split_name} split")
    return splits, run_dirs


def _feature_frame(
    dataset: HeteroGraphDataset,
    indices: list[int],
    instruction_vocab_size: int,
    cache_path: Path,
) -> pd.DataFrame:
    if cache_path.exists():
        frame = pd.read_csv(cache_path)
        if set(frame["dataset_index"]) == set(indices):
            print(f"Using cached features from {cache_path}", flush=True)
            return frame
        print("Cached feature indices differ; rebuilding", flush=True)

    rows = []
    for position, index in enumerate(indices, start=1):
        data = dataset[index]
        row = _tensor_features(data, instruction_vocab_size)
        row.update(
            dataset_index=index,
            kernel_family=dataset.type_of(index),
            tensor_path=dataset.paths[index].relative_to(dataset.root).as_posix(),
        )
        for target_index, label in enumerate(LABEL_KEYS):
            row[label] = float(data.y[target_index])
        rows.append(row)
        if position % 100 == 0 or position == len(indices):
            print(f"Tabular features: {position}/{len(indices)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(cache_path, index=False)
    return frame


def _fit_predict(
    frame: pd.DataFrame,
    train_indices: list[int],
    evaluation_indices: list[int],
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    metadata = {"dataset_index", "kernel_family", "tensor_path", *LABEL_KEYS}
    feature_columns = [column for column in frame.columns if column not in metadata]
    indexed = frame.set_index("dataset_index")
    train_x = indexed.loc[train_indices, feature_columns].to_numpy(np.float32)
    train_x = np.nan_to_num(train_x, nan=0.0, posinf=0.0, neginf=0.0)
    train_y = indexed.loc[train_indices, LABEL_KEYS].to_numpy(np.float32)
    evaluation_x = indexed.loc[evaluation_indices, feature_columns].to_numpy(np.float32)
    evaluation_x = np.nan_to_num(evaluation_x, nan=0.0, posinf=0.0, neginf=0.0)
    targets = indexed.loc[evaluation_indices, LABEL_KEYS].to_numpy(np.float32)

    input_scaler = StandardScaler().fit(train_x)
    train_x = input_scaler.transform(train_x)
    evaluation_x = input_scaler.transform(evaluation_x)
    target_scaler = StandardScaler().fit(np.log1p(train_y))
    standardized_y = target_scaler.transform(np.log1p(train_y))

    regressors = []
    classifiers = {}
    for target_index in range(len(LABEL_KEYS)):
        positive = train_y[:, target_index] > 0
        fit_rows = positive if target_index in HURDLE_TARGETS else np.ones(len(train_y), dtype=bool)
        regressor = SVR(C=10.0, epsilon=0.05, gamma="scale")
        regressor.fit(train_x[fit_rows], standardized_y[fit_rows, target_index])
        regressors.append(regressor)
        if target_index in HURDLE_TARGETS:
            classifier = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=seed
            )
            classifier.fit(train_x, positive)
            classifiers[target_index] = classifier

    started = time.perf_counter()
    prediction_scaled = np.column_stack(
        [regressor.predict(evaluation_x) for regressor in regressors]
    )
    raw_positive = np.expm1(target_scaler.inverse_transform(prediction_scaled))
    probabilities = {
        index: classifier.predict_proba(evaluation_x)[:, 1]
        for index, classifier in classifiers.items()
    }
    predictions = {}
    for mode in ("expected", "threshold"):
        values = raw_positive.copy()
        for target_index, probability in probabilities.items():
            multiplier = probability if mode == "expected" else probability >= 0.5
            values[:, target_index] *= multiplier
        predictions[mode] = np.clip(values, 0, None).astype(np.float32)
    elapsed = time.perf_counter() - started
    return predictions, targets, elapsed / len(targets)


def _metric_rows(
    scale: str,
    mode: str,
    split: str,
    families: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    latency: float,
) -> list[dict]:
    rows = []
    family_names = ["all", *sorted(set(families))]
    for family in family_names:
        mask = np.ones(len(families), dtype=bool) if family == "all" else families == family
        metrics = wahls4ml_metrics_raw(
            torch.from_numpy(predictions[mask]), torch.from_numpy(targets[mask])
        )
        for target_index, target in enumerate(LABEL_KEYS):
            rows.append(
                {
                    "scale": int(scale),
                    "model": f"logistic_hurdle_rbf_{mode}_graph",
                    "split": split,
                    "kernel_family": family,
                    "target": target,
                    "smape": float(metrics["smape"][target_index]),
                    "r2": float(metrics["r2"][target_index]),
                    "rmse": float(metrics["rmse"][target_index]),
                    "n_samples": int(mask.sum()),
                    "inference_seconds_per_sample": latency,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-dir", default="../data/tensors")
    parser.add_argument(
        "--results-root",
        default="artifacts/results/ll_hls4ml_gatv2_scaling_results",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tensor_root = Path(args.tensor_dir).resolve()
    results_root = Path(args.results_root).resolve()
    output_dir = results_root / "rbf_hurdle_matched"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = HeteroGraphDataset(tensor_root, types=FAMILIES, silent=False)
    path_to_index = {
        path.relative_to(tensor_root).as_posix(): index
        for index, path in enumerate(dataset.paths)
    }
    splits, run_dirs = _load_splits(results_root, path_to_index)
    all_indices = sorted(
        set().union(
            *(set(split_indices) for scale in SCALES for split_indices in splits[scale].values())
        )
    )
    vocabulary, _, _ = load_vocab(tensor_root / "vocab.json")
    frame = _feature_frame(
        dataset,
        all_indices,
        len(vocabulary),
        output_dir / "tabular_features.csv",
    )

    indexed = frame.set_index("dataset_index")
    metric_rows = []
    prediction_rows = []
    for scale in SCALES:
        print(f"Fitting scale{scale} on {len(splits[scale]['train'])} graphs", flush=True)
        evaluation_indices = splits[scale]["test"] + splits[scale]["exemplar"]
        all_predictions, all_targets, latency = _fit_predict(
            frame, splits[scale]["train"], evaluation_indices, args.seed
        )
        offset = 0
        for split in ("test", "exemplar"):
            indices = splits[scale][split]
            end = offset + len(indices)
            predictions = {
                mode: values[offset:end]
                for mode, values in all_predictions.items()
            }
            targets = all_targets[offset:end]
            offset = end
            families = indexed.loc[indices, "kernel_family"].to_numpy(str)
            paths = indexed.loc[indices, "tensor_path"].to_numpy(str)
            for mode, values in predictions.items():
                metric_rows.extend(
                    _metric_rows(scale, mode, split, families, values, targets, latency)
                )
                for row_index, dataset_index in enumerate(indices):
                    row = {
                        "scale": int(scale),
                        "model": f"logistic_hurdle_rbf_{mode}_graph",
                        "split": split,
                        "dataset_index": dataset_index,
                        "kernel_family": families[row_index],
                        "tensor_path": paths[row_index],
                    }
                    for target_index, label in enumerate(LABEL_KEYS):
                        row[f"target_{label}"] = targets[row_index, target_index]
                        row[f"prediction_{label}"] = values[row_index, target_index]
                    prediction_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)
    summary = (
        metrics[metrics.kernel_family == "all"]
        .groupby(["scale", "model", "split"], as_index=False)
        .agg(macro_smape=("smape", "mean"))
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    config = {
        "seed": args.seed,
        "tensor_root": str(tensor_root),
        "source_runs": {scale: str(path) for scale, path in run_dirs.items()},
        "train_sizes": {scale: len(splits[scale]["train"]) for scale in SCALES},
        "validation_size": len(splits["025"]["validation"]),
        "test_size": len(splits["025"]["test"]),
        "exemplar_size": len(splits["025"]["exemplar"]),
        "feature_set": "graph (no synthesis context)",
        "svr": {"C": 10.0, "epsilon": 0.05, "gamma": "scale"},
        "hurdle_targets": [LABEL_KEYS[index] for index in HURDLE_TARGETS],
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
