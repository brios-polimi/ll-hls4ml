#!/usr/bin/env python3
"""CPU diagnostics matched to a persisted hierarchical-model split.

The baseline deliberately sees deterministic graph summaries, not adjacency.
It therefore tests how much of a hierarchical GNN result can be recovered from
node/relation counts, opcode and pragma histograms, pooled literal/type values,
and the same causal synthesis context used by the neural model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ll-hls4ml-matplotlib")
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ll_hls4ml.data.dataset import HeteroGraphDataset
from ll_hls4ml.data.vocab import load_vocab
from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.training.targets import wahls4ml_metrics_raw
from preliminary_benchmark import FAMILIES, _tensor_features


HURDLE_TARGETS = (2, 3)
SYNTHETIC_FAMILIES = tuple(family for family in FAMILIES if family != "exemplar")
ARCHIVE_RE = re.compile(r"(?:^|/)archive_(\d+)(?:/|$)")


def archive_number(path: str) -> int:
    match = ARCHIVE_RE.search(path)
    if match is None:
        raise ValueError(f"No archive component in {path}")
    return int(match.group(1))


def git_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout)
    return {"commit": commit, "dirty": dirty}


def load_manifest(run_dir: Path) -> dict[str, list[dict]]:
    path = run_dir / "split_manifest.json"
    manifest = json.loads(path.read_text())
    expected = {"train", "validation", "test", "exemplar"}
    if set(manifest) != expected:
        raise ValueError(f"Unexpected split keys: {sorted(manifest)}")
    paths = [item["tensor_path"] for values in manifest.values() for item in values]
    if len(paths) != len(set(paths)):
        raise ValueError("Split manifest contains duplicate tensor paths")
    return manifest


def feature_frame(
    dataset: HeteroGraphDataset,
    needed_paths: set[str],
    instruction_vocab_size: int,
    cache_path: Path,
) -> pd.DataFrame:
    fingerprint = hashlib.sha256("\n".join(sorted(needed_paths)).encode()).hexdigest()
    fingerprint_path = cache_path.with_suffix(".sha256")
    if cache_path.exists() and fingerprint_path.exists():
        if fingerprint_path.read_text().strip() == fingerprint:
            frame = pd.read_pickle(cache_path)
            if set(frame.tensor_path) == needed_paths:
                print(f"Using cached features: {cache_path}", flush=True)
                return frame

    rows = []
    for position, (index, path) in enumerate(zip(range(len(dataset)), dataset.paths), start=1):
        relative = path.relative_to(dataset.root).as_posix()
        if relative not in needed_paths:
            continue
        data = dataset[index]
        row = _tensor_features(data, instruction_vocab_size)
        row.update(
            dataset_index=index,
            kernel_family=dataset.type_of(index),
            tensor_path=relative,
            archive=archive_number(relative),
        )
        for target_index, label in enumerate(LABEL_KEYS):
            row[label] = float(data.y[target_index])
        rows.append(row)
        if len(rows) % 100 == 0 or len(rows) == len(needed_paths):
            print(f"Tabular features: {len(rows)}/{len(needed_paths)}", flush=True)
    frame = pd.DataFrame(rows)
    if set(frame.tensor_path) != needed_paths:
        missing = sorted(needed_paths - set(frame.tensor_path))[:5]
        raise ValueError(f"Tensor paths missing from dataset, first five: {missing}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(cache_path)
    fingerprint_path.write_text(fingerprint + "\n")
    return frame


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    metadata = {"dataset_index", "kernel_family", "tensor_path", "archive", *LABEL_KEYS}
    all_features = [column for column in frame if column not in metadata]
    graph = [column for column in all_features if not column.startswith("context_")]
    core_context = [
        column for column in all_features
        if not column.startswith(("context_categorical_2_", "context_categorical_3_"))
    ]
    size = [
        column for column in graph
        if column in {"total_nodes", "log_total_nodes", "total_edges", "log_total_edges"}
        or (
            (column.endswith("_count") or column.endswith("_ratio"))
            and not column.startswith(("instruction_id_", "pragma_id_", "edge_"))
        )
    ]
    return {
        "core_context": core_context,
        "graph": graph,
        "graph_no_edge_summaries": [
            column for column in graph
            if not column.startswith("edge_") and column not in {"total_edges", "log_total_edges"}
        ],
        "graph_opcodes_size": [
            column for column in graph
            if column in size or column.startswith("instruction_id_")
        ],
        "graph_size_only": size,
    }


def arrays(
    indexed: pd.DataFrame,
    paths: list[str],
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x = indexed.loc[paths, columns].to_numpy(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = indexed.loc[paths, LABEL_KEYS].to_numpy(np.float32)
    return x, y


class HurdleRegressor:
    def __init__(self, kind: str, seed: int):
        self.kind = kind
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> "HurdleRegressor":
        self.x_scaler = StandardScaler().fit(x) if self.kind == "rbf" else None
        fit_x = self.x_scaler.transform(x) if self.x_scaler is not None else x
        self.y_scaler = StandardScaler().fit(np.log1p(y))
        scaled_y = self.y_scaler.transform(np.log1p(y))
        self.regressors = []
        self.classifiers = {}
        for target_index in range(len(LABEL_KEYS)):
            positive = y[:, target_index] > 0
            rows = positive if target_index in HURDLE_TARGETS else np.ones(len(y), dtype=bool)
            if self.kind == "rbf":
                regressor = SVR(C=10.0, epsilon=0.05, gamma="scale")
            elif self.kind == "extra_trees":
                regressor = ExtraTreesRegressor(
                    n_estimators=300, min_samples_leaf=2, max_features=0.8,
                    n_jobs=-1, random_state=self.seed + target_index,
                )
            else:
                raise ValueError(f"Unknown model kind {self.kind}")
            regressor.fit(fit_x[rows], scaled_y[rows, target_index])
            self.regressors.append(regressor)
            if target_index in HURDLE_TARGETS:
                if self.kind == "rbf":
                    classifier = LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=self.seed,
                    )
                else:
                    classifier = ExtraTreesClassifier(
                        n_estimators=300, min_samples_leaf=2, max_features=0.8,
                        class_weight="balanced", n_jobs=-1,
                        random_state=self.seed + 100 + target_index,
                    )
                classifier.fit(fit_x, positive)
                self.classifiers[target_index] = classifier
        return self

    def predict_modes(self, x: np.ndarray) -> dict[str, np.ndarray]:
        fit_x = self.x_scaler.transform(x) if self.x_scaler is not None else x
        scaled = np.column_stack([model.predict(fit_x) for model in self.regressors])
        positive = np.expm1(self.y_scaler.inverse_transform(scaled))
        probabilities = {
            index: model.predict_proba(fit_x)[:, 1]
            for index, model in self.classifiers.items()
        }
        predictions = {}
        for mode in ("expected", "threshold"):
            values = positive.copy()
            for target_index, probability in probabilities.items():
                multiplier = probability if mode == "expected" else probability >= 0.5
                values[:, target_index] *= multiplier
            predictions[mode] = np.clip(values, 0, None).astype(np.float32)
        return predictions


def macro_smape(prediction: np.ndarray, target: np.ndarray) -> float:
    metrics = wahls4ml_metrics_raw(torch.from_numpy(prediction), torch.from_numpy(target))
    return float(metrics["smape"].mean())


def choose_mode(model: HurdleRegressor, x: np.ndarray, y: np.ndarray) -> str:
    scores = {mode: macro_smape(prediction, y) for mode, prediction in model.predict_modes(x).items()}
    mode = min(scores, key=scores.get)
    print(f"Validation hurdle modes: {scores}; selected {mode}", flush=True)
    return mode


def metric_rows(
    experiment: str,
    model: str,
    feature_set: str,
    train_size: int,
    split: str,
    families: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
) -> list[dict]:
    rows = []
    for family in ("all", *sorted(set(families))):
        mask = np.ones(len(families), dtype=bool) if family == "all" else families == family
        result = wahls4ml_metrics_raw(
            torch.from_numpy(prediction[mask]), torch.from_numpy(target[mask])
        )
        for target_index, label in enumerate(LABEL_KEYS):
            rows.append({
                "experiment": experiment,
                "model": model,
                "feature_set": feature_set,
                "train_size": train_size,
                "split": split,
                "kernel_family": family,
                "target": label,
                "smape": float(result["smape"][target_index]),
                "r2": float(result["r2"][target_index]),
                "rmse": float(result["rmse"][target_index]),
                "n_samples": int(mask.sum()),
            })
    return rows


def bootstrap_delta(
    baseline: np.ndarray,
    neural: np.ndarray,
    target: np.ndarray,
    seed: int,
    replicates: int = 10_000,
) -> tuple[float, float, float, float]:
    # Match wahls4ml_metrics_raw exactly. The +1 stabilizer materially affects
    # the hurdle targets when either prediction or target is zero.
    sample_a = np.mean(
        200 * np.abs(baseline - target) / (np.abs(baseline) + np.abs(target) + 1.0),
        axis=1,
    )
    sample_b = np.mean(
        200 * np.abs(neural - target) / (np.abs(neural) + np.abs(target) + 1.0),
        axis=1,
    )
    delta = sample_a - sample_b
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for start in range(0, replicates, 1000):
        stop = min(start + 1000, replicates)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        means[start:stop] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(delta.mean()), float(low), float(high), float(np.mean(delta < 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default="artifacts/results/hierarchical_bottom_up_v3_seed42")
    parser.add_argument("--tensor-dir", default="../data/tensors")
    parser.add_argument("--output-dir", default="artifacts/results/hierarchical_bottom_up_v3_cpu_baselines_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-lofo", action="store_true")
    args = parser.parse_args()

    source_run = (REPO_ROOT / args.source_run).resolve()
    tensor_root = (REPO_ROOT / args.tensor_dir).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(source_run)
    needed_paths = {
        item["tensor_path"] for values in manifest.values() for item in values
    }
    dataset = HeteroGraphDataset(tensor_root, types=FAMILIES, silent=False)
    vocabulary, _, _ = load_vocab(tensor_root / "vocab.json")
    frame = feature_frame(
        dataset, needed_paths, len(vocabulary), output_dir / "tabular_features.pkl"
    )
    indexed = frame.set_index("tensor_path")
    sets = feature_sets(frame)
    split_paths = {
        name: [item["tensor_path"] for item in values]
        for name, values in manifest.items()
    }
    train_scales = {
        "archives_1_4": [path for path in split_paths["train"] if archive_number(path) <= 4],
        "archives_1_8": [path for path in split_paths["train"] if archive_number(path) <= 8],
        "archives_1_15": split_paths["train"],
    }
    run_specs = [
        (name, "rbf", "core_context", paths, None)
        for name, paths in train_scales.items()
    ]
    run_specs.extend([
        ("full_graph_only", "rbf", "graph", train_scales["archives_1_15"], None),
        ("full_no_edge_summaries", "rbf", "graph_no_edge_summaries", train_scales["archives_1_15"], None),
        ("full_opcodes_size", "rbf", "graph_opcodes_size", train_scales["archives_1_15"], None),
        ("full_size_only", "rbf", "graph_size_only", train_scales["archives_1_15"], None),
        ("full_extra_trees", "extra_trees", "core_context", train_scales["archives_1_15"], None),
    ])
    if not args.skip_lofo:
        for family in SYNTHETIC_FAMILIES:
            paths = [
                path for path in train_scales["archives_1_15"]
                if indexed.loc[path, "kernel_family"] != family
            ]
            run_specs.append((f"lofo_{family}", "rbf", "core_context", paths, family))

    metrics = []
    prediction_rows = []
    timings = []
    neural = pd.read_csv(source_run / "predictions.csv").set_index(["split", "tensor_path"])
    comparisons = []
    for experiment, kind, set_name, train_paths, heldout_family in run_specs:
        print(f"\n=== {experiment}: {kind}, {set_name}, n={len(train_paths)} ===", flush=True)
        columns = sets[set_name]
        train_x, train_y = arrays(indexed, train_paths, columns)
        validation_paths = split_paths["validation"]
        if heldout_family is not None:
            validation_paths = [
                path for path in validation_paths
                if indexed.loc[path, "kernel_family"] != heldout_family
            ]
        validation_x, validation_y = arrays(indexed, validation_paths, columns)
        started = time.perf_counter()
        model = HurdleRegressor(kind, args.seed).fit(train_x, train_y)
        fit_seconds = time.perf_counter() - started
        mode = choose_mode(model, validation_x, validation_y)
        timings.append({
            "experiment": experiment, "model": kind, "feature_set": set_name,
            "train_size": len(train_paths), "fit_seconds": fit_seconds,
            "selected_hurdle_mode": mode,
        })
        evaluation_splits = ["validation", "test", "exemplar"]
        for split in evaluation_splits:
            paths = split_paths[split]
            if heldout_family is not None:
                if split != "test":
                    continue
                paths = [path for path in paths if indexed.loc[path, "kernel_family"] == heldout_family]
            x, target = arrays(indexed, paths, columns)
            prediction = model.predict_modes(x)[mode]
            families = indexed.loc[paths, "kernel_family"].to_numpy(str)
            metrics.extend(metric_rows(
                experiment, kind, set_name, len(train_paths), split,
                families, prediction, target,
            ))
            for row_index, path in enumerate(paths):
                row = {
                    "experiment": experiment, "model": kind,
                    "feature_set": set_name, "train_size": len(train_paths),
                    "selected_hurdle_mode": mode, "split": split,
                    "tensor_path": path,
                    "kernel_family": families[row_index],
                }
                for target_index, label in enumerate(LABEL_KEYS):
                    row[f"target_{label}"] = float(target[row_index, target_index])
                    row[f"prediction_{label}"] = float(prediction[row_index, target_index])
                prediction_rows.append(row)
            if heldout_family is None and split in {"test", "exemplar"}:
                neural_rows = neural.loc[[(split, path) for path in paths]]
                neural_prediction = neural_rows[[f"prediction_{label}" for label in LABEL_KEYS]].to_numpy(np.float32)
                delta, low, high, win_fraction = bootstrap_delta(
                    prediction, neural_prediction, target, args.seed
                )
                comparisons.append({
                    "experiment": experiment, "split": split,
                    "cpu_minus_h0_smape": delta,
                    "ci95_low": low, "ci95_high": high,
                    "cpu_sample_win_fraction": win_fraction,
                    "n_samples": len(paths),
                })

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(output_dir / "metrics.csv", index=False)
    prediction_frame = pd.DataFrame(prediction_rows)
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(timings).to_csv(output_dir / "timings.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output_dir / "paired_h0_comparisons.csv", index=False)
    summary = (
        metric_frame[metric_frame.kernel_family == "all"]
        .groupby(["experiment", "model", "feature_set", "train_size", "split"], as_index=False)
        .agg(macro_smape=("smape", "mean"), macro_r2=("r2", "mean"))
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    config = {
        "seed": args.seed,
        "source_run": str(source_run),
        "tensor_root": str(tensor_root),
        "source_split_sha256": json.loads((source_run / "resolved_config.json").read_text()).get("split_sha256"),
        "git": git_state(),
        "train_scales": {name: len(paths) for name, paths in train_scales.items()},
        "validation_size": len(split_paths["validation"]),
        "test_size": len(split_paths["test"]),
        "exemplar_size": len(split_paths["exemplar"]),
        "feature_interpretation": "deterministic graph summaries; no adjacency or message passing",
        "rbf": {"C": 10.0, "epsilon": 0.05, "gamma": "scale"},
        "extra_trees": {"n_estimators": 300, "min_samples_leaf": 2, "max_features": 0.8},
        "hurdle_targets": [LABEL_KEYS[index] for index in HURDLE_TARGETS],
        "hurdle_mode_selection": "lowest validation macro SMAPE",
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("\n" + summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
