#!/usr/bin/env python3
"""Run a frozen five-family preliminary wa-hls4ml-style benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import torch
from torch.utils.data import Dataset, Subset

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.data.dataset import HeteroGraphDataset
from ll_hls4ml.data.vocab import load_vocab
from ll_hls4ml.io.load_json import load_graph_json
from ll_hls4ml.io.schema import (
    EDGE_TYPES,
    LABEL_KEYS,
    NODE_PRAGMA,
    NODE_TYPES,
    PRAGMA_VOCAB,
    PRAGMA_VOCAB_SIZE,
)
from ll_hls4ml.models.registry import build
from ll_hls4ml.training.distributed import unwrap_model
from ll_hls4ml.training.loaders import make_loader
from ll_hls4ml.training.loops import fit
from ll_hls4ml.training.targets import (
    compute_target_z_stats,
    denormalize_target,
    relative_percentage_error,
    wahls4ml_metrics_raw,
)
from ll_hls4ml.viz.training import prediction_scatter_plots, rpe_box_plots


FAMILIES = ["2layer", "3layer", "conv1d", "conv2d", "dense_resource", "dense_latency", "rule4ml", "exemplar"]
SYNTHETIC_FAMILIES = ["2layer", "3layer", "conv1d", "conv2d", "dense_resource", "dense_latency", "rule4ml"]
DISPLAY_LABELS = ["LUT", "FF", "DSP", "BRAM", "Cycles", "II"]
PAPER_ORDER = ["BRAM", "DSP", "FF", "LUT", "Cycles", "II"]
PRAGMA_EDGE_TYPES = [
    ("pragma", "applies_to", "instruction"),
    ("pragma", "applies_to", "variable"),
]


@dataclass
class Evaluation:
    model: str
    split: str
    predictions: np.ndarray
    targets: np.ndarray
    inference_seconds_per_sample: float


class PragmaAblationDataset(Dataset):
    """View of a dataset with pragma features and edges removed."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index].clone()
        data["pragma"].x.zero_()
        for edge_type in PRAGMA_EDGE_TYPES:
            data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
        return data


def _git_state(repository: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    return {"commit": commit, "dirty": dirty}


def _snapshot_id(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(
            f"{path.relative_to(root)}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()[:16]


def _split_indices(dataset: HeteroGraphDataset, seed: int):
    synthetic = [
        index for index in range(len(dataset))
        if dataset.type_of(index) in SYNTHETIC_FAMILIES
    ]
    exemplar = [
        index for index in range(len(dataset))
        if dataset.type_of(index) == "exemplar"
    ]
    families = [dataset.type_of(index) for index in synthetic]
    train_indices, holdout_indices = train_test_split(
        synthetic,
        test_size=0.30,
        random_state=seed,
        stratify=families,
    )
    holdout_families = [dataset.type_of(index) for index in holdout_indices]
    val_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=0.50,
        random_state=seed,
        stratify=holdout_families,
    )
    return {
        "train": sorted(train_indices),
        "validation": sorted(val_indices),
        "synthetic_test": sorted(test_indices),
        "exemplar": sorted(exemplar),
    }


def _split_manifest(dataset, splits, tensor_root: Path) -> dict:
    return {
        name: [
            {
                "tensor_path": str(dataset.paths[index].relative_to(tensor_root)),
                "kernel_family": dataset.type_of(index),
            }
            for index in indices
        ]
        for name, indices in splits.items()
    }


def _tensor_features(data, instruction_vocab_size: int) -> dict[str, float]:
    features: dict[str, float] = {}
    total_nodes = sum(int(data[node_type].num_nodes) for node_type in NODE_TYPES)
    features["total_nodes"] = total_nodes
    features["log_total_nodes"] = np.log1p(total_nodes)

    for node_type in NODE_TYPES:
        count = int(data[node_type].num_nodes)
        features[f"{node_type}_count"] = count
        features[f"{node_type}_ratio"] = count / total_nodes if total_nodes else 0.0

    instruction = data["instruction"].x.view(-1).numpy()
    instruction_histogram = np.bincount(
        instruction,
        minlength=instruction_vocab_size,
    )[:instruction_vocab_size]
    instruction_denominator = max(len(instruction), 1)
    for index, count in enumerate(instruction_histogram):
        features[f"instruction_id_{index}_ratio"] = count / instruction_denominator

    pragma = data["pragma"].x[:, 0].long().numpy()
    pragma_histogram = np.bincount(
        pragma, minlength=PRAGMA_VOCAB_SIZE
    )[:PRAGMA_VOCAB_SIZE]
    pragma_denominator = max(len(pragma), 1)
    for index, count in enumerate(pragma_histogram):
        features[f"pragma_id_{index}_ratio"] = count / pragma_denominator

    for node_type in ("variable", "constant"):
        values = data[node_type].x.float().numpy()
        for index in range(values.shape[1]):
            if values.shape[0]:
                features[f"{node_type}_feature_{index}_mean"] = float(
                    values[:, index].mean()
                )
                features[f"{node_type}_feature_{index}_max"] = float(
                    values[:, index].max()
                )
            else:
                features[f"{node_type}_feature_{index}_mean"] = 0.0
                features[f"{node_type}_feature_{index}_max"] = 0.0

    total_edges = 0
    for edge_type in EDGE_TYPES:
        count = int(data[edge_type].edge_index.shape[1])
        key = "__".join(edge_type)
        features[f"edge_{key}_count"] = count
        total_edges += count
    features["total_edges"] = total_edges
    features["log_total_edges"] = np.log1p(total_edges)
    for edge_type in EDGE_TYPES:
        key = "__".join(edge_type)
        features[f"edge_{key}_ratio"] = (
            features[f"edge_{key}_count"] / total_edges if total_edges else 0.0
        )
    return features


def _build_tabular_frame(
    dataset: HeteroGraphDataset,
    instruction_vocab_size: int,
) -> pd.DataFrame:
    rows = []
    for index, path in enumerate(dataset.paths):
        data = dataset[index]
        row = _tensor_features(data, instruction_vocab_size)
        row.update(
            {
                "dataset_index": index,
                "kernel_family": dataset.type_of(index),
                "tensor_path": str(path),
            }
        )
        for label_index, label in enumerate(LABEL_KEYS):
            row[label] = float(data.y[label_index])
        rows.append(row)
        if (index + 1) % 50 == 0:
            print(f"Tabular features: {index + 1}/{len(dataset)}", flush=True)
    return pd.DataFrame(rows)


def _raw_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    result = wahls4ml_metrics_raw(
        torch.from_numpy(np.array(predictions, copy=True)),
        torch.from_numpy(np.array(targets, copy=True)),
    )
    return {key: value.numpy() for key, value in result.items()}


def _evaluate_classical(
    frame: pd.DataFrame,
    splits: dict[str, list[int]],
    seed: int,
) -> list[Evaluation]:
    metadata = {"dataset_index", "kernel_family", "tensor_path", *LABEL_KEYS}
    feature_columns = [column for column in frame.columns if column not in metadata]
    indexed = frame.set_index("dataset_index")
    train_x = indexed.loc[splits["train"], feature_columns].to_numpy(np.float32)
    train_x = np.nan_to_num(train_x, nan=0.0, posinf=0.0, neginf=0.0)
    train_y = indexed.loc[splits["train"], LABEL_KEYS].to_numpy(np.float32)
    log_train_y = np.log1p(train_y)
    target_scaler = StandardScaler().fit(log_train_y)
    standardized_y = target_scaler.transform(log_train_y)

    models = {
        "median": None,
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=seed,
        ),
        "rbf_svr": make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(
                SVR(C=10.0, epsilon=0.05, gamma="scale"),
                n_jobs=-1,
            ),
        ),
    }
    median = np.median(train_y, axis=0, keepdims=True)
    evaluations: list[Evaluation] = []
    for name, model in models.items():
        print(f"Fitting {name}...", flush=True)
        if model is not None:
            model.fit(train_x, standardized_y)
        for split_name in ("synthetic_test", "exemplar"):
            indices = splits[split_name]
            x = indexed.loc[indices, feature_columns].to_numpy(np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            targets = indexed.loc[indices, LABEL_KEYS].to_numpy(np.float32)
            start = time.perf_counter()
            if model is None:
                predictions = np.repeat(median, len(indices), axis=0)
            else:
                prediction_scaled = model.predict(x)
                predictions = np.expm1(
                    target_scaler.inverse_transform(prediction_scaled)
                )
            elapsed = time.perf_counter() - start
            evaluations.append(
                Evaluation(
                    model=name,
                    split=split_name,
                    predictions=np.clip(predictions, 0, None).astype(np.float32),
                    targets=targets,
                    inference_seconds_per_sample=elapsed / len(indices),
                )
            )
    return evaluations


def _predict_neural(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    base_model = unwrap_model(model)
    predictions = []
    targets = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = denormalize_target(
                model(batch),
                base_model.y_means,
                base_model.y_stds,
            )
            predictions.append(prediction.cpu())
            targets.append(batch.y.view(-1, len(LABEL_KEYS)).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    prediction_array = torch.cat(predictions).numpy()
    target_array = torch.cat(targets).numpy()
    return (
        np.clip(prediction_array, 0, None),
        target_array,
        elapsed / len(target_array),
    )


def _evaluate_neural(
    dataset: HeteroGraphDataset,
    splits: dict[str, list[int]],
    model_name: str,
    output_name: str,
    instruction_vocab_size: int,
    max_position: int,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    without_pragmas: bool,
) -> list[Evaluation]:
    train_dataset = Subset(dataset, splits["train"])
    validation_dataset = Subset(dataset, splits["validation"])
    test_datasets = {
        "synthetic_test": Subset(dataset, splits["synthetic_test"]),
        "exemplar": Subset(dataset, splits["exemplar"]),
    }
    if without_pragmas:
        train_dataset = PragmaAblationDataset(train_dataset)
        validation_dataset = PragmaAblationDataset(validation_dataset)
        test_datasets = {
            name: PragmaAblationDataset(value)
            for name, value in test_datasets.items()
        }

    y_means, y_stds = compute_target_z_stats(train_dataset)
    model_kwargs = {
        "instruction_vocab_size": instruction_vocab_size,
        "y_means": y_means,
        "y_stds": y_stds,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.15,
        "pool": "mean",
    }
    if model_name == "rgcn":
        model_kwargs.update(
            {"edge_pos_vocab_size": max_position, "aggr": "mean"}
        )
    else:
        model_kwargs.update(
            {"num_var_embed_layers": 2, "node_aggr": "concat"}
        )
    model = build(model_name, **model_kwargs)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"Training {output_name} on {device}...", flush=True)
    model = fit(
        model,
        train_loader,
        validation_loader,
        epochs=epochs,
        criterion=torch.nn.HuberLoss(),
        optimizer=optimizer,
        scheduler=None,
        device=device,
        patience=patience,
        mode="min",
        restore_best_weights=True,
        verbose=1,
        experiment_name=output_name,
        checkpoint_dir=output_dir / "checkpoints",
    )

    evaluations = []
    for split_name, split_dataset in test_datasets.items():
        loader = make_loader(
            split_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        predictions, targets, latency = _predict_neural(model, loader, device)
        evaluations.append(
            Evaluation(
                model=output_name,
                split=split_name,
                predictions=predictions,
                targets=targets,
                inference_seconds_per_sample=latency,
            )
        )
    unwrap_model(model).cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return evaluations


def _metrics_frame(evaluations: list[Evaluation]) -> pd.DataFrame:
    rows = []
    for evaluation in evaluations:
        metrics = _raw_metrics(evaluation.predictions, evaluation.targets)
        rpe = relative_percentage_error(
            torch.from_numpy(np.array(evaluation.predictions, copy=True)),
            torch.from_numpy(np.array(evaluation.targets, copy=True)),
        ).numpy()
        for index, label in enumerate(LABEL_KEYS):
            rows.append(
                {
                    "model": evaluation.model,
                    "split": evaluation.split,
                    "target": label,
                    "r2": float(metrics["r2"][index]),
                    "smape": float(metrics["smape"][index]),
                    "rmse": float(metrics["rmse"][index]),
                    "rpe_median": float(np.median(rpe[:, index])),
                    "rpe_mean": float(np.mean(rpe[:, index])),
                    "rpe_q25": float(np.percentile(rpe[:, index], 25)),
                    "rpe_q75": float(np.percentile(rpe[:, index], 75)),
                    "inference_seconds_per_sample": evaluation.inference_seconds_per_sample,
                    "n_samples": len(evaluation.targets),
                }
            )
    return pd.DataFrame(rows)


def _save_predictions(
    evaluations: list[Evaluation],
    dataset: HeteroGraphDataset,
    splits: dict[str, list[int]],
    output_dir: Path,
) -> None:
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for evaluation in evaluations:
        indices = splits[evaluation.split]
        rows = []
        for row_index, dataset_index in enumerate(indices):
            row = {
                "dataset_index": dataset_index,
                "kernel_family": dataset.type_of(dataset_index),
                "tensor_path": str(dataset.paths[dataset_index]),
            }
            for target_index, label in enumerate(LABEL_KEYS):
                row[f"target_{label}"] = evaluation.targets[row_index, target_index]
                row[f"prediction_{label}"] = evaluation.predictions[row_index, target_index]
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            predictions_dir / f"{evaluation.model}__{evaluation.split}.csv",
            index=False,
        )


def _save_plots(evaluations: list[Evaluation], output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for evaluation in evaluations:
        title = f"{evaluation.model}: {evaluation.split.replace('_', ' ')}"
        figure, _ = rpe_box_plots(
            evaluation.predictions,
            evaluation.targets,
            DISPLAY_LABELS,
            ordering=PAPER_ORDER,
            title=title,
            show=False,
        )
        figure.savefig(
            figures_dir / f"{evaluation.model}__{evaluation.split}__rpe.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)
        figure, _ = prediction_scatter_plots(
            evaluation.predictions,
            evaluation.targets,
            DISPLAY_LABELS,
            title=title,
            show=False,
        )
        figure.savefig(
            figures_dir / f"{evaluation.model}__{evaluation.split}__scatter.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)


def _pragma_audit(graph_root: Path, dataset: HeteroGraphDataset) -> dict:
    directives = Counter()
    anchors = Counter()
    injector = Counter()
    graphs_without_pragmas = 0
    graphs_checked = 0
    for tensor_path in dataset.paths:
        graph_path = graph_root / tensor_path.relative_to(dataset.root).with_suffix(".json")
        graph = load_graph_json(graph_path)
        pragmas = [
            node for node in graph.get("nodes", [])
            if int(node.get("type", -1)) == NODE_PRAGMA
        ]
        graphs_checked += 1
        if not pragmas:
            graphs_without_pragmas += 1
        for node in pragmas:
            directives[node.get("text", "unknown")] += 1
            features = node.get("features", {})
            anchors.update(features.get("anchor_reason", []))
            injector.update(features.get("injector", []))
    return {
        "graphs_checked": graphs_checked,
        "graphs_without_pragmas": graphs_without_pragmas,
        "directive_counts": dict(directives.most_common()),
        "anchor_reason_counts": dict(anchors.most_common()),
        "injector_counts": dict(injector.most_common()),
        "known_tensor_pragma_vocab": PRAGMA_VOCAB,
    }


def _write_report(
    output_dir: Path,
    config: dict,
    manifest: dict,
    metrics: pd.DataFrame,
    pragma_audit: dict,
) -> None:
    family_counts = {
        split: dict(Counter(item["kernel_family"] for item in items))
        for split, items in manifest.items()
    }
    synthetic = metrics[metrics["split"] == "synthetic_test"]
    exemplar = metrics[metrics["split"] == "exemplar"]

    def summary_table(frame: pd.DataFrame) -> str:
        summary = (
            frame.groupby("model")
            .agg(
                macro_r2=("r2", "mean"),
                macro_smape=("smape", "mean"),
                median_abs_rpe=("rpe_median", lambda values: np.median(np.abs(values))),
            )
            .sort_values("macro_smape")
        )
        columns = list(summary.columns)
        lines = [
            "| model | " + " | ".join(columns) + " |",
            "| --- | " + " | ".join("---:" for _ in columns) + " |",
        ]
        for model, row in summary.iterrows():
            values = " | ".join(f"{float(row[column]):.3f}" for column in columns)
            lines.append(f"| {model} | {values} |")
        return "\n".join(lines)

    def smape_table(frame: pd.DataFrame) -> str:
        pivot = frame.pivot(index="model", columns="target", values="smape")
        pivot["macro"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("macro")
        columns = [*LABEL_KEYS, "macro"]
        lines = [
            "| model | " + " | ".join(columns) + " |",
            "| --- | " + " | ".join("---:" for _ in columns) + " |",
        ]
        for model, row in pivot.iterrows():
            values = " | ".join(f"{float(row[column]):.2f}" for column in columns)
            lines.append(f"| {model} | {values} |")
        return "\n".join(lines)

    def macro_smape(model: str, split: str) -> float:
        rows = metrics[(metrics["model"] == model) & (metrics["split"] == split)]
        return float(rows["smape"].mean())

    available_models = set(metrics["model"])

    def pragma_delta(model: str, ablation: str) -> str:
        if {model, ablation} <= available_models:
            delta = (
                macro_smape(model, "synthetic_test")
                - macro_smape(ablation, "synthetic_test")
            )
            return f"{delta:+.2f}"
        return "not run"

    mlp_pragma_delta = pragma_delta("pooled_mlp", "pooled_mlp_no_pragmas")
    rgcn_pragma_delta = pragma_delta("rgcn", "rgcn_no_pragmas")
    best_synthetic = (
        synthetic.groupby("model")["smape"].mean().sort_values().index[0]
    )
    best_exemplar = (
        exemplar.groupby("model")["smape"].mean().sort_values().index[0]
    )

    unknown_directives = {
        directive: count
        for directive, count in pragma_audit["directive_counts"].items()
        if directive not in PRAGMA_VOCAB
    }
    report = f"""# Preliminary five-family benchmark

Generated: {time.strftime("%Y-%m-%d %H:%M:%S %Z")}

## Scope

This is an engineering benchmark of the current tensor snapshot, not a final
comparison with wa-hls4ml. The graph compiler, static-initializer cleanup, type
encoding, and pragma injection are all research variables that may change.

- Tensor snapshot: `{config["tensor_snapshot_id"]}`
- Families present: {", ".join(FAMILIES)}
- Split counts: `{json.dumps(family_counts, sort_keys=True)}`
- Seed: {config["seed"]}
- Device: {config["device"]}
- ll-hls4ml state: `{json.dumps(config["ll_hls4ml_git"])}` 
- hls4ml_pipeline state: `{json.dumps(config["pipeline_git"])}` 

The synthetic test contains held-out samples from 2layer, 3layer, Conv1D, and
Conv2D. Exemplar is never used for training or validation and is treated as an
external inductive set, matching the paper's distinction in spirit. These are
locally generated splits, not the benchmark's official sample IDs.

## Synthetic test summary

{summary_table(synthetic)}

## Exemplar summary

{summary_table(exemplar)}

## Main findings

- `{best_synthetic}` is the strongest current synthetic-test baseline. The
  simple classical models outperform both neural architectures, so further
  model scaling is not justified yet.
- Every model fails on exemplar (`{best_exemplar}` is merely the least bad).
  This is evidence of severe family/domain shift, not a useful wa-hls4ml
  headline comparison.
- Adding pragmas changes synthetic macro SMAPE by {mlp_pragma_delta}
  points for the pooled MLP and {rgcn_pragma_delta} points for the R-GCN
  relative to their no-pragma ablations. Negative means pragmas help. The
  synthetic effect is modest and does not carry consistently to exemplar, so
  one seed cannot establish that the injected representation generalizes.
- The correct next investment is data/schema validation and group-aware
  evaluation, followed by coverage of the missing kernel families. It is not a
  larger GNN.

### Synthetic-test SMAPE by target

{smape_table(synthetic)}

### Exemplar SMAPE by target

{smape_table(exemplar)}

Full per-target R², SMAPE, RMSE, RPE quartiles, sample counts, and inference
timings are in `metrics.csv`. Per-sample predictions are in `predictions/`.
Paper-style RPE box plots and log-log prediction scatter plots are in `figures/`.

## Pragma audit

- Graphs checked: {pragma_audit["graphs_checked"]}
- Graphs without injected pragma nodes: {pragma_audit["graphs_without_pragmas"]}
- Injection anchors: `{json.dumps(pragma_audit["anchor_reason_counts"])}`
- Directives not represented distinctly by the current tensor vocabulary:
  `{json.dumps(unknown_directives)}`

The MLP consumes pragma IDs through a pooled pragma embedding. The R-GCN uses the
same embedding, sends pragma-node messages to anchored instruction/variable
nodes, and also pools the resulting pragma representation. The `*_no_pragmas`
runs zero pragma features and remove pragma edges; their difference from the
normal runs is the first empirical check of whether pragmas currently help.

This does not establish that injection is semantically correct. In particular,
function-entry fallback anchors are coarse, and compiler-carrier and diagnostic
records may overlap. Every directive observed in this snapshot now has a
distinct tensor ID; future unknown directives will still collapse to UNK.

## Interpretation rules

- Prefer per-target results over macro averages; RMSE is scale-dependent.
- RPE follows the paper: `(target - prediction) / (target + 1) * 100`.
  Positive values mean underprediction and negative values mean overprediction.
- DSP and BRAM contain genuine zeros, so RPE and SMAPE use the paper's `+1`
  denominator convention.
- Exemplar performance is the more relevant generalization warning, but the
  present sample is small and the training families are incomplete.
- Do not compare these numbers directly with the published headline results
  until official split membership, all intended families, graph provenance, and
  target definitions are aligned.

## Data-retention consequence

Graph JSON, pragma dumps, and manifests are the durable research artifacts.
Tensors are cheap derived files: rebuild an affected archive when the feature
schema changes rather than adding loader compatibility for old tensor layouts.
Keep one small representative source archive/project set as an end-to-end
compiler/injection canary. Process large missing families one archive at a time;
after graph/tensor counts and failures are verified, remove extracted projects.
Keep the compressed tarball only when its redownload cost is worth the disk
space. Never purge automatically after a partial compilation failure.
"""
    (output_dir / "REPORT.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-dir", default="../data/tensors")
    parser.add_argument("--graph-dir", default="../data/graphs")
    parser.add_argument("--vocab", default="artifacts/vocab/default.json")
    parser.add_argument("--output-dir", default="artifacts/results/preliminary-five-family")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument(
        "--neural-models",
        nargs="+",
        choices=("pooled_mlp", "pooled_mlp_no_pragmas", "rgcn", "rgcn_no_pragmas"),
        default=("pooled_mlp", "pooled_mlp_no_pragmas", "rgcn", "rgcn_no_pragmas"),
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ll-hls4ml-matplotlib")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tensor_root = Path(args.tensor_dir).resolve()
    graph_root = Path(args.graph_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = HeteroGraphDataset(tensor_root, types=FAMILIES, silent=False)
    vocabulary, max_position, _ = load_vocab(Path(args.vocab))
    if max(int(dataset[index]["instruction"].x.max()) for index in range(len(dataset))) >= len(vocabulary):
        raise ValueError("Tensor instruction IDs exceed the selected vocabulary")

    splits = _split_indices(dataset, args.seed)
    manifest = _split_manifest(dataset, splits, tensor_root)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "families": FAMILIES,
        "synthetic_families": SYNTHETIC_FAMILIES,
        "tensor_root": str(tensor_root),
        "graph_root": str(graph_root),
        "tensor_snapshot_id": _snapshot_id(dataset.paths, tensor_root),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "ll_hls4ml_git": _git_state(_REPO_ROOT),
        "pipeline_git": _git_state(_REPO_ROOT.parent / "hls4ml_pipeline"),
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2)
    )

    frame_path = output_dir / "tabular_features.csv"
    frame_snapshot_path = output_dir / "tabular_features.snapshot"
    cached_snapshot = (
        frame_snapshot_path.read_text().strip()
        if frame_snapshot_path.exists()
        else None
    )
    if frame_path.exists() and cached_snapshot == config["tensor_snapshot_id"]:
        frame = pd.read_csv(frame_path)
    else:
        frame = _build_tabular_frame(dataset, len(vocabulary))
        frame.to_csv(frame_path, index=False)
        frame_snapshot_path.write_text(config["tensor_snapshot_id"] + "\n")

    evaluations = _evaluate_classical(frame, splits, args.seed)
    if not args.skip_neural:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_specs = (
            ("mlp", "pooled_mlp", False),
            ("mlp", "pooled_mlp_no_pragmas", True),
            ("rgcn", "rgcn", False),
            ("rgcn", "rgcn_no_pragmas", True),
        )
        for model_name, output_name, without_pragmas in model_specs:
            if output_name not in args.neural_models:
                continue
            evaluations.extend(
                _evaluate_neural(
                    dataset=dataset,
                    splits=splits,
                    model_name=model_name,
                    output_name=output_name,
                    instruction_vocab_size=len(vocabulary),
                    max_position=max_position,
                    output_dir=output_dir,
                    device=device,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    without_pragmas=without_pragmas,
                )
            )

    metrics = _metrics_frame(evaluations)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    _save_predictions(evaluations, dataset, splits, output_dir)
    _save_plots(evaluations, output_dir)
    pragma_audit = _pragma_audit(graph_root, dataset)
    (output_dir / "pragma_audit.json").write_text(
        json.dumps(pragma_audit, indent=2)
    )
    _write_report(output_dir, config, manifest, metrics, pragma_audit)
    print(f"Wrote benchmark report to {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
