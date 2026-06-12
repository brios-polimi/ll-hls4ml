"""Training and validation loops."""

from __future__ import annotations

from pathlib import Path
import copy
import gc
import os
import sys
import torch
import torch.distributed as dist
import json
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ll_hls4ml.data.dataset import HeteroGraphDataset
from ll_hls4ml.training.targets import normalize_target
from ll_hls4ml.training.loaders import make_loader
from ll_hls4ml.training.distributed import (
    is_distributed,
    is_main_process,
    unwrap_model,
    wrap_ddp,
)
from ll_hls4ml.data.splits import compute_target_stats, random_train_val_split
from ll_hls4ml.io.schema import LABEL_KEYS


def _use_progress_bar() -> bool:
    """Use tqdm only on an interactive terminal; log files get one-line epoch summaries."""
    if not is_main_process():
        return False
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return False
    if os.environ.get("LL_HLS4ML_TQDM", "1").lower() in {"0", "false", "no"}:
        return False
    return sys.stdout.isatty()


def _persist_on_cpu(value):
    """Recursively copy tensors to CPU numpy for long-lived training results."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: _persist_on_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_persist_on_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_persist_on_cpu(v) for v in value)
    if isinstance(value, np.ndarray):
        return np.copy(value)
    return value


def _json_converter(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_converter(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_converter(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_json_converter(v) for v in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def _release_cuda_memory(*objects: object) -> None:
    """Move modules off GPU, collect, and return cached memory to the driver."""
    for obj in objects:
        if isinstance(obj, torch.nn.Module):
            unwrap_model(obj).cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _optimizer_for_model(template: torch.optim.Optimizer, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Fresh optimizer for ``model`` with the same hyperparameters as ``template``."""
    cls = type(template)
    params = unwrap_model(model).parameters()
    if len(template.param_groups) == 1:
        kwargs = {k: v for k, v in template.param_groups[0].items() if k != "params"}
        return cls(params, **kwargs)
    param_groups = [
        {"params": list(params), **{k: v for k, v in group.items() if k != "params"}}
        for group in template.param_groups
    ]
    return cls(param_groups)


def _broadcast_target_stats(
    train_dataset,
    device: torch.device,
    distributed: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if distributed and is_main_process():
        y_means, y_stds = compute_target_stats(train_dataset)
    elif distributed:
        y_means = torch.zeros(len(LABEL_KEYS))
        y_stds = torch.ones(len(LABEL_KEYS))
    else:
        y_means, y_stds = compute_target_stats(train_dataset)

    y_means = y_means.to(device)
    y_stds = y_stds.to(device)

    if distributed:
        dist.broadcast(y_means, src=0)
        dist.broadcast(y_stds, src=0)

    return y_means, y_stds


def _save_state_dict(model: torch.nn.Module, path: Path) -> None:
    torch.save(unwrap_model(model).state_dict(), path)


def _load_state_dict(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    unwrap_model(model).load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    y_means,
    y_stds,
    pbar=None,
    distributed: bool = False,
):
    model.train()
    running_loss = 0.0
    num_batches = 0
    num_targets = y_means.shape[0]

    for batch in train_loader:
        batch = batch.to(device)
        target = normalize_target(batch.y.view(-1, num_targets), y_means, y_stds)

        pred = model(batch)
        loss = criterion(pred, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        num_batches += 1
        if pbar is not None:
            pbar.update(1)

    if distributed and dist.is_initialized():
        stats = torch.tensor([running_loss, float(num_batches)], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        running_loss = stats[0].item()
        num_batches = int(stats[1].item())

    return running_loss / num_batches if num_batches else 0.0


def validate_one_epoch(
    model,
    val_loader,
    criterion,
    device,
    y_means,
    y_stds,
    pbar=None,
    distributed: bool = False,
):
    if distributed and not is_main_process():
        return 0.0, None, None, None, None

    model.eval()
    all_preds, all_targets = [], []
    running_loss = 0.0

    num_targets = y_means.shape[0]

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            target = normalize_target(batch.y.view(-1, num_targets), y_means, y_stds)

            pred = model(batch)
            loss = criterion(pred, target)

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
            running_loss += loss.item()
            if pbar is not None:
                pbar.update(1)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    r_per_target = np.array([
        np.corrcoef(preds[:, i], targets[:, i])[0, 1]
        if targets[:, i].std() > 1e-6 and preds[:, i].std() > 1e-6
        else np.nan
        for i in range(preds.shape[1])
    ])

    target_stds = targets.std(axis=0)
    pred_stds = preds.std(axis=0)
    std_ratio_per_target = np.where(target_stds > 1e-6, pred_stds / target_stds, np.nan)

    epoch_loss = running_loss / len(val_loader)
    return epoch_loss, preds, targets, r_per_target, std_ratio_per_target


def fit(
    model,
    train_loader,
    val_loader,
    epochs,
    criterion,
    optimizer,
    device,
    patience=50,
    evaluation_metric="val_loss",
    mode="min",
    restore_best_weights=True,
    writer=None,
    verbose=2,
    experiment_name="",
    checkpoint_dir: str | Path | None = None,
    distributed: bool = False,
):
    """
    Train model with optional early stopping.

    Checkpoints are saved to ``checkpoint_dir / {experiment_name}_model.pt``.
  """
    distributed = distributed or is_distributed()
    main = is_main_process()

    checkpoint_dir = Path(checkpoint_dir or "artifacts/checkpoints")
    if main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{experiment_name}_model.pt"
    backup_path = checkpoint_dir / f"{experiment_name}_backup.pt"

    model = model.to(device)
    if distributed:
        model = wrap_ddp(model, device)

    training_history = {"train_loss": [], "val_loss": [], "r_per_target": [], "std_ratio_per_target": []}

    y_means, y_stds = _broadcast_target_stats(train_loader.dataset, device, distributed)

    train_sampler = (
        train_loader.sampler
        if isinstance(train_loader.sampler, DistributedSampler)
        else None
    )

    if patience > 0:
        patience_counter = 0
        best_metric = float("-inf") if mode == "max" else float("inf")
        best_epoch = 0

    if main:
        print(f"Training {epochs} epochs...")

    should_stop = torch.zeros(1, device=device)

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        log_epoch = main and (epoch == 1 or (verbose > 0 and epoch % verbose == 0))
        pbar = None
        if log_epoch and _use_progress_bar():
            pbar = tqdm(
                total=len(train_loader) + len(val_loader),
                desc=f"Epoch {epoch:3d}/{epochs}",
                unit="batch",
                leave=False,
            )

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, y_means, y_stds,
            pbar=pbar, distributed=distributed,
        )
        val_loss, preds, targets, r_per_target, std_ratio_per_target = validate_one_epoch(
            model, val_loader, criterion, device, y_means, y_stds,
            pbar=pbar, distributed=distributed,
        )

        if pbar is not None:
            if r_per_target is not None:
                valid_R = np.nanmean(r_per_target[~np.isnan(r_per_target)])
                valid_std_r = np.nanmean(std_ratio_per_target[~np.isnan(std_ratio_per_target)])
                pbar.set_postfix({
                    "train": f"{train_loss:.4f}",
                    "val": f"{val_loss:.4f}",
                    "R": f"{valid_R:.2f}",
                    "std_r": f"{valid_std_r:.2f}",
                })
            pbar.close()
        elif log_epoch and r_per_target is not None:
            valid_R = np.nanmean(r_per_target[~np.isnan(r_per_target)])
            valid_std_r = np.nanmean(std_ratio_per_target[~np.isnan(std_ratio_per_target)])
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"R={valid_R:.2f}  std_r={valid_std_r:.2f}",
                flush=True,
            )

        if main:
            training_history["train_loss"].append(train_loss)
            training_history["val_loss"].append(val_loss)
            training_history["r_per_target"].append(r_per_target)
            training_history["std_ratio_per_target"].append(std_ratio_per_target)

        should_stop.zero_()

        if patience > 0 and main:
            current_metric = training_history[evaluation_metric][-1]
            is_improvement = (
                current_metric > best_metric if mode == "max" else current_metric < best_metric
            )

            if is_improvement:
                best_metric = current_metric
                best_epoch = epoch
                _save_state_dict(model, ckpt_path)
                training_history["best_val_preds"] = _persist_on_cpu(preds)
                training_history["best_val_targets"] = _persist_on_cpu(targets)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    should_stop[0] = 1

        if distributed:
            dist.broadcast(should_stop, src=0)

        if patience > 0 and should_stop.item() == 1:
            break

        if main and epoch % 10 == 0:
            _save_state_dict(model, backup_path)
            print(f"Model saved to {backup_path}")

    if restore_best_weights and patience > 0 and ckpt_path.exists():
        _load_state_dict(model, ckpt_path, device)
        if main:
            print(
                f"Best model restored from epoch {best_epoch} "
                f"with {evaluation_metric} {best_metric:.4f}"
            )

    if patience == 0 and main:
        best_metric = training_history[evaluation_metric][-1]
        best_epoch = epoch
        _save_state_dict(model, ckpt_path)
        training_history["best_val_preds"] = _persist_on_cpu(preds)
        training_history["best_val_targets"] = _persist_on_cpu(targets)

    if distributed:
        dist.barrier()

    return (
        model,
        y_means,
        y_stds,
        _persist_on_cpu(training_history) if main else {},
        best_metric if main else None,
        best_epoch if main else None,
    )


def transductive_vs_inductive_fit(
    original_model,
    max_per_kernel_type: dict[str, int],
    tensor_dir: str | Path,
    epochs,
    batch_size,
    criterion,
    optimizer,
    device,
    patience=50,
    evaluation_metric="val_loss",
    mode="min",
    restore_best_weights=True,
    writer=None,
    verbose=10,
    experiment_name="",
    checkpoint_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    distributed: bool = False,
):
    """
    Fits N models, one for each kernel type, where that kernel type is used as the inductive set,
    and the rest of the kernel types are used as the transductive set.
    Validation/early stopping is done on the transductive set's loss.

    Returns:
        A dictionary of metrics, keyed by inductive kernel type, containing:
            train_loss: Train loss
            val_loss: Validation loss
            r_per_target: Pearson correlation coefficient per target
            std_ratio_per_target: Ratio of predicted standard deviation to target standard deviation per target
            best_val_preds: Best validation predictions
            best_val_targets: Best validation targets
            best_val_loss: Best validation loss during training
            best_epoch: Best epoch during training
            test_loss: Test loss
            test_preds: Test predictions
            test_targets: Test targets
            y_means: Per-target means used for normalization (CPU numpy)
            y_stds: Per-target stds used for normalization (CPU numpy)

        All array/tensor fields are stored as CPU numpy arrays.
        Under DDP, only rank 0 returns the full dictionary; other ranks return {}.
    """
    distributed = distributed or is_distributed()
    main = is_main_process()

    checkpoint_dir = Path(checkpoint_dir or "artifacts/checkpoints")
    results_dir = Path(results_dir) if results_dir is not None else checkpoint_dir.parent
    if main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    kernel_types = list(max_per_kernel_type.keys())
    device_type = device.type if isinstance(device, torch.device) else str(device)
    use_cuda = device_type.startswith("cuda") and torch.cuda.is_available()

    training_histories = {}
    for kernel_type in kernel_types:
        model = None
        loop_optimizer = None
        train_loader = val_loader = test_loader = None
        train_ds = val_ds = test_ds = train_val_ds = None
        y_means = y_stds = None

        try:
            if use_cuda:
                _release_cuda_memory()

            train_val_ds = HeteroGraphDataset(
                tensor_dir,
                types=[t for t in kernel_types if t != kernel_type],
                max_per_type={
                    k: v for k, v in max_per_kernel_type.items() if k != kernel_type
                },
            )
            test_ds = HeteroGraphDataset(
                tensor_dir,
                types=[kernel_type],
                max_per_type=max_per_kernel_type[kernel_type],
            )

            train_ds, val_ds = random_train_val_split(train_val_ds, val_fraction=0.2)

            train_loader = make_loader(
                train_ds, batch_size=batch_size, shuffle=True, distributed=distributed,
            )
            val_loader = make_loader(
                val_ds, batch_size=batch_size, shuffle=False, distributed=False,
            )
            test_loader = make_loader(
                test_ds, batch_size=batch_size, shuffle=False, distributed=False,
            )

            if main:
                print(
                    f"*********** \"{kernel_type}\" as inductive set ***********\n"
                    f"  Train size: {len(train_ds)}, "
                    f"  Val size: {len(val_ds)}, "
                    f"  Test size: {len(test_ds)}"
                )

            model = copy.deepcopy(original_model).to(device)
            loop_optimizer = _optimizer_for_model(optimizer, model)
            model, y_means, y_stds, training_history, best_val_loss, best_epoch = fit(
                model,
                train_loader,
                val_loader,
                epochs,
                criterion,
                loop_optimizer,
                device,
                patience,
                evaluation_metric,
                mode,
                restore_best_weights,
                writer,
                verbose,
                experiment_name + f"_{kernel_type}",
                checkpoint_dir,
                distributed=distributed,
            )

            if main:
                print("Evaluating on test set...")
                test_loss, test_preds, test_targets, _, _ = validate_one_epoch(
                    model, test_loader, criterion, device, y_means, y_stds
                )

                training_histories[kernel_type] = _persist_on_cpu(
                    {
                        **training_history,
                        "y_means": y_means,
                        "y_stds": y_stds,
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,
                        "test_loss": test_loss,
                        "test_preds": test_preds,
                        "test_targets": test_targets,
                    }
                )

                th_path = results_dir / f"training_histories_{kernel_type}.json"
                with open(th_path, "w") as f:
                    json.dump(training_histories[kernel_type], f, default=_json_converter)

                print(
                    f"Test loss: {test_loss:.4f}, "
                    f"Best validation loss: {best_val_loss:.4f}, "
                    f"Inductive-Transductive gap: {(test_loss - best_val_loss):.4f}"
                )
        finally:
            _release_cuda_memory(
                model,
                loop_optimizer,
                train_loader,
                val_loader,
                test_loader,
                train_ds,
                val_ds,
                test_ds,
                train_val_ds,
                y_means,
                y_stds,
            )
        if main:
            print("***********************************************\n\n")

        if distributed:
            dist.barrier()

    return training_histories
