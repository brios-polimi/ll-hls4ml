"""Training and validation loops."""

from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ll_hls4ml.data.dataset import HeteroGraphDataset
from ll_hls4ml.training.targets import (
    compute_target_z_stats,
    normalize_target,
    wahls4ml_metrics,
)
from ll_hls4ml.training.loaders import make_loader
from ll_hls4ml.training.distributed import (
    is_distributed,
    is_main_process,
    unwrap_model,
    wrap_ddp,
)
from ll_hls4ml.data.splits import random_train_val_test_split
from ll_hls4ml.io.schema import LABEL_KEYS


def _use_progress_bar() -> bool:
    """Use tqdm only on an interactive terminal; log files get one-line epoch summaries."""
    if not is_main_process():
        return False
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return False
    if os.environ.get("LL_HLS4ML_TQDM", "1").lower() in {"0", "false", "no"}:
        return False
    return True


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
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
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
    raise ValueError("Leave-family-out training currently requires one optimizer group")


def _save_checkpoint(epoch, model, optimizer, scheduler, path):
    torch.save({
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }, path)


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    pbar=None,
    distributed: bool = False,
):
    model.train()
    base_model = unwrap_model(model)
    loss_sum = 0.0
    num_samples = 0
    num_targets = len(base_model.y_means)

    for batch in train_loader:
        batch = batch.to(device)
        target = normalize_target(
            batch.y.view(-1, num_targets),
            base_model.y_means,
            base_model.y_stds,
        )

        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        n = batch.num_graphs
        loss_sum += loss.item() * n
        num_samples += n

        if pbar is not None:
            pbar.update(1)

    if distributed and dist.is_initialized():
        stats = torch.tensor([loss_sum, float(num_samples)], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        loss_sum, num_samples = stats.tolist()

    return loss_sum / num_samples if num_samples else 0.0


def validate_one_epoch(
    model,
    val_loader,
    criterion,
    device,
    pbar=None,
    distributed: bool = False,
):
    if distributed and not is_main_process():
        return 0.0, None, None, None, None

    base_model = unwrap_model(model)
    evaluation_model = base_model if distributed else model
    evaluation_model.eval()
    all_preds, all_targets = [], []
    loss_sum = 0.0
    num_samples = 0
    num_targets = len(base_model.y_means)

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            target = normalize_target(
                batch.y.view(-1, num_targets),
                base_model.y_means,
                base_model.y_stds,
            )

            pred = evaluation_model(batch)
            loss = criterion(pred, target)

            all_preds.append(pred)
            all_targets.append(target)

            n = batch.num_graphs
            loss_sum += loss.item() * n
            num_samples += n

            if pbar is not None:
                pbar.update(1)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # print(model.y_means, model.y_stds, "\n", preds, "\n", targets)
    # print(preds.std(dim=0))

    metrics = wahls4ml_metrics(
        preds, targets, base_model.y_means, base_model.y_stds
    )
    metrics["loss"] = loss_sum / num_samples if num_samples else 0.0
    return metrics


def fit(
    model,
    train_loader,
    val_loader,
    epochs,
    criterion,
    optimizer,
    scheduler,
    device,
    patience=50,
    mode="min",
    restore_best_weights=True,
    writer=None,
    verbose=2,
    experiment_name="",
    checkpoint_dir: str | Path | None = None,
    resume_from_backup: str | Path | None = None,
    distributed: bool = False,
):
    """
    Train model with optional early stopping.

    Checkpoints are saved under ``checkpoint_dir`` using ``experiment_name``.
    """
    distributed = distributed or is_distributed()
    main = is_main_process()

    checkpoint_dir = Path(checkpoint_dir or "artifacts/checkpoints")
    if main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{experiment_name}_checkpoint.pt"
    backup_path = checkpoint_dir / f"{experiment_name}_backup.pt"

    if resume_from_backup:
        checkpoint = torch.load(resume_from_backup, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 1

    model = model.to(device)
    if distributed:
        model = wrap_ddp(model, device)

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

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        log_epoch = main and (epoch == 1 or (verbose > 0 and epoch % verbose == 0))
        pbar = None
        if _use_progress_bar():
            pbar = tqdm(
                total=len(train_loader) + len(val_loader),
                desc=f"Epoch {epoch:3d}/{epochs}",
                unit="batch",
                leave=log_epoch,
            )

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            pbar=pbar, distributed=distributed,
        )
        val_metrics = validate_one_epoch(
            model, val_loader, criterion, device,
            pbar=pbar, distributed=distributed,
        )

        if pbar is not None:
            pbar.set_postfix({
                "train": f"{train_loss:.4f}",
                "val": f"{val_metrics['loss']:.4f}"
            })
            pbar.close()
        elif log_epoch:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train={train_loss:.4f}  val={val_metrics['loss']:.4f}",
                flush=True,
            )

        if main and writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch) 
            writer.add_scalar("loss/val", val_metrics['loss'], epoch) 
            for name, value in val_metrics.items(): 
                if name != 'loss':
                    for i, lbl in enumerate(LABEL_KEYS):
                        writer.add_scalar(f"{name}/{lbl}", value[i].item(), epoch)

        should_stop.zero_()

        if patience > 0 and main:
            current_metric = val_metrics["loss"]
            is_improvement = (
                current_metric > best_metric if mode == "max" else current_metric < best_metric
            )

            if is_improvement:
                best_metric = current_metric
                best_epoch = epoch
                patience_counter = 0
                _save_checkpoint(epoch, model, optimizer, scheduler, ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    should_stop[0] = 1

        if main and epoch % 10 == 0:
            _save_checkpoint(epoch, model, optimizer, scheduler, backup_path)
            print(f"Model saved to {backup_path}")

        if distributed:
            dist.broadcast(should_stop, src=0)

        if patience > 0 and should_stop.item() == 1:
            break

    if restore_best_weights and patience > 0 and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        unwrap_model(model).load_state_dict(checkpoint["model"])
        if main:
            print(
                f"Best model restored from epoch {best_epoch} "
                f"with validation loss {best_metric:.4f}"
            )

    if patience == 0 and main:
        _save_checkpoint(epoch, model, optimizer, scheduler, ckpt_path)

    if distributed:
        dist.barrier()

    return model


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

            train_ds, val_ds, _unused_test = random_train_val_test_split(
                train_val_ds,
                val_fraction=0.2,
                test_fraction=0.0,
            )

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
            y_means, y_stds = compute_target_z_stats(train_ds)
            model.y_means.copy_(y_means)
            model.y_stds.copy_(y_stds)
            loop_optimizer = _optimizer_for_model(optimizer, model)
            model = fit(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                criterion=criterion,
                optimizer=loop_optimizer,
                scheduler=None,
                device=device,
                patience=patience,
                mode=mode,
                restore_best_weights=restore_best_weights,
                writer=writer,
                verbose=verbose,
                experiment_name=experiment_name + f"_{kernel_type}",
                checkpoint_dir=checkpoint_dir,
                distributed=distributed,
            )

            if main:
                print("Evaluating on test set...")
                test_metrics = validate_one_epoch(
                    model, test_loader, criterion, device
                )

                training_histories[kernel_type] = _persist_on_cpu(
                    {
                        "y_means": y_means,
                        "y_stds": y_stds,
                        "test_metrics": test_metrics,
                    }
                )

                th_path = results_dir / f"training_histories_{kernel_type}.json"
                with open(th_path, "w") as f:
                    json.dump(training_histories[kernel_type], f, default=_json_converter)

                print(f"Test loss: {test_metrics['loss']:.4f}")
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
