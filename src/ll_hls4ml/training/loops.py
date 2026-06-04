"""Training and validation loops."""

from __future__ import annotations

from pathlib import Path
import copy
import gc
import time
import torch
import numpy as np
from tqdm.notebook import tqdm

from ll_hls4ml.data.dataset import HeteroGraphDataset
from ll_hls4ml.training.targets import normalize_target
from ll_hls4ml.training.loaders import make_loader
from ll_hls4ml.data.splits import compute_target_stats, random_train_val_split
from ll_hls4ml.io.schema import NODE_TYPES, LABEL_KEYS


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


def _release_cuda_memory(*objects: object) -> None:
    """Move modules off GPU, collect, and return cached memory to the driver."""
    for obj in objects:
        if isinstance(obj, torch.nn.Module):
            obj.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _optimizer_for_model(template: torch.optim.Optimizer, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Fresh optimizer for ``model`` with the same hyperparameters as ``template``."""
    cls = type(template)
    if len(template.param_groups) == 1:
        kwargs = {k: v for k, v in template.param_groups[0].items() if k != "params"}
        return cls(model.parameters(), **kwargs)
    param_groups = [
        {"params": list(model.parameters()), **{k: v for k, v in group.items() if k != "params"}}
        for group in template.param_groups
    ]
    return cls(param_groups)


def train_one_epoch(model, train_loader, criterion, optimizer, device, y_means, y_stds, pbar=None):
    model.train()
    running_loss = 0.0
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
        if pbar is not None:
            pbar.update(1)

    return running_loss / len(train_loader)


def validate_one_epoch(model, val_loader, criterion, device, y_means, y_stds, pbar=None):
    model.eval()
    all_preds, all_targets = [], []
    running_loss = 0.0

    num_targets = y_means.shape[0]

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            target = normalize_target(batch.y.view(-1, num_targets), y_means, y_stds)   # [batch_size, num_targets]

            pred = model(batch)                                                         # [batch_size, num_targets]
            loss = torch.nn.functional.huber_loss(pred, target, delta=1.0)

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
):
    """
    Train model with optional early stopping.

    Checkpoints are saved to ``checkpoint_dir / {experiment_name}_model.pt``.
    """
    checkpoint_dir = Path(checkpoint_dir or "artifacts/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{experiment_name}_model.pt"
    backup_path = checkpoint_dir / f"{experiment_name}_backup.pt"

    training_history = {"train_loss": [], "val_loss": [], "r_per_target": [], "std_ratio_per_target": []}

    y_means, y_stds = compute_target_stats(train_loader.dataset)
    y_means, y_stds = y_means.to(device), y_stds.to(device)

    if patience > 0:
        patience_counter = 0
        best_metric = float("-inf") if mode == "max" else float("inf")
        best_epoch = 0

    print(f"Training {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        leave_bar = True if epoch == 1 or (verbose > 0 and epoch % verbose == 0) else False
        pbar = tqdm(
            total=len(train_loader) + len(val_loader),
            desc=f"Epoch {epoch:3d}/{epochs}",
            unit="batch",
            leave=leave_bar)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, y_means, y_stds, pbar=pbar
        )
        val_loss, preds, targets, r_per_target, std_ratio_per_target = validate_one_epoch(
            model, val_loader, criterion, device, y_means, y_stds, pbar=pbar
        )

        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["r_per_target"].append(r_per_target)
        training_history["std_ratio_per_target"].append(std_ratio_per_target)

        if leave_bar:
            # take mean of R's onoly where not NaN
            valid_R = np.nanmean(r_per_target[~np.isnan(r_per_target)])
            valid_std_r = np.nanmean(std_ratio_per_target[~np.isnan(std_ratio_per_target)])
            pbar.set_postfix({
                "train": f"{train_loss:.4f}",
                "val": f"{val_loss:.4f}",
                "R": f"{valid_R:.2f}",
                "std_r": f"{valid_std_r:.2f}",
            })
        pbar.close()


        if patience > 0:
            current_metric = training_history[evaluation_metric][-1]
            is_improvement = (
                current_metric > best_metric if mode == "max" else current_metric < best_metric
            )

            if is_improvement:
                best_metric = current_metric
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
                training_history["best_val_preds"] = _persist_on_cpu(preds)
                training_history["best_val_targets"] = _persist_on_cpu(targets)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        if epoch % 10 == 0:
            torch.save(model.state_dict(), backup_path)
            print(f"Model saved to {backup_path}")

    if restore_best_weights and patience > 0 and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        print(
            f"Best model restored from epoch {best_epoch} "
            f"with {evaluation_metric} {best_metric:.4f}"
        )

    if patience == 0:
        torch.save(model.state_dict(), ckpt_path)
        training_history["best_val_preds"] = _persist_on_cpu(preds)
        training_history["best_val_targets"] = _persist_on_cpu(targets)

    return (
        model,
        y_means,
        y_stds,
        _persist_on_cpu(training_history),
        best_metric,
        best_epoch,
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
):
    """
    Fits N models, one for each kernel type, where that kernel type is used as the inductive set,
    and the rest of the kernel types are used as the transductive set.
    Validation/early stopping is done on the transductive set's MAPE (mean absolute percentage error).

    Returns:
        A dictionary of metrics, keyed by inductive kernel type, containing:
            train_loss: Train loss (MSE)
            val_loss: Validation loss (MAPE)
            r_per_target: Pearson correlation coefficient per target
            std_ratio_per_target: Ratio of predicted standard deviation to target standard deviation per target
            best_val_preds: Best validation predictions
            best_val_targets: Best validation targets
            best_val_loss: Best validation loss (MAPE) during training
            best_epoch: Best epoch during training
            test_loss: Test loss (MAPE)
            test_preds: Test predictions
            test_targets: Test targets
            y_means: Per-target means used for normalization (CPU numpy)
            y_stds: Per-target stds used for normalization (CPU numpy)

        All array/tensor fields are stored as CPU numpy arrays.
    """
    kernel_types = list(max_per_kernel_type.keys())
    device_type = device.type if isinstance(device, torch.device) else str(device)
    use_cuda = device_type.startswith("cuda") and torch.cuda.is_available()

    training_histories = {}
    for kernel_type in kernel_types if kernel_type in ["dense_resource", "exemplar", "rule4ml"]:
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

            train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False)
            test_loader = make_loader(test_ds, batch_size=batch_size, shuffle=False)

            # Count number of Out-of-Vocabulary (OOV) nodes
            nodes_per_type = {nt: 0 for nt in NODE_TYPES}
            oov_nodes_per_type = {nt: 0 for nt in NODE_TYPES}
            for batch in test_loader:
                for nt in NODE_TYPES:
                    nodes_per_type[nt] += batch[nt].x.shape[0]
                    oov_nodes_per_type[nt] += (batch[nt].x == 0).sum().item()

            print(
                f"*********** \"{kernel_type}\" as inductive set ***********\n"
                f"  Train size: {len(train_ds)}, "
                f"  Val size: {len(val_ds)}, "
                f"  Test size: {len(test_ds)}"
            )
            for nt in NODE_TYPES:
                oov_pct = oov_nodes_per_type[nt] / nodes_per_type[nt] * 100
                print(
                    f"  {nt}: {nodes_per_type[nt]} nodes, "
                    f"{oov_nodes_per_type[nt]} OOV nodes ({oov_pct:.2f}%)"
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
            )

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

            print(
                f"Test loss (MAPE): {test_loss * 100:.2f}%, "
                f"Best validation loss (MAPE): {best_val_loss * 100:.2f}%, "
                f"Inductive-Transductive gap: {(test_loss - best_val_loss) * 100:.2f}%"
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
        print("***********************************************\n\n")

    return training_histories