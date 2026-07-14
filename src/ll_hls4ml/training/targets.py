"""Target normalization for log-transformed LUT (or other) labels."""

import torch

from ll_hls4ml.io.schema import LABEL_KEYS

def compute_target_z_stats(dataset, std_floor: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std of log1p(targets) over dataset

    Returns tensors of shape (len(LABEL_KEYS)) for mean and std.
    """
    log_ys = torch.log1p(torch.stack([graph.y for graph in dataset], dim=0))
    y_means = log_ys.mean(dim=0)
    y_stds = torch.clamp(log_ys.std(dim=0), min=std_floor)
    return y_means, y_stds


def normalize_target(y: torch.Tensor, y_means: torch.Tensor, y_stds: torch.Tensor) -> torch.Tensor:
    """Normalize targets to have mean 0 and std 1."""
    return (torch.log1p(y) - y_means) / y_stds


def denormalize_target(pred: torch.Tensor, y_means: torch.Tensor, y_stds: torch.Tensor) -> torch.Tensor:
    """Denormalize model outputs back to original scales."""
    return torch.expm1(pred * y_stds + y_means)

def wahls4ml_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    y_means: torch.Tensor,
    y_stds: torch.Tensor
):
    """
    `predictions` and `targets` in normalized log form,
    function denormalizes both

    Returns wa-hls4ml's required metrics per target:
     Coefficient of determination (R2)
     Symmetric mean absolute percentage error (SMAPE)
     Root mean square error (RMSE)
    """
    predictions = denormalize_target(predictions.clone(), y_means, y_stds)
    targets = denormalize_target(targets.clone(), y_means, y_stds)


    R2 = 1 - torch.sum((targets - predictions) ** 2, dim=0) / torch.sum((targets - torch.mean(targets, dim=0)) ** 2, dim=0)
    SMAPE = torch.mean(torch.abs(targets - predictions) / (torch.abs(targets) + torch.abs(predictions) + 1.0), dim=0) * 200
    RMSE = torch.sqrt(torch.mean((targets - predictions) ** 2, dim=0))

    return {
        "r2": R2,
        "smape": SMAPE,
        "rmse": RMSE
    }
