"""Target normalization and wa-hls4ml benchmark metrics."""

import torch
import torch.nn as nn
from torch.utils.data import Subset

from ll_hls4ml.io.schema import LABEL_KEYS


class LogHuberLoss(nn.Module):
    """Huber loss with one fixed meaning in log1p space for every target."""

    def __init__(self, y_stds: torch.Tensor, delta: float = 0.35):
        super().__init__()
        self.register_buffer("y_stds", y_stds.detach().clone())
        self.delta = delta

    def forward(
        self,
        normalized_prediction: torch.Tensor,
        normalized_target: torch.Tensor,
    ) -> torch.Tensor:
        residual = (
            normalized_prediction - normalized_target
        ) * self.y_stds
        absolute = residual.abs()
        loss = torch.where(
            absolute < self.delta,
            0.5 * residual.square(),
            self.delta * (absolute - 0.5 * self.delta),
        )
        return loss.mean()


class LogHuberHurdleLoss(nn.Module):
    """Log-space regression plus zero/nonzero heads for DSP and BRAM."""

    hurdle_target_indices = (2, 3)

    def __init__(
        self,
        y_means: torch.Tensor,
        y_stds: torch.Tensor,
        delta: float = 0.35,
        classification_weight: float = 0.25,
    ):
        super().__init__()
        self.register_buffer("y_means", y_means.detach().clone())
        self.register_buffer("y_stds", y_stds.detach().clone())
        self.delta = delta
        self.classification_weight = classification_weight
        self.classification = nn.BCEWithLogitsLoss()

    def forward(
        self,
        prediction: torch.Tensor,
        normalized_target: torch.Tensor,
    ) -> torch.Tensor:
        regression = prediction[:, : len(LABEL_KEYS)]
        residual = (regression - normalized_target) * self.y_stds
        absolute = residual.abs()
        element_loss = torch.where(
            absolute < self.delta,
            0.5 * residual.square(),
            self.delta * (absolute - 0.5 * self.delta),
        )
        target_log = normalized_target * self.y_stds + self.y_means
        regression_mask = torch.ones_like(element_loss)
        for target_index in self.hurdle_target_indices:
            regression_mask[:, target_index] = (
                target_log[:, target_index] > 0
            )
        regression_loss = (
            element_loss * regression_mask
        ).sum() / regression_mask.sum().clamp_min(1)
        positive = (
            target_log[:, self.hurdle_target_indices] > 0
        ).to(regression.dtype)
        classification_loss = self.classification(
            prediction[:, len(LABEL_KEYS):],
            positive,
        )
        return regression_loss + self.classification_weight * classification_loss

def compute_target_z_stats(dataset, std_floor: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std of log1p(targets) over dataset

    Returns tensors of shape (len(LABEL_KEYS)) for mean and std.
    """
    targets = getattr(dataset, "targets", None)
    if targets is None and isinstance(dataset, Subset):
        base_targets = getattr(dataset.dataset, "targets", None)
        if base_targets is not None:
            targets = base_targets[torch.as_tensor(dataset.indices, dtype=torch.long)]
    if targets is None:
        targets = torch.stack([graph.y for graph in dataset], dim=0)

    log_ys = torch.log1p(targets)
    y_means = log_ys.mean(dim=0)
    y_stds = torch.clamp(log_ys.std(dim=0), min=std_floor)
    return y_means, y_stds


def normalize_target(y: torch.Tensor, y_means: torch.Tensor, y_stds: torch.Tensor) -> torch.Tensor:
    """Normalize targets to have mean 0 and std 1."""
    return (torch.log1p(y) - y_means) / y_stds


def denormalize_target(pred: torch.Tensor, y_means: torch.Tensor, y_stds: torch.Tensor) -> torch.Tensor:
    """Denormalize model outputs back to original scales."""
    return torch.expm1(pred * y_stds + y_means)


def apply_hurdle_prediction(
    prediction: torch.Tensor,
    y_means: torch.Tensor,
    y_stds: torch.Tensor,
    mode: str = "expected",
) -> torch.Tensor:
    """Apply optional DSP/BRAM hurdle outputs and return normalized values."""
    regression = prediction[:, : len(LABEL_KEYS)]
    if prediction.shape[-1] == len(LABEL_KEYS):
        return regression
    if prediction.shape[-1] != len(LABEL_KEYS) + 2:
        raise ValueError(f"Unexpected prediction width: {prediction.shape[-1]}")
    raw = denormalize_target(regression, y_means, y_stds).clamp_min(0)
    probability = prediction[:, len(LABEL_KEYS):].sigmoid()
    if mode == "expected":
        multiplier = probability
    elif mode == "threshold":
        multiplier = (probability >= 0.5).to(raw.dtype)
    else:
        raise ValueError(f"Unknown hurdle prediction mode: {mode}")
    raw = raw.clone()
    raw[:, 2:4] *= multiplier
    return (torch.log1p(raw) - y_means) / y_stds


def wahls4ml_metrics_raw(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute the paper's R², SMAPE, and RMSE on original-scale targets."""
    predictions = predictions.float()
    targets = targets.float()
    residual_sum = torch.sum((targets - predictions) ** 2, dim=0)
    total_sum = torch.sum(
        (targets - torch.mean(targets, dim=0)) ** 2,
        dim=0,
    ).clamp_min(torch.finfo(targets.dtype).eps)
    r2 = 1 - residual_sum / total_sum
    smape = (
        torch.mean(
            torch.abs(targets - predictions)
            / (torch.abs(targets) + torch.abs(predictions) + 1.0),
            dim=0,
        )
        * 200
    )
    rmse = torch.sqrt(torch.mean((targets - predictions) ** 2, dim=0))
    return {"r2": r2, "smape": smape, "rmse": rmse}


def relative_percentage_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Paper RPE: ``(target - prediction) / (target + 1) * 100``."""
    return (targets - predictions) / (targets + 1.0) * 100


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


    return wahls4ml_metrics_raw(predictions, targets)
