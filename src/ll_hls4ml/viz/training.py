"""Training evaluation plots."""

import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator
import torch
import numpy as np

from ll_hls4ml.training.targets import denormalize_target, relative_percentage_error
from ll_hls4ml.io.schema import LABEL_KEYS


def plot_loss_curves(training_history, title="Training Loss"):
    plt.figure()
    plt.plot(training_history["train_loss"], label="Train Loss")
    plt.plot(training_history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_predictions_vs_targets(
    model,
    val_loader,
    device,
    y_mean,
    y_std,
    figsize=(12, 6),
):
    """Sorted prediction vs target curves on validation set."""
    model.eval()
    num_targets = len(LABEL_KEYS)
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = denormalize_target(
                model(batch),
                y_mean.to(device),
                y_std.to(device),
            )
            all_preds.append(preds.cpu())
            all_targets.append(batch.y.view(-1, num_targets).cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    for i in range(num_targets):
        sort_idx = all_targets[:, i].argsort()
        sorted_targets = all_targets[:, i][sort_idx]
        sorted_preds = all_preds[:, i][sort_idx]

        err = np.abs((sorted_preds - sorted_targets) / (sorted_targets)) * 100
        err = err.mean()
        print(f"Mean error for {LABEL_KEYS[i]}: {err:.2f}%")


        plt.figure(figsize=figsize)
        plt.plot(sorted_preds + 1, label="Prediction")
        plt.plot(sorted_targets + 1, label="Target")
        #plt.yscale("log")
        plt.xlabel("Samples (sorted by target)")
        plt.ylabel(f"{LABEL_KEYS[i]}")
        plt.title(f"Predictions vs Targets for {LABEL_KEYS[i]} ({err:.2f}% err)")
        plt.legend()
        plt.show()


def rpe_box_plots(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    labels: list[str],
    ordering: list[str] | None = None,
    title: str = "Relative prediction errors",
    show: bool = True,
):
    """
    Box plots of the relative prediction error (RPE) for each label
    in the style of the wa-hls4ml benchmark.
    y-axis is symmetric log scale.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    rpe = relative_percentage_error(
        torch.from_numpy(np.array(predictions, copy=True)),
        torch.from_numpy(np.array(targets, copy=True)),
    ).numpy()

    fig, axes = plt.subplots(
        1, len(labels),
        figsize=(1.5 * len(labels), 7),
        sharey=True
    )
    box_width = 0.25
    x_min, x_max = 1 - box_width / 2, 1 + box_width / 2
    
    plot_ordering = ordering if ordering else labels
    for (ax, label) in zip(axes, plot_ordering):
        i = labels.index(label)

        box = ax.boxplot(
            rpe[:, i],
            widths=box_width,
            patch_artist=True,
            showfliers=True,
            flierprops={"markersize": 2, "alpha": 0.35},
        )
        box["boxes"][0].set_facecolor(f"C{i % 10}")
        box["boxes"][0].set_alpha(0.45)

        median_line = ax.hlines(
            np.median(rpe[:, i]),
            x_min,
            x_max,
            color="orange",
            linestyle="--",
            label="Median"
        )

        mean_line = ax.hlines(
            np.mean(rpe[:, i]),
            x_min,
            x_max,
            color="green",
            linestyle="--",
            label="Mean"
        )

        ax.set_yscale("symlog")
        ax.yaxis.set_major_locator(
            SymmetricalLogLocator(base=10, linthresh=1)
        )
        ax.grid(True, which="major", axis="y", linestyle=":")

        ax.set_xlabel(label)
        
        ax.set_xticks([])

    axes[0].set_ylabel("Relative percentage error (%)")
    fig.suptitle(title)

    fig.legend(
        handles=[median_line, mean_line],
        labels=["Median", "Mean"],
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.9, 1.0)
    )

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


def prediction_scatter_plots(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    labels: list[str],
    title: str = "Predictions versus synthesized targets",
    show: bool = True,
):
    """Paper-style per-target scatter plots with logarithmic axes."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    n_columns = 3
    n_rows = int(np.ceil(len(labels) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.2 * n_columns, 3.7 * n_rows),
        squeeze=False,
    )
    for index, (ax, label) in enumerate(zip(axes.flat, labels)):
        target = np.clip(targets[:, index], 0, None) + 1.0
        prediction = np.clip(predictions[:, index], 0, None) + 1.0
        lower = min(float(target.min()), float(prediction.min()))
        upper = max(float(target.max()), float(prediction.max()))
        ax.scatter(target, prediction, s=14, alpha=0.55)
        ax.plot([lower, upper], [lower, upper], color="red", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Synthesized")
        ax.set_ylabel("Predicted")
        ax.set_title(label)
        ax.grid(True, which="major", linestyle=":", alpha=0.45)
    for ax in axes.flat[len(labels):]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes
