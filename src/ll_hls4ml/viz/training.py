"""Training evaluation plots."""

import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator
import torch
import numpy as np

from ll_hls4ml.training.targets import denormalize_target
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
    predictions: torch.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    labels: list[str],
    ordering: list[str] = None,
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
        
    rpe = (targets - predictions) / (targets + 1.0) * 100

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

        ax.boxplot(rpe[:, i], widths=box_width)

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

    axes[0].set_ylabel("Relative Percent Error")
    fig.suptitle("GNN Prediction Errors on Test Set")

    fig.legend(
        handles=[median_line, mean_line],
        labels=["Median", "Mean"],
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.9, 1.0)
    )

    #plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.tight_layout()
    plt.show()

        