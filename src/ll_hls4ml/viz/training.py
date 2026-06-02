"""Training evaluation plots."""

import matplotlib.pyplot as plt
import torch

from ll_hls4ml.training.targets import to_luts
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
            preds = to_luts(
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

        plt.figure(figsize=figsize)
        plt.plot(sorted_preds, label="Prediction")
        plt.plot(sorted_targets, label="Target")
        plt.xlabel("Samples (sorted by target)")
        plt.ylabel(f"{LABEL_KEYS[i]}")
        plt.title(f"Predictions vs Targets for {LABEL_KEYS[i]}")
        plt.legend()
        plt.show()
