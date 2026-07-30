import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from ll_hls4ml.training.loops import fit


class TrainingResumeTests(unittest.TestCase):
    def test_resume_preserves_history_and_early_stopping_state(self):
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as temp:
            checkpoint_dir = Path(temp)
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loader = DataLoader([torch.tensor([0.0])])

            with (
                patch(
                    "ll_hls4ml.training.loops.train_one_epoch",
                    side_effect=[0.8, 0.7, 0.6, 0.5, 0.4],
                ),
                patch(
                    "ll_hls4ml.training.loops.validate_one_epoch",
                    side_effect=[
                        {"loss": 0.8, "smape": torch.tensor([50.0])},
                        {"loss": 0.7, "smape": torch.tensor([40.0])},
                        {"loss": 0.6, "smape": torch.tensor([35.0])},
                        {"loss": 0.5, "smape": torch.tensor([30.0])},
                        {"loss": 0.4, "smape": torch.tensor([25.0])},
                    ],
                ),
            ):
                fit(
                    model,
                    train_loader=loader,
                    val_loader=loader,
                    epochs=5,
                    criterion=torch.nn.MSELoss(),
                    optimizer=optimizer,
                    scheduler=None,
                    device=device,
                    patience=10,
                    verbose=0,
                    experiment_name="resume",
                    checkpoint_dir=checkpoint_dir,
                    early_stopping_metric="smape",
                )

            resumed_model = torch.nn.Linear(1, 1)
            resumed_optimizer = torch.optim.Adam(
                resumed_model.parameters(), lr=1e-3
            )
            backup = checkpoint_dir / "resume_backup.pt"

            with (
                patch(
                    "ll_hls4ml.training.loops.train_one_epoch",
                    side_effect=[0.3, 0.2],
                ),
                patch(
                    "ll_hls4ml.training.loops.validate_one_epoch",
                    side_effect=[
                        {"loss": 0.3, "smape": torch.tensor([24.0])},
                        {"loss": 0.2, "smape": torch.tensor([20.0])},
                    ],
                ),
            ):
                fit(
                    resumed_model,
                    train_loader=loader,
                    val_loader=loader,
                    epochs=7,
                    criterion=torch.nn.MSELoss(),
                    optimizer=resumed_optimizer,
                    scheduler=None,
                    device=device,
                    patience=10,
                    verbose=0,
                    experiment_name="resume",
                    checkpoint_dir=checkpoint_dir,
                    resume_from_backup=backup,
                    early_stopping_metric="smape",
                )

            self.assertEqual(
                [row["epoch"] for row in resumed_model.training_history],
                [1, 2, 3, 4, 5, 6, 7],
            )
            self.assertEqual(resumed_model.best_epoch, 7)
            self.assertEqual(resumed_model.best_metric, 20.0)


if __name__ == "__main__":
    unittest.main()
