import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from ll_hls4ml.training.loops import fit, train_one_epoch


class TrainingResumeTests(unittest.TestCase):
    def test_default_clipping_limits_gradient_norm(self):
        class ToyModel(torch.nn.Linear):
            def __init__(self):
                super().__init__(1, 1, bias=False)
                self.register_buffer("y_means", torch.zeros(1))
                self.register_buffer("y_stds", torch.ones(1))

            def forward(self, batch):
                return super().forward(batch.x)

        class ToyBatch:
            def __init__(self):
                self.x = torch.tensor([[100.0]])
                self.y = torch.tensor([[0.0]])
                self.num_graphs = 1

            def to(self, *_args, **_kwargs):
                return self

        model = ToyModel()
        model.weight.data.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        _, profile = train_one_epoch(
            model,
            [ToyBatch()],
            torch.nn.MSELoss(),
            optimizer,
            torch.device("cpu"),
            return_profile=True,
            gradient_clip_norm=0.1,
        )

        self.assertGreater(profile["mean_pre_clip_gradient_norm"], 0.1)
        self.assertLessEqual(float(model.weight.grad.norm()), 0.100001)

    def test_plateau_scheduler_uses_validation_metric(self):
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as temp:
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            loader = DataLoader([torch.tensor([0.0])])
            with (
                patch(
                    "ll_hls4ml.training.loops.train_one_epoch",
                    return_value=0.5,
                ),
                patch(
                    "ll_hls4ml.training.loops.validate_one_epoch",
                    side_effect=[
                        {"loss": 0.5, "smape": torch.tensor([20.0])},
                        {"loss": 0.5, "smape": torch.tensor([20.0])},
                    ],
                ),
            ):
                fit(
                    model,
                    train_loader=loader,
                    val_loader=loader,
                    epochs=2,
                    criterion=torch.nn.MSELoss(),
                    optimizer=optimizer,
                    scheduler=None,
                    device=device,
                    patience=0,
                    verbose=0,
                    experiment_name="scheduler",
                    checkpoint_dir=temp,
                    early_stopping_metric="smape",
                    lr_scheduler_patience=0,
                    lr_scheduler_factor=0.5,
                )

            self.assertEqual(model.training_history[-1]["learning_rate"], 0.1)
            self.assertEqual(model.training_history[-1]["next_learning_rate"], 0.05)
            checkpoint = torch.load(
                Path(temp) / "scheduler_checkpoint.pt", weights_only=True
            )
            self.assertIsNotNone(checkpoint["scheduler"])

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
