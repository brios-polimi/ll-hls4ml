import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch_geometric  # noqa: F401
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "requires PyTorch and PyG")
class TrainingSmokeTests(unittest.TestCase):
    def test_json_tensorization_and_training_for_both_models(self):
        from ll_hls4ml.data.dataset import HeteroGraphDataset
        from ll_hls4ml.data.tensorize import create_graph_tensors
        from ll_hls4ml.models.registry import build
        from ll_hls4ml.training.loaders import make_loader
        from ll_hls4ml.training.loops import fit, validate_one_epoch
        from ll_hls4ml.training.targets import compute_target_z_stats

        graph = {
            "nodes": [
                {"id": 0, "type": 0, "text": "add"},
                {"id": 1, "type": 1, "text": "ap_fixed<16, 6, AP_TRN, AP_WRAP>"},
                {"id": 2, "type": 2, "text": "i32"},
                {
                    "id": 3,
                    "type": 3,
                    "text": "pragma.pipeline",
                    "features": {
                        "schema_version": ["2"],
                        "arguments_json": [
                            json.dumps({"ii": ["2"], "rewind": ["true"]})
                        ],
                    },
                },
                {
                    "id": 4,
                    "type": 4,
                    "text": "llvm.basic_block",
                    "features": {
                        "name": ["ReuseLoop"],
                        "is_source_loop": ["true"],
                    },
                },
            ],
            "links": [
                {"source": 0, "target": 0, "flow": 0, "position": 0},
                {"source": 0, "target": 1, "flow": 1, "position": 0},
                {"source": 1, "target": 0, "flow": 1, "position": 1},
                {"source": 2, "target": 0, "flow": 1, "position": 2},
                {"source": 0, "target": 0, "flow": 2, "position": 0},
                {"source": 3, "target": 0, "flow": 3, "position": 0},
                {"source": 3, "target": 1, "flow": 3, "position": 0},
                {"source": 3, "target": 4, "flow": 3, "position": 0},
                {"source": 4, "target": 0, "flow": 4, "position": 0},
                {"source": 0, "target": 4, "flow": 4, "position": 0},
            ],
        }
        labels = [100, 200, 3, 4, 1000, 10]
        instruction_vocab = {"UNK": 0, "add": 1}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            graph_dir = root / "graphs"
            archive_dir = graph_dir / "smoke" / "archive_1"
            exemplar_dir = graph_dir / "exemplar" / "archive_1"
            archive_dir.mkdir(parents=True)
            exemplar_dir.mkdir(parents=True)
            for index in range(8):
                sample = dict(graph)
                sample["labels"] = {
                    key: value + index
                    for key, value in zip(
                        (
                            "lut",
                            "ff",
                            "dsp",
                            "bram",
                            "cycles_max",
                            "interval_max",
                        ),
                        labels,
                    )
                }
                (archive_dir / f"sample_{index}.json").write_text(json.dumps(sample))
                if index < 2:
                    (exemplar_dir / f"sample_{index}.json").write_text(
                        json.dumps(sample)
                    )

            tensor_dir = root / "tensors"
            create_graph_tensors(
                graph_dir,
                tensor_dir,
                instruction_vocab,
                kernel_subset=["smoke", "exemplar"],
                n_workers=1,
            )
            dataset = HeteroGraphDataset(tensor_dir, types=["smoke"])
            self.assertTrue((tensor_dir / "labels.json").exists())
            self.assertEqual(dataset.targets.shape, (8, 6))
            y_means, y_stds = compute_target_z_stats(dataset)
            loader = make_loader(
                dataset,
                batch_size=4,
                shuffle=False,
                num_workers=0,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Smoke training device: {device}")

            for name, extra in (
                ("mlp", {}),
                ("rgcn", {"edge_pos_vocab_size": 2}),
            ):
                model = build(
                    name,
                    instruction_vocab_size=len(instruction_vocab),
                    y_means=y_means,
                    y_stds=y_stds,
                    hidden_dim=16,
                    num_layers=1,
                    dropout=0.0,
                    **extra,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                model = fit(
                    model,
                    loader,
                    loader,
                    epochs=1,
                    criterion=torch.nn.HuberLoss(),
                    optimizer=optimizer,
                    scheduler=None,
                    device=device,
                    patience=0,
                    verbose=0,
                    experiment_name=f"smoke_{name}",
                    checkpoint_dir=root / "checkpoints",
                )
                metrics = validate_one_epoch(
                    model,
                    loader,
                    torch.nn.HuberLoss(),
                    device,
                )
                self.assertTrue(torch.isfinite(metrics["r2"]).all())
                self.assertTrue((root / "checkpoints" / f"smoke_{name}_checkpoint.pt").is_file())

            vocab_path = root / "vocab.json"
            vocab_path.write_text(
                json.dumps({"vocab": instruction_vocab, "max_pos": 2})
            )
            config_path = root / "train.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "cli_smoke",
                        "model": "mlp",
                        "tensor_dir": str(tensor_dir),
                        "vocab_path": str(vocab_path),
                        "checkpoint_dir": str(root / "cli_checkpoints"),
                        "results_dir": str(root / "results"),
                        "kernel_types": ["smoke"],
                        "seed": 42,
                        "val_fraction": 0.125,
                        "test_fraction": 0.125,
                        "batch_size": 4,
                        "num_workers": 0,
                        "epochs": 1,
                        "patience": 0,
                        "hidden_dim": 16,
                        "num_layers": 1,
                        "dropout": 0.0,
                    }
                )
            )
            repository = Path(__file__).resolve().parents[1]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(repository / "src")
            environment["LL_HLS4ML_TQDM"] = "0"
            subprocess.run(
                [
                    sys.executable,
                    str(repository / "scripts" / "train.py"),
                    "--config",
                    str(config_path),
                ],
                check=True,
                cwd=repository,
                env=environment,
            )
            result_dir = root / "results" / "cli_smoke"
            result = json.loads((result_dir / "summary.json").read_text())
            self.assertEqual(
                result["sizes"],
                {"train": 6, "validation": 1, "test": 1, "exemplar": 2},
            )
            self.assertTrue((result_dir / "metrics.csv").is_file())
            self.assertTrue((result_dir / "predictions.csv").is_file())
            self.assertTrue((result_dir / "split_manifest.json").is_file())
            self.assertTrue((result_dir / "figures" / "test__rpe.png").is_file())
            self.assertTrue((result_dir / "figures" / "test__scatter.png").is_file())

            ddp_config = json.loads(config_path.read_text())
            ddp_config.update(
                {
                    "experiment_name": "cli_ddp_smoke",
                    "batch_size": 2,
                    "epochs": 2,
                }
            )
            config_path.write_text(json.dumps(ddp_config))
            ddp_environment = environment.copy()
            ddp_environment["CUDA_VISIBLE_DEVICES"] = ""
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=2",
                    str(repository / "scripts" / "train.py"),
                    "--config",
                    str(config_path),
                ],
                check=True,
                cwd=repository,
                env=ddp_environment,
                timeout=60,
            )
            self.assertTrue(
                (root / "results" / "cli_ddp_smoke" / "summary.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
