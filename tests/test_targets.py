import unittest

import torch

from ll_hls4ml.training.targets import apply_hurdle_prediction


class HurdlePredictionTests(unittest.TestCase):
    def test_threshold_mode_emits_exact_zeros(self):
        y_means = torch.zeros(6)
        y_stds = torch.ones(6)
        prediction = torch.zeros((2, 8))
        prediction[:, 2:4] = torch.log1p(torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        prediction[:, 6:] = torch.tensor([[-1.0, 1.0], [1.0, -1.0]])

        normalized = apply_hurdle_prediction(
            prediction,
            y_means,
            y_stds,
            mode="threshold",
        )
        raw = torch.expm1(normalized)

        self.assertEqual(raw[0, 2].item(), 0.0)
        self.assertEqual(raw[1, 3].item(), 0.0)
        self.assertGreater(raw[0, 3].item(), 0.0)
        self.assertGreater(raw[1, 2].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
