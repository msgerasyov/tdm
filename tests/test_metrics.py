import unittest

import torch
from tdm.utils import metrics


class TestDiceScore(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.n_categories = 2

    def test_binary(self):
        y_true = torch.randint(low=0, high=2, size=(self.batch_size, 388, 388))
        y_pred = torch.randint(low=0, high=2, size=(self.batch_size, 388, 388))
        score = metrics.dice_score(y_true, y_pred,
                                   from_logits=False).data.numpy()
        with self.subTest():
            self.assertGreaterEqual(score, 0)
        with self.subTest():
            self.assertLessEqual(score, 1)

    def test_binary_from_logits(self):
        y_true = torch.randint(low=0, high=2, size=(self.batch_size, 388, 388))
        logits = torch.randn(size=(self.batch_size, 1, 388, 388))
        score = metrics.dice_score(y_true, logits,
                                   from_logits=True).data.numpy()
        with self.subTest():
            self.assertGreaterEqual(score, 0)
        with self.subTest():
            self.assertLessEqual(score, 1)

    def test_multiclass(self):
        y_true = torch.randint(low=0,
                               high=self.n_categories + 1,
                               size=(self.batch_size, 388, 388))
        y_pred = torch.randint(low=0,
                               high=self.n_categories + 1,
                               size=(self.batch_size, 388, 388))
        score = metrics.dice_score(
            y_true, y_pred, from_logits=False,
            n_categories=self.n_categories).data.numpy()
        with self.subTest():
            self.assertGreaterEqual(score, 0)
        with self.subTest():
            self.assertLessEqual(score, 1)

    def test_multiclass_from_logits(self):
        y_true = torch.randint(low=0,
                               high=self.n_categories + 1,
                               size=(self.batch_size, 388, 388))
        logits = torch.randn(size=(self.batch_size, self.n_categories + 1, 388,
                                   388))
        score = metrics.dice_score(
            y_true, logits, from_logits=True,
            n_categories=self.n_categories).data.numpy()
        with self.subTest():
            self.assertGreaterEqual(score, 0)
        with self.subTest():
            self.assertLessEqual(score, 1)


if __name__ == "__main__":
    unittest.main()
