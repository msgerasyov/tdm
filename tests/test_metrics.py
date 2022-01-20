import unittest

import torch
from tdm.utils import metrics


class TestDiceScore(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.n_categories = 2

        self.y_true_binary = torch.randint(low=0,
                                           high=2,
                                           size=(self.batch_size, 388, 388))
        self.y_pred_binary = torch.randint(low=0,
                                           high=2,
                                           size=(self.batch_size, 388, 388))
        self.y_true_multiclass = torch.randint(low=0,
                                               high=self.n_categories + 1,
                                               size=(self.batch_size, 388,
                                                     388))
        self.y_pred_multiclass = torch.randint(low=0,
                                               high=self.n_categories + 1,
                                               size=(self.batch_size, 388,
                                                     388))

    def test_binary(self):
        score = metrics.dice_score(self.y_true_binary,
                                   self.y_pred_binary).data.numpy()
        with self.subTest():
            self.assertGreaterEqual(score, 0)
        with self.subTest():
            self.assertLessEqual(score, 1)

    def test_multiclass(self):
        score = metrics.dice_score(self.y_true_multiclass,
                                   self.y_pred_multiclass,
                                   n_categories=self.n_categories).data.numpy()
        with self.subTest():
            self.assertGreaterEqual(score, 0)
        with self.subTest():
            self.assertLessEqual(score, 1)


if __name__ == "__main__":
    unittest.main()
