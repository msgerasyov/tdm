import unittest

import torch
from tdm import models


class TestUnet(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.n_categories = 3
        self.X = torch.randn((self.batch_size, 1, 572, 572))
        self.y = torch.randint(low=0,
                               high=self.n_categories + 1,
                               size=(self.batch_size, 388, 388))
        self.model = models.UNet(1, self.n_categories + 1)

    def test_forward(self):
        with torch.no_grad():
            out = self.model(self.X)
            pred = torch.softmax(out, dim=1).argmax(dim=1)
        self.assertSequenceEqual(self.y.shape, pred.shape, seq_type=torch.Size)

    def test_backward(self):
        loss_function = torch.nn.CrossEntropyLoss()
        out = self.model(self.X)
        loss = loss_function(out, self.y)
        loss.backward()
        has_nan = torch.isnan(loss).any().data.numpy()
        self.assertFalse(has_nan)


if __name__ == "__main__":
    unittest.main()
