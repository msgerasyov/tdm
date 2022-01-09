import unittest

import torch

from tdm import models


class TestUnet(unittest.TestCase):
    def test_forward(self):
        X, y = _generate_sample_batch(16, 2)
        model = models.UNet(1, 2)
        with torch.no_grad():
            out = model(X)
            pred = torch.softmax(out, dim=1).argmax(dim=1)
        self.assertEqual(y.shape, pred.shape)

    def test_backward(self):
        X, y = _generate_sample_batch(16)
        model = models.UNet(1, 2)
        loss_function = torch.nn.CrossEntropyLoss()
        out = model(X)
        loss = loss_function(out, y)
        loss.backward()
        has_nan = torch.isnan(loss).any().data.numpy()
        self.assertFalse(has_nan)


def _generate_sample_batch(batch_size=16, n_classes=2):
    X = torch.randn((batch_size, 1, 572, 572))
    y = torch.randint(low=0, high=n_classes, size=(batch_size, 388, 388))
    return X, y


if __name__ == "__main__":
    unittest.main()
