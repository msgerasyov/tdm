import unittest

import numpy as np
import torch
from PIL import Image
from tdm.transforms import segmentation as S


class TestSegmentationTransforms(unittest.TestCase):
    def setUp(self):
        self.transforms = [
            S.Resize((256, 256)),
            S.RandomAffine(15, translate=(0.2, 0.2), scale=(0.75, 1.25)),
            S.ColorJitter(0.3, 0.3),
            S.ToTensor(),
            S.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]

    def test_tensor(self):
        transform = S.Compose(self.transforms[:-2])
        image = torch.as_tensor(
            np.random.normal(loc=0, scale=1, size=(3, 355, 510)))
        mask = torch.as_tensor(
            np.random.randint(low=0, high=2, size=(1, 355, 510)))
        image, mask = transform(image, mask)
        with self.subTest():
            self.assertSequenceEqual(image.shape, (3, 256, 256))
        with self.subTest():
            self.assertSequenceEqual(mask.shape, (1, 256, 256))

    def test_image(self):
        transform = S.Compose(self.transforms)
        image = Image.fromarray(np.random.normal(loc=0,
                                                 scale=1,
                                                 size=(355, 510, 3)),
                                mode='RGB')
        mask = Image.fromarray(np.random.randint(low=0,
                                                 high=2,
                                                 size=(355, 510)),
                               mode='I')
        image, mask = transform(image, mask)
        with self.subTest():
            self.assertSequenceEqual(image.shape, (3, 256, 256))
        with self.subTest():
            self.assertSequenceEqual(mask.shape, (256, 256))


if __name__ == "__main__":
    unittest.main()