import os
import unittest

import torch
from tdm import datasets
from torch.utils.data import DataLoader
from torchvision import transforms


class TestOxfordPetDataset(unittest.TestCase):
    def setUp(self):
        self.root_dir = './data/'
        # Make sure self.root_dir is empty or doesn't exist.
        # It's going to be wiped after the tests.
        assert os.path.isdir(self.root_dir) and len(os.listdir(
            self.root_dir)) == 0 or not os.path.exists(
                self.root_dir), 'Root directory contains files'
        self.img_size = 256
        self.batch_size = 4
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

    def test_loading_data(self):
        dataset = datasets.OxfordPetDataset(root=self.root_dir,
                                            transform=self.transform,
                                            target_transform=self.transform,
                                            download=True)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                drop_last=True)

        sample_imgs, sample_masks = next(iter(dataloader))

        with self.subTest():
            self.assertTupleEqual(
                tuple(sample_imgs.shape),
                (self.batch_size, 3, self.img_size, self.img_size),
            )
        with self.subTest():
            self.assertTupleEqual(
                tuple(sample_masks.shape),
                (self.batch_size, 1, self.img_size, self.img_size),
            )
        with self.subTest():
            self.assertSequenceEqual(torch.unique(sample_masks),
                                     (1.0, 2.0, 3.0))

    def tearDown(self):
        # From https://docs.python.org/3/library/os.html
        # CAUTION:  This is dangerous!  For example, if top == '/', it
        # could delete all your disk files.
        for root, dirs, files in os.walk(self.root_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir('./data/')


if __name__ == "__main__":
    unittest.main()