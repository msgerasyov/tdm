import os
from os.path import abspath, dirname, join
import unittest

import numpy as np
import torch
from tdm import datasets
from tdm.transforms import segmentation as S
from torch.utils.data import DataLoader


class TestOxfordPetDataset(unittest.TestCase):
    def setUp(self):
        self.root_dir = join(dirname(abspath(__file__)), 'data/')
        # Make sure self.root_dir is empty or doesn't exist.
        # It's going to be wiped after the tests.
        assert os.path.isdir(self.root_dir) and len(os.listdir(
            self.root_dir)) == 0 or not os.path.exists(
                self.root_dir), 'Root directory contains files'
        self.img_size = 256
        self.batch_size = 4
        self.transform = S.Compose(
            [S.Resize((self.img_size, self.img_size)),
             S.ToTensor()])


    def test_loading_data(self):
        dataset = datasets.OxfordPetDataset(
            root=self.root_dir,
            transform=self.transform,
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
                (self.batch_size, self.img_size, self.img_size),
            )
        with self.subTest():
            self.assertSequenceEqual(torch.unique(sample_masks), (1, 2, 3))

    def tearDown(self):
        # From https://docs.python.org/3/library/os.html
        # CAUTION:  This is dangerous!  For example, if top == '/', it
        # could delete all your disk files.
        for root, dirs, files in os.walk(self.root_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.root_dir)


if __name__ == "__main__":
    unittest.main()
