import hashlib
import os
import tarfile
import urllib

import requests
import torch
from torch.utils.data import Dataset


class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset
    link: https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    _imgs_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    _imgs_md5 = "5c4f3ee8e5d25df40f4fd59a7f44e54c"
    _masks_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    _masks_md5 = "95a8c909bbe2e81eed6a22bccdf3f68f"

    def __init__(self, root, download=True):
        self.root = root
        self.download = download
        self._imgs_tar_path = os.path.join(self.root,
                                           self._imgs_url.split('/')[-1])
        self._masks_tar_path = os.path.join(self.root,
                                            self._masks_url.split('/')[-1])
        self._imgs_path = os.path.join(self.root, 'images')
        self._masks_path = os.path.join(self.root, 'annotations')

        if self.download:
            self._download()
        else:
            self._chech_and_extract(self._imgs_path, self._imgs_md5)
            self._chech_and_extract(self._masks_path, self._masks_md5)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def _download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self._download_tar(self._imgs_url, self._imgs_tar_path, self._imgs_md5)
        self._download_tar(self._masks_url, self._masks_path, self._masks_md5)

    def _download_tar(self, tar_url, tar_path, tar_md5):
        if os.path.exists(tar_path) and os.path.isfile(tar_path):
            print(f"Found existing tar file at {tar_path}")

        else:
            urllib.request.urlretrieve(tar_url, tar_path)

        self._chech_and_extract(tar_path, tar_md5)

    def _check_integrity(self, tar_path, ref_md5):
        md5_hash = hashlib.md5()
        with open(tar_path, 'rb') as f:
            md5_hash.update(f.read())
            md5 = md5_hash.hexdigest()
            if md5 != ref_md5:
                return False
            else:
                return True

    def _extract_tar(self, tar_path):
        with tarfile.open(tar_path, 'r:gz') as f:
            f.extractall()

    def _chech_and_extract(self, tar_path, ref_md5):
        if self._check_integrity(tar_path, ref_md5):
            print(f'Extracting to {self.root}')
            self._extract_tar(tar_path)
        else:
            raise RuntimeError(f"Hash mismatch for {tar_path}")

if __name__ == "__main__":
    dataset = OxfordPetDataset("./", True)
