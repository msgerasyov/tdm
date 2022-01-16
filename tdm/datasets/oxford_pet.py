import hashlib
import os
import tarfile
import urllib

import requests
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


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
            self._download_dataset()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def _download_dataset(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self._download_url(self._imgs_url, self._imgs_tar_path)
        self._check_and_extract_tar(self._imgs_tar_path, self._imgs_md5)
        self._download_url(self._masks_url, self._masks_tar_path)
        self._check_and_extract_tar(self._masks_tar_path, self._masks_md5)

    def _pb_update_hook(self, pb):
        def update_to(b=1, bsize=1, tsize=None):
            if tsize is not None:
                pb.total = tsize
            pb.update(b * bsize - pb.n)

        return update_to

    def _download_url(self, url, path):
        if os.path.exists(path) and os.path.isfile(path):
            print(f"Found existing tar file at {path}")
        else:
            with tqdm(unit='B',
                      unit_scale=True,
                      unit_divisor=1024,
                      miniters=1,
                      desc=path) as pb:
                urllib.request.urlretrieve(url,
                                           path,
                                           reporthook=self._pb_update_hook(pb))

    def _check_integrity(self, file_path, ref_md5):
        md5_hash = hashlib.md5()
        with open(file_path, 'rb') as f:
            md5_hash.update(f.read())
            md5 = md5_hash.hexdigest()
            if md5 != ref_md5:
                return False
            else:
                return True

    def _extract_tar(self, tar_path, extraction_path=None):
        if not extraction_path:
            extraction_path = self.root

        with tarfile.open(tar_path, 'r:gz') as f:
            f.extractall(path=extraction_path)

    def _check_and_extract_tar(self, tar_path, ref_md5, extraction_path=None):
        if self._check_integrity(tar_path, ref_md5):
            print(f'Extracting to {self.root}')
            self._extract_tar(tar_path, extraction_path)
        else:
            raise RuntimeError(f"Hash mismatch for {tar_path}")


if __name__ == "__main__":
    dataset = OxfordPetDataset("./data/", True)
