import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)

        return image, mask


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = F.resize(image, self.size)
        mask = F.resize(mask,
                        self.size,
                        interpolation=InterpolationMode.NEAREST)

        return image, mask


class RandomAffine(T.RandomAffine):
    def __init__(self,
                 degrees,
                 translate=None,
                 scale=None,
                 shear=None,
                 interpolation=InterpolationMode.NEAREST,
                 fill=0):
        super().__init__(degrees, translate, scale, shear, interpolation, fill)

    def __call__(self, image, mask):
        img_size, num_channels = self._get_image_dimensions(image)
        if isinstance(image, torch.Tensor):
            if isinstance(self.fill, (int, float)):
                self.fill = [float(self.fill)] * num_channels
            else:
                self.fill = [float(f) for f in self.fill]
        params = T.RandomAffine.get_params(self.degrees, self.translate,
                                           self.scale, self.shear, img_size)
        image = F.affine(image,
                         *params,
                         interpolation=self.interpolation,
                         fill=self.fill)
        mask = F.affine(mask,
                        *params,
                        interpolation=InterpolationMode.NEAREST)

        return image, mask

    def _get_image_dimensions(self, image):
        if isinstance(image, torch.Tensor):
            if image.ndim == 2:
                n_channels = 1
            elif image.ndim > 2:
                n_channels = image.shape[-3]
            else:
                raise TypeError("Number of dimensions should be 2 or more")
            return [image.shape[-1], image.shape[-2]], n_channels
        elif isinstance(image, Image.Image):
            if image.mode in ['L', 'I', 'F']:
                n_channels = 1
            else:
                n_channels = 3
            return list(image.size), n_channels
        else:
            raise TypeError(f'Unexpected type {type(image)}')


class ColorJitter(T.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)

    def __call__(self, image, mask):
        image = super().forward(image)

        return image, mask


class ConvertImageDtype:
    def __init__(self, dtype) -> None:
        self.dtype = dtype

    def __call__(self, image, mask):
        image = F.convert_image_dtype(image, self.dtype)

        return image, mask


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = F.normalize(image, self.mean, self.std)

        return image, mask


class ToTensor:
    def __call__(self, image, mask):
        image = F.to_tensor(image)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask
