import torch
import torch.nn as nn
import torchvision

from tdm.models import UNet


def main():
    print("Train unet")
    model = UNet(3, 2)


if __name__ == '__main__':
    main()
