import argparse

import torch
import torch.nn as nn
import torchvision
from tdm.models import UNet
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train UNet on Pascal VOC dataset')
    parser.add_argument('--batch-size', default=32, metavar='BS', type=int)
    parser.add_argument('--data-dir',
                        default='./data/',
                        metavar='DIR',
                        help='Directory to store the dataset')

    return parser.parse_args()


def main():
    args = parse_args()
    train_image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Pad((5 * 2**5) // 2, padding_mode='reflect'),
        transforms.ToTensor(),
    ])
    train_target_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    train_dataset = VOCSegmentation(root=args.data_dir,
                                    image_set='train',
                                    download=True,
                                    transform=train_image_transform,
                                    target_transform=train_target_transform)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0)
    model = UNet(in_channels=3, n_classes=20)
    sample_imgs, sample_masks = next(iter(train_dataloader))
    sample_out = model(sample_imgs)
    print(sample_imgs.shape)
    print(sample_out.shape)
    print(sample_masks.shape)


if __name__ == '__main__':
    main()
