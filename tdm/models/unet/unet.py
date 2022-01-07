import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU()
        ]
        super().__init__(*layers)


class UpConv(nn.Sequential):
    def __init__(self, in_channels):
        out_channels = in_channels // 2
        layers = [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 2)
        ]
        super().__init__(*layers)


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, hid_size=64):
        print("Created UNet")
        super().__init__()

    def forward(self, x):
        pass
