import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*layers)


class ContractingBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        layers = [nn.MaxPool2d(2), ConvBlock(in_channels, out_channels)]
        super().__init__(*layers)


class ExpansiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.convblock = ConvBlock(in_channels, out_channels)

    def forward(self, x_e, x_c):
        h_e = self.upconv(x_e)
        x = torch.cat([x_c, h_e], dim=1)
        h = self.convblock(x)
        return h


class UNet(nn.Module):
    """Implements a modified version of UNet model

    :param in_channels: Number of channels in the input image
    :type in_channels: int
    :param out_channels: Number of channels produced by the model
    :type out_channels: int
    :param hid_size: Number of channels at the output of the first conv block
    type hid_size: int, optional
    """
    def __init__(self, in_channels, out_channels, hid_size=64):
        super().__init__()
        self.cblock0 = ConvBlock(in_channels, hid_size)
        self.cblock1 = ContractingBlock(hid_size, hid_size * 2)
        self.cblock2 = ContractingBlock(hid_size * 2, hid_size * 4)
        self.cblock3 = ContractingBlock(hid_size * 4, hid_size * 8)
        self.cblock4 = ContractingBlock(hid_size * 8, hid_size * 16)
        self.eblock3 = ExpansiveBlock(hid_size * 16, hid_size * 8)
        self.eblock2 = ExpansiveBlock(hid_size * 8, hid_size * 4)
        self.eblock1 = ExpansiveBlock(hid_size * 4, hid_size * 2)
        self.eblock0 = ExpansiveBlock(hid_size * 2, hid_size)
        self.head = nn.Conv2d(hid_size, out_channels, 1)

    def forward(self, x):
        h_c0 = self.cblock0(x)
        h_c1 = self.cblock1(h_c0)
        h_c2 = self.cblock2(h_c1)
        h_c3 = self.cblock3(h_c2)
        h_c4 = self.cblock4(h_c3)
        h_e3 = self.eblock3(h_c4, h_c3)
        h_e2 = self.eblock2(h_e3, h_c2)
        h_e1 = self.eblock1(h_e2, h_c1)
        h_e0 = self.eblock0(h_e1, h_c0)
        out = self.head(h_e0)
        return out
