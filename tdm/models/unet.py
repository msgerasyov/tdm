import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU(),
        ]
        super().__init__(*layers)


class ContractingBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        layers = [nn.MaxPool2d(2), ConvBlock(in_channels, out_channels)]
        super().__init__(*layers)


class UpConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        layers = [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 2)
        ]
        super().__init__(*layers)


class ExpansiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.convblock = ConvBlock(in_channels, out_channels)

    def forward(self, x_e, x_c):
        h_e = self.upconv(x_e)
        diff_h = x_c.size(2) - h_e.size(2)
        diff_w = x_c.size(3) - h_e.size(3)
        x_c_cropped = x_c[:, :, diff_h // 2:-diff_h // 2,
                          diff_w // 2:-diff_w // 2]
        x = torch.cat([x_c_cropped, h_e], dim=1)
        h = self.convblock(x)
        return h


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, hid_size=64):
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
        self.head = nn.Conv2d(hid_size, n_classes, 1)

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
