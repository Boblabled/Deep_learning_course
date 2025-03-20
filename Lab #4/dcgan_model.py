import torch
from torch import nn


class GeneratorBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class GeneratorDCGAN(nn.Module):
    def __init__(self, latent_dim, img_size, channels, features=128):
        super().__init__()
        self.init_size = img_size // 2 ** 2
        self.l1 = nn.Sequential(nn.Linear(latent_dim, features * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(features),
            GeneratorBlock(in_channel=features, out_channel=features),
            GeneratorBlock(in_channel=features, out_channel=features // 2),
            nn.Conv2d(features // 2, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bn=True):
        super().__init__()
        block = [
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        ]
        if bn:
            block.append(nn.BatchNorm2d(out_channel, 0.8))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class DiscriminatorDCGAN(nn.Module):
    def __init__(self, img_size, channels, features=128):
        super().__init__()

        self.model = nn.Sequential(
            DiscriminatorBlock(channels, features // 8, bn=False),
            DiscriminatorBlock(features // 8, features // 4),
            DiscriminatorBlock(features // 4, features // 2),
            DiscriminatorBlock(features // 2, features),
        )

        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(features * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity