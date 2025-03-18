from torch import nn


class GeneratorBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channel, out_channel, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class GeneratorDCGAN(nn.Module):
    def __init__(self, latent_dim, img_size, channels):
        super(GeneratorDCGAN, self).__init__()

        self.init_size = img_size // 2**2
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            GeneratorBlock(in_channel=128, out_channel=128),
            GeneratorBlock(in_channel=128, out_channel=64),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
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
    def __init__(self, channels, img_size):
        super().__init__()

        self.model = nn.Sequential(
            DiscriminatorBlock(channels, 16, bn=False),
            DiscriminatorBlock(16, 32),
            DiscriminatorBlock(32, 64),
            DiscriminatorBlock(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity