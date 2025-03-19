import torch.nn as nn

class Generator(nn.Module):
    def init(self, z_dim, ngf, nc):
        super(Generator, self).init()
        self.main = nn.Sequential(
            # Вход: (z_dim, 1, 1) -> (ngf*4, 4, 4)
            nn.ConvTranspose2d(z_dim, ngf*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            # (ngf*4, 4, 4) -> (ngf*2, 8, 8)
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            # (ngf*2, 8, 8) -> (ngf, 16, 16)
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf, 16, 16) -> (nc, 32, 32)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # выход в диапазоне [-1, 1]
        )

    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def init(self, nc, ndf):
        super(Discriminator, self).init()
        self.main = nn.Sequential(
            # (nc, 32, 32) -> (ndf, 16, 16)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf, 16, 16) -> (ndf*2, 8, 8)
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2, 8, 8) -> (ndf*4, 4, 4)
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4, 4, 4) -> (1, 1, 1)
            nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Выход (batch, 1, 1, 1) -> (batch, 1)
        return self.main(x).view(-1, 1)