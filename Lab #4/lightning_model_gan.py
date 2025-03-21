import os

import lightning as L
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from tbparse import SummaryReader
from IPython.display import clear_output
from torch import nn


def visualization(log_dir):
    # visualization without TensorBoard for TensorBoard logs
    clear_output()
    reader = SummaryReader(log_dir)
    df = reader.scalars.drop_duplicates()

    uniq = set(df.tag.unique())
    uniq.remove("epoch")
    uniq = list(uniq)
    uniq.sort()

    i = 0
    ax_dict = {}
    for item in uniq:
        metric = item.split("/")[0]  # log shoud have tag
        if metric not in ax_dict:
            ax_dict[metric] = i
            i += 1

    fig, axs = plt.subplots(len(ax_dict), 1, figsize=(12, 3.5 * len(ax_dict)))
    for item in uniq:
        metric = item.split("/")[0]
        if len(ax_dict) > 1:
            ax = axs[ax_dict[metric]]
        else:
            ax = axs
        sub_df = df[df["tag"] == item]
        ax.plot(sub_df.step, sub_df.value, label=item)
        ax.set_ylabel(metric)
        ax.legend()
        ax.set_xlabel("iter")

    plt.grid()
    plt.show()


def visualize_images(tensors):
    # Преобразуем тензор в формат (80, 3, 32, 32)

    # Визуализируем первые два изображения
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(2):
        tensor = tensors[i]
        if tensor.dim() == 3:
            tensor = tensor.squeeze(1)
        tensor = tensor.view(-1, 3, 32, 32)
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        img = tensor[0].cpu().detach().permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.show()

def visualise_predictions(model, prediction, N, path):
    grid = torchvision.utils.make_grid(prediction.cpu(), nrow=N, normalize=True)
    plt.figure(figsize=(N, N))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.title(f"Generated images ({model.__class__.__name__}), seed={torch.initial_seed()}")
    plt.axis("off")
    plt.savefig(path)
    plt.show()


class LGAN(L.LightningModule):
    def __init__(self, generator, discriminator, lr=0.0002, betas=(0.9, 0.999), latent_dim=10):
        super().__init__()
        self.automatic_optimization = False
        self.__generator = generator
        self.__discriminator = discriminator
        self.__criterion = nn.BCELoss()
        self.__REAL_LABEL = 1.0
        self.__FAKE_LABEL = 0.0
        self.__lr = lr
        self.__betas = betas
        self.__latent_dim = latent_dim
        self.__fixed_noise = None


    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(
            self.__generator.parameters(),
            lr=self.__lr,
            betas=self.__betas,
        )
        opt_disc = torch.optim.Adam(
            self.__discriminator.parameters(),
            lr=self.__lr,
            betas=self.__betas,
        )
        return [opt_gen, opt_disc], []

    def on_train_epoch_start(self):
        pass

    def training_step(self, batch, batch_idx):
        self.real_items, _ = batch  # for heandmade dataset
        opt_gen, opt_disc = self.optimizers()
        batch_size = self.real_items.shape[0]
        noises = torch.randn((batch_size, self.__latent_dim), dtype=torch.float32).to(self.device)
        real_label = torch.full(size=(batch_size, 1), fill_value=self.__REAL_LABEL, dtype=torch.float).to(self.device)
        fake_label = torch.full(size=(batch_size, 1), fill_value=self.__FAKE_LABEL, dtype=torch.float).to(self.device)

        # ---------------------
        # Train discriminator
        # ---------------------
        self.__discriminator.zero_grad()
        # 1. discriminator on real items
        disc_label = self.__discriminator(self.real_items)
        loss_disc_real = self.__criterion(disc_label, real_label)

        # 2. discriminator on fake items
        self.fake_items = self.__generator(noises)
        disc_label = self.__discriminator(self.fake_items)
        loss_disc_fake = self.__criterion(disc_label, fake_label)

        loss_disc = loss_disc_fake + loss_disc_real
        loss_disc.backward()

        # 3. discriminator optimizer step (on real and fake items)
        opt_disc.step()
        loss_disc = 0.5 * loss_disc_real + 0.5 * loss_disc_fake
        self.log("loss/disc", loss_disc, on_epoch=False, on_step=True)

        # ---------------------
        # Train generator
        # ---------------------
        self.__generator.zero_grad()
        self.fake_items = self.__generator(noises)
        disc_label = self.__discriminator(self.fake_items)
        loss_gen = self.__criterion(disc_label, real_label)
        loss_gen.backward()

        opt_gen.step()
        self.log("loss/gen", loss_gen, on_epoch=False, on_step=True)

    def on_train_epoch_end(self):
        # visualize_images([self.real_items, self.fake_items])
        N = 8
        if self.__fixed_noise is None:
            self.__fixed_noise = torch.randn((N**2, self.__latent_dim), dtype=torch.float32).to(self.device)
            os.makedirs(os.path.join(self.logger.log_dir, "result"), exist_ok=True)
        visualise_predictions(self.__generator, self.__generator(self.__fixed_noise), N, os.path.join(self.logger.log_dir, "result", f"Generation_result_{self.current_epoch}"))
        visualization(self.logger.log_dir)
        torch.save(self.__generator.state_dict(), f"{self.__generator.__class__.__name__}_weights.pth")
        torch.save(self.__discriminator.state_dict(), f"{self.__discriminator.__class__.__name__}_weights.pth")


    def on_validation_epoch_start(self):
        # called only if validation_step implemented
        pass

    def validation_step(self, batch, batch_idx):
       pass

    def on_validation_epoch_end(self):
       pass