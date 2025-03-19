import lightning as L
import torch
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


class LGAN(L.LightningModule):
    def __init__(self, generator, discriminator, betas=(0.9, 0.999), latent_dim=10):
        super().__init__()
        self.automatic_optimization = False  # for hand made settings
        self.generator = generator
        self.discriminator = discriminator
        self.criterion = nn.BCELoss()
        self.real_label = 1.0
        self.fake_label = 0.0
        self.betas = betas
        self.latent_dim = latent_dim


    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.001,
            betas=self.betas,
        )
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=0.000025,
            betas=self.betas,
        )
        return [opt_gen, opt_disc], []

    def on_train_epoch_start(self):
        pass

    def training_step(self, batch, batch_idx):
        self.real_items, _ = batch  # for heandmade dataset
        opt_gen, opt_disc = self.optimizers()
        noises = torch.randn((self.real_items.shape[0], self.latent_dim), dtype=torch.float32).to(self.device)
        real_label = torch.full(
            size=(self.real_items.shape[0], 1),
            fill_value=self.real_label,
            dtype=torch.float,
        ).to(self.device)
        fake_label = torch.full(
            size=(self.real_items.shape[0], 1),
            fill_value=self.fake_label,
            dtype=torch.float,
        ).to(self.device)

        # ---------------------
        # Train discriminator
        # ---------------------
        self.discriminator.zero_grad()
        # 1. discriminator on real items
        disc_label = self.discriminator(self.real_items)
        loss_disc_real = self.criterion(disc_label, real_label)
        # loss_disc_real.backward()

        # 2. discriminator on fake items
        self.fake_items = self.generator(noises)
        disc_label = self.discriminator(self.fake_items)
        loss_disc_fake = self.criterion(disc_label, fake_label)
        # loss_disc_fake.backward()
        loss_disc = loss_disc_fake + loss_disc_real
        loss_disc.backward()

        # 3. discriminator optimizer step (on real and fake items)
        opt_disc.step()
        loss_disc = 0.5 * loss_disc_real + 0.5 * loss_disc_fake
        self.log("loss/disc", loss_disc, on_epoch=False, on_step=True)

        # ---------------------
        # Train generator
        # ---------------------
        self.generator.zero_grad()
        self.fake_items = self.generator(noises)
        disc_label = self.discriminator(self.fake_items)
        loss_gen = self.criterion(disc_label, real_label)
        loss_gen.backward()

        opt_gen.step()
        self.log("loss/gen", loss_gen, on_epoch=False, on_step=True)

    def on_train_epoch_end(self):
        visualize_images([self.real_items, self.fake_items])
        visualization(self.logger.log_dir)
        torch.save(self.generator.state_dict(),f"{self.generator.__class__.__name__}_weights.pth")
        torch.save(self.discriminator.state_dict(), f"{self.discriminator.__class__.__name__}_weights.pth")


    def on_validation_epoch_start(self):
        # called only if validation_step implemented
        pass

    def validation_step(self, batch, batch_idx):
       pass

    def on_validation_epoch_end(self):
       pass