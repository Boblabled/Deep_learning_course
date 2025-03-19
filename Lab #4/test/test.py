import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from head import get_device, download_dataset

from head.DCGAN import Generator, Discriminator


def train(num_epochs: int = 50, model_save: bool = True):
    device = get_device()
    print('вычислитель: ', device)

    # Папка для сохранения результатов
    if model_save:
        os.makedirs("checkpoints", exist_ok=True)

    # -----------------------------
    # 1. Загрузка датасета CIFAR-10
    # -----------------------------

    train_loader = download_dataset()

    # -----------------------------------
    # 2. Гиперпараметры и настройки
    # -----------------------------------

    num_epochs = num_epochs

    z_dim = 128  # размер вектора шумa. Чем больше z_dim, тем более «богатым» может быть пространство шума — оно способно генерировать более разнообразные образы
    ngf = 128  # "количество фич" для генератора. Чем больше ngf, тем больше параметров у генератора, и тем потенциально более детальные и качественные изображения он может генерировать
    ndf = 128  # "количество фич" для дискриминатора.Дискриминатор с большим ndf может лучше различать детали (т. е. «мощнее» в распознавании фейков).

    nc = 3  # каналов (RGB)

    lr = 0.0002  # классическое рекомендованное значение для DCGAN
    beta1 = 0.5  # Adam betas
    beta2 = 0.999

    # Label smoothing: реальный лейбл = 0.9 (вместо 1.0)
    REAL_LABEL_SM = 0.95
    FAKE_LABEL = 0.0  # фейк = 0

    # -----------------------------------
    # 3. Определяем DCGAN-архитектуру
    # ----------------------------------

    # Создаем модели
    generator = Generator(z_dim, ngf, nc).to(device)
    discriminator = Discriminator(nc, ndf).to(device)

    # -----------------------------------
    # 5. Инициализация весов (DCGAN-style)
    # -----------------------------------
    def weights_init(m):
        classname = m.

        class .name

        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # -----------------------------------
    # 6. Функции потерь и оптимизаторы
    # -----------------------------------
    criterion = nn.BCELoss()
    optim_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optim_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    # Фиксированный шум (для мониторинга качества)
    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

    # -----------------------------------
    # 7. Цикл обучения
    # -----------------------------------
    D_losses = []
    G_losses = []

    print("Начало обучения")

    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(train_loader):
            # ----------------------
            # Обновление D
            # ----------------------
            discriminator.zero_grad()

            real_images = real_images.to(device)
            b_size = real_images.size(0)

            # Метки (label smoothing: реальные -> 0.9)
            real_labels = torch.full((b_size, 1), REAL_LABEL_SM, device=device)
            fake_labels = torch.full((b_size, 1), FAKE_LABEL, device=device)

            # 1) Лосс D на реальном батче
            output_real = discriminator(real_images)
            d_loss_real = criterion(output_real, real_labels)

            # 2) Лосс D на фейковом батче
            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(output_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optim_d.step()

            # ----------------------
            # Обновление G
            # ----------------------
            generator.zero_grad()
            # Генератор хочет, чтобы дискриминатор счёл фейк за реальное
            output_fake_for_g = discriminator(fake_images)
            g_loss = criterion(output_fake_for_g, real_labels)  # используем те же "real_labels" (0.9)

            g_loss.backward()
            optim_g.step()

            # Сохраняем значения лоссов для логов
            D_losses.append(d_loss.item())
            G_losses.append(g_loss.item())

            if (i + 1) % 100 == 0:
                print(f"[Epoch {epoch + 1}/{num_epochs} | Step {i + 1}/{len(train_loader)}] "
                      f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

            # Сохраняем чекпоинты (например, каждые 10 эпох)
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"checkpoints/generator_epoch_{epoch + 1}.pth")
            torch.save(discriminator.state_dict(), f"checkpoints/discriminator_epoch_{epoch + 1}.pth")

        print("Training finished!")

        # -----------------------------------
        # 8. Итоговая визуализация
        # -----------------------------------
        # Генерируем изображения из фиксированного шума
        with torch.no_grad():
            fake_images_final = generator(fixed_noise).cpu()  # shape: (64, 3, 32, 32)

        # Создаём коллаж
        grid = torchvision.utils.make_grid(fake_images_final, nrow=8, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        plt.title("Final generated images (DCGAN)")
        plt.axis("off")
        plt.show()

        # При желании можно также посмотреть графики лоссов
        plt.figure(figsize=(10, 4))
        plt.title("DCGAN Losses")
        plt.plot(D_losses, label="D loss")
        plt.plot(G_losses, label="G loss")
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.legend()
        plt.show()

if __name__ == "main":
    train()