import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from gan_model import GeneratorGAN, DiscriminatorGAN
from dcgan_model import GeneratorDCGAN, DiscriminatorDCGAN
from dim_models import Generator, Discriminator
from lightning_model_gan import LGAN
import lightning as L

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # Нормализация значений пикселей в диапазон [-1, 1] для tanh
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    BATCH_SIZE = 512
    NUM_WORKERS = 20
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True)

    LATENT_DIM = 256
    IMG_SIZE = 32
    FEATURES = 512
    # generator = GeneratorGAN(latent_space=latent_dim, hidden_dim=img_size ** 2)
    # discriminator = DiscriminatorGAN(hidden_dim=img_size ** 2)
    generator = GeneratorDCGAN(LATENT_DIM, IMG_SIZE, 3, FEATURES)
    discriminator = DiscriminatorDCGAN(IMG_SIZE, 3, FEATURES)
    # generator = Generator(LATENT_DIM, IMG_SIZE, 3)
    # discriminator = Discriminator(3, IMG_SIZE)

    NUM_EPOCHS = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pl_model = LGAN(generator, discriminator, lr=0.0002, betas=(0.5, 0.999), latent_dim=LATENT_DIM).to(device)
    trainer = L.Trainer(
        max_epochs=NUM_EPOCHS,
        logger=L.pytorch.loggers.TensorBoardLogger(save_dir="./log_gan/"),
    )

    trainer.fit(model=pl_model, train_dataloaders=train_loader)