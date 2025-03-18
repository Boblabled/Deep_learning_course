from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from gan_model import GeneratorGAN, DiscriminatorGAN
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

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    latent_dim = 100
    hidden_dim =32*32
    generator = GeneratorGAN(latent_space=latent_dim, hidden_dim=hidden_dim)
    discriminator = DiscriminatorGAN(hidden_dim=hidden_dim)

    num_epochs = 10
    pl_model = LGAN(generator, discriminator, batch_size, latent_dim=latent_dim)
    trainer = L.Trainer(
        max_epochs=num_epochs,
        logger=L.pytorch.loggers.TensorBoardLogger(save_dir="./log_gan/"),
    )

    trainer.fit(model=pl_model, train_dataloaders=train_loader)