import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

from gan_model import GeneratorGAN, DiscriminatorGAN
from dcgan_model import GeneratorDCGAN, DiscriminatorDCGAN, ImprovedGenerator

def visualise_predictions(model, prediction, N, path):
    grid = torchvision.utils.make_grid(prediction, nrow=N, normalize=True)
    plt.figure(figsize=(N, N))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.title(f"Generated images ({model.__class__.__name__}), seed={torch.initial_seed()}")
    plt.axis("off")
    plt.savefig(path)
    plt.show()


if __name__ == '__main__':
    N = 8

    LATENT_DIM = 256
    IMG_SIZE = 32
    BASE_CHANNELS = 512

    # generator = GeneratorGAN(latent_space=latent_dim, hidden_dim=img_size ** 2)
    # discriminator = DiscriminatorGAN(hidden_dim=img_size ** 2)
    # generator = GeneratorDCGAN(LATENT_DIM, IMG_SIZE, 3, BASE_CHANNELS)
    generator = ImprovedGenerator(LATENT_DIM, IMG_SIZE, 3, BASE_CHANNELS)
    # generator.load_state_dict(torch.load("DestGeneratorDCGAN_weights.pth"))
    # generator.load_state_dict(torch.load("GeneratorDCGAN_weights.pth"))
    generator.load_state_dict(torch.load("ImprovedGenerator_weights_100_epoch.pth"))

    torch.manual_seed(100)
    z = torch.randn(N**2, LATENT_DIM)

    with torch.no_grad():
        generated_images = generator(z)

    generated_images = generated_images.cpu()

    grid = torchvision.utils.make_grid(generated_images, nrow=N, normalize=True)
    plt.figure(figsize=(N, N))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.title(f"Generated images ({generator.__class__.__name__}), seed={torch.initial_seed()}")
    plt.axis("off")
    plt.show()


    # generated_images = np.transpose(generated_images, (0, 2, 3, 1))
    #
    # generated_images = (generated_images + 1) / 2
    # generated_images = np.clip(generated_images, 0, 1)
    #
    # fig, axes = plt.subplots(N, N, figsize=(10, 10))
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(generated_images[i])
    #     ax.axis('off')
    # plt.tight_layout()
    # plt.show()