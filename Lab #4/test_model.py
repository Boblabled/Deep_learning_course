import numpy as np
import torch
from matplotlib import pyplot as plt

from gan_model import GeneratorGAN, DiscriminatorGAN
from dcgan_model import GeneratorDCGAN, DiscriminatorDCGAN


if __name__ == '__main__':
    N = 8

    LATENT_DIM = 256
    IMG_SIZE = 32
    BASE_CHANNELS = 512

    # generator = GeneratorGAN(latent_space=latent_dim, hidden_dim=img_size ** 2)
    # discriminator = DiscriminatorGAN(hidden_dim=img_size ** 2)
    generator = GeneratorDCGAN(LATENT_DIM, IMG_SIZE, 3, BASE_CHANNELS)
    discriminator = DiscriminatorDCGAN(IMG_SIZE, 3, BASE_CHANNELS)
    # generator.load_state_dict(torch.load("DestGeneratorDCGAN_weights.pth"))
    # discriminator.load_state_dict(torch.load("BestDiscriminatorDCGAN_weights.pth"))
    generator.load_state_dict(torch.load("GeneratorDCGAN_weights.pth"))
    discriminator.load_state_dict(torch.load("DiscriminatorDCGAN_weights.pth"))

    z = torch.randn(N**2, LATENT_DIM)

    with torch.no_grad():
        generated_images = generator(z)

    generated_images = generated_images.cpu().numpy()
    generated_images = np.transpose(generated_images, (0, 2, 3, 1))

    generated_images = (generated_images + 1) / 2
    generated_images = np.clip(generated_images, 0, 1)

    fig, axes = plt.subplots(N, N, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()