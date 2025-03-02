import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch import nn
import lightning as L
from torchvision.models import AlexNet_Weights, EfficientNet_B0_Weights

from dataset import MyDataset
from lightning_model import LModel
from models import my_alexnet, my_efficient_net


def train_test_model(model, weights, epochs, batch_size, data_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    train_data = MyDataset(os.path.join(data_path, "train"),
                           transform=weights.transforms())
    labels_names = train_data.get_labels()
    valid_data = MyDataset(os.path.join(data_path, "validation"),
                           transform=weights.transforms(),
                           labels=labels_names)
    test_data = MyDataset(os.path.join(data_path, "test"),
                          transform=weights.transforms(),
                          labels=labels_names)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                 persistent_workers=True)

    # num_classes = len(labels_names)
    model = model(len(labels_names), weights)

    lit_model = LModel(model, labels_names, device)
    trainer = L.Trainer(max_epochs=epochs)

    trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(model=lit_model, dataloaders=test_dataloader)


if __name__ == '__main__':
    L.seed_everything(100)
    EPOCHS = 10
    BATCH_SIZE = 8
    NUM_WORKERS = 7
    DATA_PATH = "data/Vegetable_Images"

    torch.set_float32_matmul_precision('high')

    train_test_model(my_efficient_net, EfficientNet_B0_Weights.IMAGENET1K_V1, EPOCHS, BATCH_SIZE, DATA_PATH)
    train_test_model(my_alexnet, AlexNet_Weights.IMAGENET1K_V1, EPOCHS, BATCH_SIZE, DATA_PATH)

