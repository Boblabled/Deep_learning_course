import os
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import lightning as L
from torchvision import transforms
from torchvision.models import AlexNet_Weights, EfficientNet_B0_Weights

from dataset import MyDataset
from lightning_model import LModel
from models import my_alexnet, my_efficient_net, MyModel


def train_test_model(model, epochs, batch_size, data_path,  weights=None, transform=None, learning_rate=0.01):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    train_data = MyDataset(os.path.join(data_path, "train"),
                           transform=transform)
    labels_names = train_data.get_labels()
    valid_data = MyDataset(os.path.join(data_path, "validation"),
                           transform=transform,
                           labels=labels_names)
    test_data = MyDataset(os.path.join(data_path, "test"),
                          transform=transform,
                          labels=labels_names)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                 persistent_workers=True)

    # num_classes = len(labels_names)
    model = model(len(labels_names), weights=weights)

    logger = TensorBoardLogger(save_dir="logs", name=model.__class__.__name__)
    lit_model = LModel(model, labels_names, device, learning_rate=learning_rate)
    trainer = L.Trainer(max_epochs=epochs, logger=logger)

    trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(model=lit_model, dataloaders=test_dataloader)


if __name__ == '__main__':
    L.seed_everything(100)
    EPOCHS = 10
    BATCH_SIZE = 8
    NUM_WORKERS = 15
    DATA_PATH = "data/Vegetable_Images"

    torch.set_float32_matmul_precision('high')

    train_test_model(my_efficient_net, EPOCHS, BATCH_SIZE, DATA_PATH,
                     weights=EfficientNet_B0_Weights.IMAGENET1K_V1,
                     transform=EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()) # Test accuracy: 99.73%
    # train_test_model(my_alexnet, EPOCHS, BATCH_SIZE, DATA_PATH,
    #                  weights=AlexNet_Weights.IMAGENET1K_V1,
    #                  transform=AlexNet_Weights.IMAGENET1K_V1.transforms())
    # train_test_model(MyModel, EPOCHS, BATCH_SIZE, DATA_PATH, transform=EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()) # Test accuracy: 96.97%

