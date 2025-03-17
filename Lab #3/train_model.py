import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import LetterDictionary, MyDataset, WordDictionary
from model import MyModel, initMyModel

if __name__ == '__main__':
    seq_length = 10
    batch_size = 512
    learning_rate = 0.01
    num_epochs = 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    dictionary = WordDictionary()
    # dictionary.add_data("WAP_DATA.txt")
    dictionary.add_data("data/file.txt")

    dictionary.save_dict("dict.txt")


    dataset = MyDataset("data/file.txt", dictionary, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = initMyModel(len(dictionary)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Обучение модели
    for epoch in range(num_epochs):
        for batch_idx, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            output, _ = model(x.to(device))
            loss = criterion(output.transpose(1, 2), y.to(device))
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), f"GRU_weights.pth")


