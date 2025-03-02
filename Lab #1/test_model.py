import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import accuracy_score

from dataset import MyDataset
from models import my_efficient_net


def visualize_predictions(images, labels, predictions, class_names, save_path="predictions.png"):
    plt.figure(figsize=(8, 8))
    for i in range(min(16, len(images))):
        plt.subplot(4, 4, i + 1)

        # Нормализация изображения для корректного отображения
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Нормализация в диапазон [0, 1]
        plt.imshow(img)
        plt.axis("off")
        true_label = int(labels[i])
        pred_label = int(predictions[i])
        color = "green" if true_label == pred_label else "red"
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", color=color, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    data_path = "data/Vegetable_Images"
    batch_size = 8

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    test_data = MyDataset(os.path.join(data_path, "test"),
                         transform=weights.transforms())
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = my_efficient_net(15, weights)
    model.load_state_dict(torch.load("BestEfficientNet.pth"))
    model.eval()

    class_names = {v: k for k, v in test_data.get_labels().items()}

    # Оценка модели
    all_preds, all_labels, all_images = [], [], []

    i = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(images.cpu())
            i += 1
            print(i)

    # Вычисление точности
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Визуализация результатов
    visualize_predictions(all_images, all_labels, all_preds, class_names, save_path="predictions_grid.png")
