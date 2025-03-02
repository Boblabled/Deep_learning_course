import lightning as L
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torch import nn
import torch

class LModel(L.LightningModule):
    def __init__(self, model, labels_names, device, learning_rate=0.01):
        super().__init__()
        self.__model = model
        self.__device = device
        self.__learning_rate = learning_rate
        self.__criterion = nn.CrossEntropyLoss()

        num_classes = len(labels_names)
        self.__metrics = MetricCollection([MulticlassAccuracy(num_classes=num_classes, ),
                                           MulticlassF1Score(num_classes=num_classes,)])

        self.__train_metrics = self.__metrics.clone(postfix="/train")
        self.__valid_metrics = self.__metrics.clone(postfix="/valid")
        self.__best_val_accuracy = 0.0

        self.__all_pred, self.__all_labels, self.__all_images = [], [], []
        self.__class_names = {v: k for k, v in labels_names.items()}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.__learning_rate, momentum=0.9)
        return optimizer

    def on_train_epoch_start(self):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.__model(x.to(self.__device))
        loss = self.__criterion(pred, y.to(self.__device))
        self.__train_metrics.update(pred, y.to(self.__device))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.__train_metrics.compute())
        self.__train_metrics.reset()

    def on_validation_epoch_start(self):
        # called only if validation_step implemented
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.__model(x.to(self.__device))
        self.__valid_metrics.update(pred, y.to(self.__device))

    def on_validation_epoch_end(self):
        metrics = self.__valid_metrics.compute()
        self.log_dict(metrics)
        self.__valid_metrics.reset()

        if metrics['MulticlassAccuracy/valid'] > self.__best_val_accuracy:
            self.__best_val_accuracy = metrics['MulticlassAccuracy/valid']
            torch.save(self.__model.state_dict(), f'{self.__model.__class__.__name__}.pth')
            print(f"\nNew best model saved with accuracy: {self.__best_val_accuracy}")

    def on_fit_start(self):
        # Инициализация лучшей точности в начале обучения
        self.__best_val_accuracy = 0.0

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.__model(x.to(self.__device))
        loss = self.__criterion(outputs, y.to(self.__device))
        _, pred = torch.max(outputs, 1)
        self.__all_pred.extend(pred.cpu().numpy())
        self.__all_labels.extend(y.cpu().numpy())
        self.__all_images.extend(x.cpu())
        return loss
        # return {'pred': pred, 'labels': y}

    def on_test_end(self):
        # Вычисление точности
        accuracy = accuracy_score(self.all_labels, self.all_preds)
        print(f"Test accuracy: {accuracy * 100:.2f}%")
        self.visualize_predictions(self.__all_images, self.__all_labels, self.__all_pred, self.__class_names)

    def visualize_predictions(self, images, labels, predictions, class_names, save_path="predictions.png"):
        plt.figure(figsize=(8, 8))
        for i in range(min(16, len(images))):
            plt.subplot(4, 4, i + 1)
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

