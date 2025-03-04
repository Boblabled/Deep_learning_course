from torch import nn
from torchvision.models import efficientnet_b0
from torchvision.models import alexnet

class MyModel(nn.Module):
    def __init__(self, num_classes, weights):
        super().__init__()
        self.layers_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Linear(128 * 28 * 28, 512),  # Пример для изображений 224x224
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Пример для изображений 224x224
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.layers_stack(x)

def my_efficient_net(num_classes, weights):
    model = efficientnet_b0(weights = weights)
    # model.features.requires_grad = False
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    model.classifier.append(nn.Softmax(dim=1))
    return model

def my_alexnet(num_classes, weights):
    model = alexnet(weights = weights)
    num_features = model.classifier[-1].out_features
    # model.classifier.append(nn.Dropout(0.5, inplace=False))
    model.classifier.append(nn.ReLU(inplace=True))
    model.classifier.append(nn.Linear(num_features, num_classes))
    model.classifier.append(nn.Softmax(dim=1))
    return model