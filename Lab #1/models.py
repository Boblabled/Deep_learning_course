import torch
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, AlexNet_Weights, EfficientNet_B0_Weights
from torchvision.models import alexnet

class MyModel(nn.Module):
    def __init__(self, num_classes, weights):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def my_efficient_net(num_classes, weights):
    model = efficientnet_b0(weights = weights)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    model.classifier.append(nn.Softmax(dim=1))
    return model

def my_alexnet(num_classes, weights):
    model = alexnet(weights = weights)
    out_features = model.classifier[-3].out_features
    model.classifier[-1] = (nn.Linear(out_features, num_classes))
    model.classifier.append(nn.Softmax(dim=1))
    return model


if __name__ == '__main__':
    print(my_alexnet(15, None))
    print(AlexNet_Weights.IMAGENET1K_V1.transforms())
    print(EfficientNet_B0_Weights.IMAGENET1K_V1.transforms())
    # print(my_efficient_net(15, None))