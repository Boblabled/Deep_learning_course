from torch import nn
from torchvision.models import efficientnet_b0
from torchvision.models import alexnet
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from torchvision.models import AlexNet_Weights

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