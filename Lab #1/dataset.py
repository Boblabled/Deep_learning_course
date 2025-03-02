import os
from torch.utils.data import Dataset
from torchvision.io import read_image

class MyDataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None, labels={}):
        self.__labels = labels
        self.__img_labels = []
        self.__img_paths = []
        for i, sub_dir in enumerate(os.listdir(path)):
            if sub_dir not in self.__labels:
              self.__labels[sub_dir] = len(self.__labels)
            sub_dir_path = os.path.join(path, sub_dir)
            for image_path in os.listdir(sub_dir_path):
                self.__img_labels.append(i)
                self.__img_paths.append(os.path.join(sub_dir_path, image_path))
        self.__transform = transform
        self.__target_transform = target_transform

    def __len__(self):
        return len(self.__img_labels)

    def __getitem__(self, idx):
        image = read_image(self.__img_paths[idx])
        label = self.__img_labels[idx]
        if self.__transform:
            image = self.__transform(image)
        if self.__target_transform:
            label = self.__target_transform(label)
        return image, label

    def get_labels(self):
        return self.__labels