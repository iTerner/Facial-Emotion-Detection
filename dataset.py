import torch
import numpy as np
from torch.utils.data import Dataset


class PlainDataset(Dataset):
    def __init__(self, data, upper_transform=None, lower_transform=None):
        """
        PyTorch dataset class
        Args:
            data - list(tuple(upper_img, lower_img, label))
        Return:
            image, labels
        """
        self.upper_transform = upper_transform
        self.lower_transform = lower_transform
        self.upper_imgs = [img for img, _, _ in data]
        self.lower_imgs = [img for _, img, _ in data]
        self.labels = [label for _, _, label in data]
        assert (len(self.upper_imgs) == len(self.lower_imgs)
                and len(self.lower_imgs) == len(self.labels))

    def __len__(self):
        return len(self.upper_imgs)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        labels = np.array(self.labels[index])
        labels = torch.from_numpy(labels).long()
        if self.upper_transform:
            upper_img = self.upper_transform(self.upper_imgs[index])
        if self.lower_transform:
            lower_img = self.lower_transform(self.lower_imgs[index])

        return upper_img, lower_img, labels
