from __future__ import print_function, absolute_import
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        if self.labels is not None:
            label= self.labels[index]
        #img = Image.fromarray(img.astype('uint8'), 'RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            label = torch.FloatTensor(label)
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.images)