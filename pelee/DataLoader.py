from PIL import Image
import cv2 as cv
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy import io as iomat
from skimage import io
import torch
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Self_dataset(Dataset):
    def __init__(self, root_dir=None, transform=None, size=(32, 32), feature_size=16):
        self.root_yes = os.path.join(root_dir, 'yes')
        self.root_no = os.path.join(root_dir, 'no')
        self.yes_image = os.listdir(self.root_yes)
        self.no_image = os.listdir(self.root_no)
        self.transform = transform
        self.size = size
        self.feature_size = feature_size

    def __len__(self):
        return len(self.yes_image) + len(self.no_image)

    def __getitem__(self, item):
        sample = []
        if item < len(self.yes_image):
            path = os.path.join(self.root_yes, self.yes_image[item])
            images = os.listdir(path)
            images.sort()
            for i in range(min(8,len(images))):
                image = io.imread(os.path.join(path, images[i]))
                image = torch.tensor(image, dtype=torch.float,device='cpu',requires_grad=True)
                image = image.permute(2, 0, 1)
                sample.append(image.unsqueeze(0))
            label = 0
        else:
            path = os.path.join(self.root_no, self.no_image[item - len(self.yes_image)])
            images = os.listdir(path)
            images.sort()
            for i in range(min(8,len(images))):
                image = io.imread(os.path.join(path, images[i]))
                image = torch.tensor(image, dtype=torch.float,device ='cpu',requires_grad=True)
                image = image.permute(2, 0, 1)
                sample.append(image.unsqueeze(0))
            label = 1
        sample = torch.cat(sample, dim=0)
        label = torch.tensor(label, dtype=torch.long,device='cpu')
        samples = {'image': sample, 'label': label}
        return samples

