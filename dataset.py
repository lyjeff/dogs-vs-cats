import os
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
ImageFile.LOAD_TRUNCATED_IMAGES = True

# image transform for train and test
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CatDataset(Dataset):
    def __init__(self, path):
        self.data = []

        for img_name in glob(os.path.join(path, '*.jpg')):
            label = 1.0 if os.path.basename(img_name).split('.')[0] == 'dog' else 0.0
            self.data.append((Image.open(img_name), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = data_transforms(self.data[idx][0])
        return data, self.data[idx][1]


def cross_validation(full_set, p=0.8):
    """
    hold out cross validation
    """
    train_len = len(full_set)

    # get shuffled indices
    indices = np.random.permutation(range(train_len))
    split_idx = int(train_len * p)

    train_idx, valid_idx = indices[:split_idx], indices[split_idx:]
    full_set = np.array(full_set)
    train_set = full_set[list(SubsetRandomSampler(train_idx))]
    valid_set = full_set[list(SubsetRandomSampler(valid_idx))]
    train_set = torch.from_numpy(train_set)
    valid_set = torch.from_numpy(valid_set)
    return train_set, valid_set