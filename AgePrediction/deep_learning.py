import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image

torch.backends.cudnn.deterministic = True

TRAIN_CSV_PATH = './datasets/cacd_train.csv'
VALID_CSV_PATH = './datasets/cacd_valid.csv'
TEST_CSV_PATH = './datasets/cacd_test.csv'
IMAGE_PATH = './datasets/CACD2000'

# Hyperparameters
learning_rate = 0.0005
num_epochs = 0  # 200

# Architecture
NUM_CLASSES = 49
BATCH_SIZE = 512  # 256
GRAYSCALE = False

class CACDDataset(Dataset):
    """Custom Dataset for loading CACD face images"""

    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['file'].values
        self.y = df['age'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]

        return img, label

    def __len__(self):
        return self.y.shape[0]