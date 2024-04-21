import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np

class CACDDataset():
    """Custom Dataset for loading CACD face images"""

    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['file'].values
        self.y = df['age'].values

    def __getitem__(self, index):
        img = None
        label = None

        if self.img_names[index].isascii():
            img = cv2.imread(os.path.join(self.img_dir, self.img_names[index]))
            label = self.y[index]

        return img, label

    def __len__(self):
        return self.y.shape[0]
    