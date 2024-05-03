import os.path
from HYPERPARAMETERS import IMAGE_PATH, TRAIN_CSV_PATH, TEST_CSV_PATH, VALID_CSV_PATH
import pandas as pd
from PIL import Image

class CACDDataset():

    def run(self):
        self.train_df = self.loadcsv(TRAIN_CSV_PATH)
        self.test_df = self.loadcsv(TEST_CSV_PATH)
        self.valid_df = self.loadcsv(VALID_CSV_PATH)
        return self.train_df, self.test_df, self.valid_df

    def loadcsv(self, path_img):
        df = pd.read_csv(path_img, index_col=0)
        df['file'] = df['file'].apply(lambda x: os.path.join(IMAGE_PATH,x))
        return df

    def getitem(self, index):
        img = Image.open(self.train_df.iloc[index]['file'])
        label = self.train_df.iloc[index]['age']
        return img, label

