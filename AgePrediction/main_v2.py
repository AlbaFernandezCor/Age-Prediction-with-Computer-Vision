from data_generator import CACDDataset
from HYPERPARAMETERS import TRAIN_CSV_PATH, IMAGE_PATH, TEST_CSV_PATH, BATCH_SIZE
from utils_main import CACDplot
from models.old_school import OldSchoolMethod
from torch.utils.data import DataLoader
import numpy as np
import time

def run():
    print('Start main')
    s = time.time()
    train_dataset = CACDDataset(csv_path=TRAIN_CSV_PATH, img_dir=IMAGE_PATH)
    print('Train dataset done:',np.round(time.time()-s),'s')
    s = time.time()
    X_train, y_train = train_dataset.generate_Xy_dataset()
    print('X y train generated:',np.round(time.time()-s),'s')
    s = time.time()
    test_dataset = CACDDataset(csv_path=TEST_CSV_PATH, img_dir=IMAGE_PATH)
    print('Test dataset done:',np.round(time.time()-s),'s')
    # X_test, y_test = test_dataset.generate_Xy_dataset()
    # CACDplot(train_dataset)
    # CACDplot(test_dataset)
    # OldSchoolMethod(train_dataset)



if __name__ == '__main__':
    run()