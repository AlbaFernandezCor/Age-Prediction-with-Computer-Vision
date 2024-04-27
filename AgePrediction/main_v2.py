from data_generator import CACDDataset
from HYPERPARAMETERS import TRAIN_CSV_PATH, IMAGE_PATH, TEST_CSV_PATH, BATCH_SIZE
# from models.old_school import OldSchoolMethod
from models.deep_learning import DeepLearningMethod
from torchvision import transforms
import numpy as np
import time

def run():
    print('Start main')
    # Train
    s = time.time()
    transform = transforms.Compose([transforms.Resize((128, 128)),
                                        transforms.RandomCrop((120, 120)),
                                        transforms.ToTensor()])
    train_dataset = CACDDataset(csv_path=TRAIN_CSV_PATH, img_dir=IMAGE_PATH, transform=transform)
    print('X y train cargados:', np.round(time.time()-s),'s')

    # Test
    s = time.time()
    test_dataset = CACDDataset(csv_path=TEST_CSV_PATH, img_dir=IMAGE_PATH, transform=transform)
    print('X y test cargados:', np.round(time.time()-s),'s')

    # s = time.time()
    # OldSchoolMethod().run(train_dataset, test_dataset)
    # print('Fin Old School Method:', np.round(time.time()-s),'s')

    s = time.time()
    DeepLearningMethod().run(train_dataset, test_dataset)
    print('Fin Deep Learning Method:', np.round(time.time()-s),'s')
    



if __name__ == '__main__':
    run()