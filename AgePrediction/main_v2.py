from data_generator import CACDDataset
from HYPERPARAMETERS import TRAIN_CSV_PATH, IMAGE_PATH, TEST_CSV_PATH, BATCH_SIZE
from utils_main import CACDplot, generate_Xy_dataset
from models.old_school import OldSchoolMethod
from torch.utils.data import DataLoader
import numpy as np
import time

def run():
    print('Start main')
    # Train
    s = time.time()
    train_dataset = CACDDataset(csv_path=TRAIN_CSV_PATH, img_dir=IMAGE_PATH)
    # X_train, y_train = generate_Xy_dataset(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("passo train loader")
    all_features = []
    all_targets = []
    for i, (features, targets) in enumerate(train_loader):
        if features is not None:
            all_features.append(features)
            all_targets.append(targets)
            print("Batch:", i)
    exit()
    print('X y train cargados:',np.round(time.time()-s),'s')

    # Test
    s = time.time()
    try:
        print("Intentando cargar los datos de prueba desde el archivo...")
        X_test, y_test = cargar_datos("test_data.pkl")
    except FileNotFoundError:
        test_dataset = CACDDataset(csv_path=TEST_CSV_PATH, img_dir=IMAGE_PATH)
        X_test, y_test = generate_Xy_dataset(test_dataset)
        guardar_datos(X_test, y_test, "test_data.pkl")
    print('X y test cargados:',np.round(time.time()-s),'s')

    # CACDplot(train_dataset)
    # CACDplot(test_dataset)
    # OldSchoolMethod(train_dataset)



if __name__ == '__main__':
    run()