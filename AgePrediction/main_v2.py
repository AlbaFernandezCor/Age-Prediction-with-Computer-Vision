from data_generator import CACDDataset
from HYPERPARAMETERS import TRAIN_CSV_PATH, IMAGE_PATH, TEST_CSV_PATH, BATCH_SIZE
from utils_main import CACDplot, generate_Xy_dataset
from models.old_school import OldSchoolMethod
from torch.utils.data import DataLoader
import numpy as np
import time
import pickle
import os

def guardar_datos(X, y, nombre_archivo):
    try:
        with open(nombre_archivo, 'wb') as f:
            pickle.dump((X, y), f)
    except FileNotFoundError:
        print("Creant fitxer pkl")
        os.makedirs(os.path.dirname(nombre_archivo), exist_ok=True)
        with open(nombre_archivo, 'wb') as f:
            pickle.dump((X, y), f)

def cargar_datos(nombre_archivo):
    with open(nombre_archivo, 'rb') as f:
        X, y = pickle.load(f)
    return X, y

def run():
    print('Start main')
    # Train
    s = time.time()
    try:
        print("Intentando cargar los datos de entrenamiento desde el archivo...")
        X_train, y_train = cargar_datos("train_data.pkl")
    except FileNotFoundError:
        train_dataset = CACDDataset(csv_path=TRAIN_CSV_PATH, img_dir=IMAGE_PATH)
        # X_train, y_train = generate_Xy_dataset(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=train_dataset.__len__(), shuffle=True)
        for i, (features, targets) in enumerate(train_loader):
            print("features\n", features)
            print("targets\n", targets)
            len(targets)
    exit()
    guardar_datos(X_train, y_train, "train_data.pkl")
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