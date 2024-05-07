import cv2
import numpy as np
import os
import joblib
from models.utils import plot_results
from models.rf import RandomForest
from PIL import Image
 
class OldSchoolMethod():

    def run(self, train_dataset, test_dataset, model_loaded = None):
        if model_loaded: # hola alba
            model = self.model_load()
        else:
            print('Preparant les dades de train...')
            X_train, y_train = self.preprocesing_Xy(train_dataset)
            print('Carregant model...')
            model = RandomForest().model(X_train, y_train)
            self.model_save(model)
        print('Generant prediccions i gràfics...')
        y_pred, error = self.generate_predictions(model, X_test, y_test)
        plot_results(y_pred, y_test, error)


    def extract_sift_features(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        _, descriptors = sift.detectAndCompute(gray_image, None)
        return descriptors
    

    def preprocesing_Xy(self, df):
        X_list = []
        for index in range(len(df)):
            descriptors = self.extract_sift_features(np.array(Image.open(df.iloc[index]['file'])))
            X_list.append(np.concatenate(descriptors, axis=0)[:1000])
            if (index % 5000 == 0):
                print("Index img: ", index)

        assert len(X_list) == len(df['age']), "Les longituds de Imatges i edats no son iguals!"
        return np.array(X_list), np.array(df['age'])


    def generate_predictions(self, model, X, y):
        y_pred = model.predict(X)
        errors = np.abs(y_pred - y)
        return y_pred, errors
    
    def model_save(self, model):
        joblib.dump(model, 'AgePredictionBo/models/checkpoints/modelRF_trained.joblib', compress=3)

    def model_load(self):
        return joblib.load('AgePredictionBo/models/checkpoints/modelRF_trained.joblib')
