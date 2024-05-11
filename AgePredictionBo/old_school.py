import cv2
import numpy as np
import os
import joblib
from models.utils import plot_results
from models.rf import RandomForest
from PIL import Image
import pandas as pd
from math import ceil
 
class OldSchoolMethod():

    def run(self, train_dataset, test_dataset, model_loaded = None):
        if model_loaded:
            model = self.model_load()
            print('Preparant les dades de test...')
            X_test_preprocess, y_test_preprocess = self.preprocesing_Xy(test_dataset)
            X_test, y_test = self.modify_descriptors(X_test_preprocess, y_test_preprocess, n_features = model.n_features_in_)
        else:
            print('Preparant les dades de train...')
            X_train_preprocess, y_train_preprocess = self.preprocesing_Xy(train_dataset)
            print('Preparant les dades de test...')
            X_test_preprocess, y_test_preprocess = self.preprocesing_Xy(test_dataset)
            X_test, y_test, X_train, y_train = self.modify_descriptors(X_test_preprocess, y_test_preprocess, X_train_preprocess, y_train_preprocess)
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

    def extract_harris_features(self, image): # CANVIAR!
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        harris = cv2.cornerHarris(gray_image,2,3,0.04)
        return harris

    def preprocesing_Xy(self, df):
        X_list = []
        y_list = []
        img_log = ceil(len(df)*0.05)
        print(img_log)
        for index in range(len(df)):
            descriptors = self.extract_sift_features(np.array(Image.open(df.iloc[index]['file'])))
            if descriptors is not None:
                X_list.append(np.array(np.concatenate(descriptors, axis=0).tolist()))
                y_list.append(df.iloc[index]['age'])
                if (index % img_log == 0):
                    print("Index img: ", index)

        return X_list, y_list

    
    def modify_descriptors(self, X_test, y_test, X_train=None, y_train=None, n_features=None):
        if n_features:
            df = pd.DataFrame({'X':X_test, 'len_X': list(map(len, X_test)), 'y':y_test})
            df2 = df[df['len_X'] >= n_features].reset_index(drop=True)
            df2['X_short'] = df2['X'].apply(lambda x: x[:ceil(n_features)])
            return list(df2['X_short']), list(df2['y'])
        else:
            df_train = pd.DataFrame({'X':X_train, 'len_X': list(map(len, X_train)), 'y':y_train})
            df_test = pd.DataFrame({'X':X_test, 'len_X': list(map(len, X_test)), 'y':y_test})
            minim_val = min(np.percentile(df_train['len_X'], 60), np.percentile(df_test['len_X'], 60))
            df_train2 = df_train[df_train['len_X'] >= minim_val].reset_index(drop=True)
            df_train2['X_short'] = df_train2['X'].apply(lambda x: x[:ceil(minim_val)])
            df_test2 = df_test[df_test['len_X'] >= minim_val].reset_index(drop=True)
            df_test2['X_short'] = df_test2['X'].apply(lambda x: x[:ceil(minim_val)])
            return list(df_train2['X_short']), list(df_train2['y']), list(df_test2['X_short']), list(df_test2['y'])

    def generate_predictions(self, model, X, y):
        y_pred = model.predict(X)
        errors = np.abs(y_pred - y)
        return y_pred, errors
    
    def model_save(self, model):
        joblib.dump(model, 'AgePredictionBo/models/checkpoints/modelRF_trained.joblib', compress=3)

    def model_load(self):
        return joblib.load('AgePredictionBo/models/checkpoints/modelRF_trained.joblib')
