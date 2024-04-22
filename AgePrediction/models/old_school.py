import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from math import ceil
import seaborn as sns
import matplotlib.pylab as plt
from utils_models import generate_RFR_model, generate_predictions, plot_results
 
class OldSchoolMethod():

    def run(self, train_dataset, test_dataset):
        X_train, y_train = self.preprocesing_Xy(train_dataset)
        X_test, y_test = self.preprocesing_Xy(test_dataset)
        model = generate_RFR_model(X_train, y_train)
        y_pred, error = generate_predictions(model, X_test, y_test)
        plot_results(y_pred, y_test, error)

    def extract_sift_features(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        _, descriptors = sift.detectAndCompute(gray_image, None)
        return descriptors

    def preprocesing_Xy(self, dataset):
        X_list = []
        y_list = []
        for index in range(ceil(dataset.__len__()*0.01)):
            img, age = dataset.__getitem__(index)
            descriptors = self.extract_sift_features(img)
            y_list.append(age)
            X_list.append(np.array(np.concatenate(descriptors, axis=0).tolist())[:1000])
            if (index % 50 == 0):
                print("Index img: ", index)
        return X_list, y_list


    def RFR(self, X_train, y_train, X_test, y_test):
        model = RandomForestRegressor()
        model.fit(np.array(X_train), np.array(y_train))
        y_pred = model.predict(X_test)
        errors = np.abs(y_pred - y_test)
        return y_pred, errors
    
    def MSE(self, y_real, y_pred):
        vari = []
        for i in range(len(y_real)):
            vari.append((y_pred[i] - y_real[i])**2)
        
        return vari
        
    def plot_results(self, y_pred, y_test, vari):
        # Plot y_pred - MSE
        sns.scatterplot(x=y_pred,y=vari)
        plt.title('y_pred - MSE')
        plt.xlabel('y_pred')
        plt.ylabel('MSE')
        print("MSE Mean: ", np.array(vari).mean())
        plt.show()
        # Plot y_pred - y_test
        sns.scatterplot(x=y_test,y=y_pred)
        sns.lineplot(x=y_test, y=y_test)
        plt.title('y_pred - y_test')
        plt.xlabel('y_pred')
        plt.ylabel('y_test')
        plt.show()

