import cv2
import numpy as np
from math import ceil
from AgePrediction.models.utils_models_oldschool import generate_RFR_model, generate_predictions, plot_results
 
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

