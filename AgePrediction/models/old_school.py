import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

 
class OldSchoolMethod():

    def run(self, X_train, y_train, X_test, y_test):
        X_train_descriptor = self.preprocesing_X(X_train)
        X_test_descriptor = self.preprocesing_X(X_test)
        y_pred, error = self.RFR(X_train_descriptor, y_train, X_test_descriptor, y_test)
        return y_pred, error

    def extract_sift_features(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        return keypoints, descriptors

    def preprocesing_X(self, X):
        descriptors = []
        for image in X:
            keypoints, descriptors = self.extract_sift_features(image)
            descriptors.append(descriptors)

        return np.concatenate(descriptors, axis=0)


    def RFR(self, X_train, y_train, X_test, y_test):
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        errors = np.abs(y_pred - y_test)
        return y_pred, errors
