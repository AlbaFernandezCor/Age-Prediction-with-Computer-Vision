import numpy as np
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pylab as plt

class RandomForest():

    def model(self, X_train, y_train):
        model = RandomForestRegressor(n_jobs=-1)
        model.fit(X_train, y_train)
        return model


    


    