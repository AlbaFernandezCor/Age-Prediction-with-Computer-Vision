from HYPERPARAMETERS import TARGET_SIZE
from sklearn.metrics import r2_score
from models.cnn import CNNRegressor
import tensorflow as tf
import numpy as np
import os
from models.utils import plot_results
from models.cnn import AgeRegressionModel
from torchvision import transforms
import torch
from PIL import Image

class DeepLearning():

    def run(self, train_df, test_df, df_type='age'):
        train_img = self.image_preparation(train_df, df_type)
        test_img = self.image_preparation(test_df, df_type, True)
        model = CNNRegressor().model1()  # Canviar HYPERPARAMETERS.TARGET_SIZE model2 --> (224, 224) o model1 --> (120, 120)
        model = self.train_loop2(model, train_img, test_img)
        # model = self.load_model()
        
        # RESULTS
        y_pred, y_real = self.metrics(model, test_img)
        plot_results(y_pred, y_real, np.abs(y_pred - y_real))

    # def load_model(self):
    #     return tf.keras.models.load_model('/content/models/checkpoints/model_trained.h5')

    def image_preparation(self, df, df_type, df_test=None):
        with tf.device('GPU'):
            image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255
                )
            if df_test:
                df_images = image_generator.flow_from_dataframe(
                    dataframe=df,
                    x_col='image',
                    y_col=df_type,
                    target_size=TARGET_SIZE,
                    color_mode='rgb',
                    class_mode='raw',
                    batch_size=64,
                    shuffle=False
                )
            else: # Training
                df_images = image_generator.flow_from_dataframe(
                    dataframe=df,
                    x_col='image',
                    y_col=df_type,
                    target_size=TARGET_SIZE,
                    color_mode='rgb',
                    class_mode='raw',
                    batch_size=64,
                    shuffle=True,
                    seed=42
                )
        return df_images
    
        
    def train_loop(self, model, train_img):
        with tf.device('GPU'):
            model.compile(
                optimizer='adam',
                loss='mse'
            )

            history = model.fit(
                train_img,
                epochs=10,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ]
            )
            model.save(os.path.join('AgePredictionUTK/models/checkpoints/', 'model_trained_cnn.h5'))

        return model
    
    def train_loop2(self, age_model, train_img, test_img):
        age_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print(age_model.summary())              
                                
        history_age = age_model.fit(train_img, validation_data=test_img, epochs=10)

        age_model.save('age_model_50epochs.h5')
    
    def metrics(self, model, test_img):
        predicted_ages = np.squeeze(model.predict(test_img))
        true_ages = test_img.labels

        rmse = np.sqrt(model.evaluate(test_img, verbose=0))
        print("Test RMSE: {:.5f}".format(rmse))

        r2 = r2_score(true_ages, predicted_ages)
        print("Test R^2 Score: {:.5f}".format(r2))

        return predicted_ages, true_ages