from HYPERPARAMETERS import TARGET_SIZE
from sklearn.metrics import r2_score
from models.cnn import CNNRegressor
import tensorflow as tf
import numpy as np
import os


class DeepLearning():

    def run(self, train_df, test_df, valid_df):
        train_img = self.image_preparation(train_df)
        test_img = self.image_preparation(test_df, True)
        valid_img = self.image_preparation(valid_df)
        model = CNNRegressor().model1()  # Canviar HYPERPARAMETERS.TARGET_SIZE model2 --> (224, 224) o model1 --> (120, 120)
        model = self.train_loop(model, train_img, valid_img)
        self.metrics(model, test_img)

    def image_preparation(self, df, df_test=None):
        with tf.device('GPU'):
            image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255
                )
            if df_test:
                df_images = image_generator.flow_from_dataframe(
                    dataframe=df,
                    x_col='file',
                    y_col='age',
                    target_size=TARGET_SIZE,
                    color_mode='rgb',
                    class_mode='raw',
                    batch_size=64,
                    shuffle=False
                )
            else:
                df_images = image_generator.flow_from_dataframe(
                    dataframe=df,
                    x_col='file',
                    y_col='age',
                    target_size=TARGET_SIZE,
                    color_mode='rgb',
                    class_mode='raw',
                    batch_size=64,
                    shuffle=True,
                    seed=42
                )
        return df_images
    
        
    def train_loop(self, model, train_img, valid_img):
        with tf.device('GPU'):
            model.compile(
                optimizer='adam',
                loss='mse'
            )

            history = model.fit(
                train_img,
                validation_data=valid_img,
                epochs=100,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ]
            )

            model.save(os.path.join('AgePredictionBo/models/checkpoints/', 'model_trained.h5'))

        return model
    
    def metrics(self, model, test_img):
        predicted_ages = np.squeeze(model.predict(test_img))
        true_ages = test_img.labels

        rmse = np.sqrt(model.evaluate(test_img, verbose=0))
        print("Test RMSE: {:.5f}".format(rmse))

        r2 = r2_score(true_ages, predicted_ages)
        print("Test R^2 Score: {:.5f}".format(r2))