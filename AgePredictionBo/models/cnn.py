import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import models

class CNNRegressor():

    def model1(self):
        inputs = tf.keras.Input(shape=(120, 120, 3))
        x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def model2(self):
        input = tf.keras.Input(shape=(224, 224, 3))
        cnn1 = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu')(input)
        cnn1 = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu')(cnn1)
        cnn1 = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu')(cnn1)
        cnn1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(cnn1)

        cnn2 = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu')(cnn1)
        cnn2 = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu')(cnn2)
        cnn2 = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu')(cnn2)
        cnn2 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(cnn2)

        cnn3 = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu')(cnn2)
        cnn3 = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu')(cnn3)
        cnn3 = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu')(cnn3)
        cnn3 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(cnn3)

        cnn4 = tf.keras.layers.Conv2D(512, kernel_size=3, activation='relu')(cnn3)
        cnn4 = tf.keras.layers.Conv2D(512, kernel_size=3, activation='relu')(cnn4)
        cnn4 = tf.keras.layers.Conv2D(512, kernel_size=3, activation='relu')(cnn4)
        cnn4 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(cnn4)

        dense = tf.keras.layers.Flatten()(cnn4)
        dense = tf.keras.layers.Dropout(0.2)(dense)
        dense = tf.keras.layers.Dense(1024, activation='relu')(dense)
        dense = tf.keras.layers.Dense(1024, activation='relu')(dense)

        output = tf.keras.layers.Dense(1, activation='linear', name='age')(dense)
        model = tf.keras.Model(input, output)
        return model
    

class AgeRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnext101_32x8d(pretrained=True)
        self.model.fc = nn.Linear(512 * 4, 1)

    def forward(self, x: torch.Tensor):
        x_age = self.model(x)
        return x_age
      
#Model 2
#Provar codi del link https://mohameddhaoui.github.io/deeplearning/Age_detection/
    
