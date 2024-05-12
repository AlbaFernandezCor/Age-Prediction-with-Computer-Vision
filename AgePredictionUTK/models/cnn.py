import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import models
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input

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
    
    def model3(self, x_train_age, y_train_age, x_test_age, y_test_age):
        age_model = Sequential()
        age_model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200,200,3)))
        #age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
        age_model.add(MaxPool2D(pool_size=3, strides=2))

        age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
        #age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
        age_model.add(MaxPool2D(pool_size=3, strides=2))
                    
        age_model.add(Conv2D(256, kernel_size=3, activation='relu'))
        #age_model.add(Conv2D(256, kernel_size=3, activation='relu'))
        age_model.add(MaxPool2D(pool_size=3, strides=2))

        age_model.add(Conv2D(512, kernel_size=3, activation='relu'))
        #age_model.add(Conv2D(512, kernel_size=3, activation='relu'))
        age_model.add(MaxPool2D(pool_size=3, strides=2))

        age_model.add(Flatten())
        age_model.add(Dropout(0.2))
        age_model.add(Dense(512, activation='relu'))

        age_model.add(Dense(1, activation='linear', name='age'))

        return age_model
                    

    

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