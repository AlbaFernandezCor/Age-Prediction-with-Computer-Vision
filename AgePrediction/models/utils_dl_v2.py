from tensorflow.python.keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.python.keras.models import Model
import tensorflow as tf


input = Input(shape=(224, 224, 3))

cnn1 = Conv2D(128, kernel_size=3, activation='relu')(input)
cnn1 = Conv2D(128, kernel_size=3, activation='relu')(cnn1)
cnn1 = Conv2D(128, kernel_size=3, activation='relu')(cnn1)
cnn1 = MaxPool2D(pool_size=3, strides=2)(cnn1)

cnn2 = Conv2D(128, kernel_size=3, activation='relu')(cnn1)
cnn2 = Conv2D(128, kernel_size=3, activation='relu')(cnn2)
cnn2 = Conv2D(128, kernel_size=3, activation='relu')(cnn2)
cnn2 = MaxPool2D(pool_size=3, strides=2)(cnn2)

cnn3 = Conv2D(256, kernel_size=3, activation='relu')(cnn2)
cnn3 = Conv2D(256, kernel_size=3, activation='relu')(cnn3)
cnn3 = Conv2D(256, kernel_size=3, activation='relu')(cnn3)
cnn3 = MaxPool2D(pool_size=3, strides=2)(cnn3)

cnn4 = Conv2D(512, kernel_size=3, activation='relu')(cnn3)
cnn4 = Conv2D(512, kernel_size=3, activation='relu')(cnn4)
cnn4 = Conv2D(512, kernel_size=3, activation='relu')(cnn4)
cnn4 = MaxPool2D(pool_size=3, strides=2)(cnn4)

dense = Flatten()(cnn4)
dense = Dropout(0.2)(dense)
dense = Dense(1024, activation='relu')(dense)
dense = Dense(1024, activation='relu')(dense)

output = Dense(1, activation='linear', name='age')(dense)

model = Model(input, output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001), loss='mse', metrics=['mae'])
