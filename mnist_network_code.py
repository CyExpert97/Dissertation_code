# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 23:06:42 2020

@author: Ioannis Theocharides 957865
"""

import keras
import keras.backend as k
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential


# Load mnist dataset
mnist = keras.datasets.mnist

# x_train is the list of images y_train is the labels assigned to each image

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalise values to range (0,1)
x_train, x_test = x_train/255.0, x_test/255.0

# y_train = [tf.float32(i) for i in y_train]
# y_test = [numpy.float16(i) for i in y_test]
y_train = k.cast(y_train, 'float32')
y_test = k.cast(y_test, 'float32')
print(x_train.shape)

# model=model.keras.Sequential()
model = Sequential()
# optimizer='adam'
# (28,28) represents the dimensions of image in pixels
input_layer = Flatten(input_shape=(28, 28))
model.add(input_layer)

# Activation function is relu
hidden_layer_1 = Dense(128, activation='relu')
model.add(hidden_layer_1)

# Percentage of nodes destroyed
hidden_layer_2 = Dropout(0.3)
model.add(hidden_layer_2)

# Activation function is softmax
output_layer = Dense(10, activation='softmax')
model.add(output_layer)


def custom_loss(y_true, y_pred):
    return k.sum(k.log(y_true) - k.log(y_pred))


# Building model with appropriate loss function and optimizer.
# Metrics is values you want to show i.e in this case accuracy
model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])

# Training sets for code with 5 iterations of training
model.fit(x_train, y_train, epochs=5)

# The final test set checking the models performance vs actual test data
score = model.evaluate(x_test, y_test)
print(' accuracy ', score[1])
