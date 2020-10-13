# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 23:06:42 2020

@author: Ioannis Theocharides 957865
"""
import tensorflow as tf
import keras
import keras.backend as k
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential


from tensorflow.python.keras.losses import Loss


# Creates Model for neural network
class Model():

# Giving initial values

    def __init__(self, data):
# Defining data structure and how model compiles
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.x_train, self.x_test = self.x_train/255.0, self.x_test/255.0
        self.y_train = k.cast(self.y_train, 'float32')
        self.y_test = k.cast(self.y_test, 'float32')
        # Model
        self.model = self.build_model()
        # Print a model summary
        self.model.summary()
        self.compile_model()
        self.fit_model()
# Describing how model is built

    def build_model(self):
        model = Sequential()
        input_layer = Flatten(input_shape=(28, 28))
        model.add(input_layer)
        hidden_layer_1 = Dense(128, activation='relu')
        model.add(hidden_layer_1)
        hidden_layer_2 = Dropout(0.3)
        model.add(hidden_layer_2)
        outer_layer = Dense(10, activation='softmax')
        model.add(outer_layer)
        return model

# Defining Loss function, optimizer and accuracy

    def compile_model(self):
        self.model.compile(loss=Loss_function(), optimizer='adam', metrics=['accuracy'])
# How the model fits and how many iterations it runs

    def fit_model(self):
        self.model.fit(self.x_train, self.y_train, epochs=5)
# Return the accuracy of the model

    def return_score(self):
        score = self.model.evaluate(self.x_test, self.y_test)
        print('accuracy', score[1])


# Defining the Loss function in seperate class
class Loss_function(Loss):

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return k.sum(k.log(y_true) - k.log(y_pred))


# Calling mnist dataset and model
mnist = keras.datasets.mnist
x = Model(mnist.load_data())
x.return_score()
