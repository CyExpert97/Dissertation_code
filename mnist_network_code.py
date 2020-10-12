# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 23:06:42 2020

@author: Ioannis Theocharides 957865
"""

import keras
import keras.backend as k
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential

class Model():

    def __init__(self, data):

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

    def build_model(self):
        model = Sequential()
        input_layer = Flatten(input_shape=(28 ,28))
        model.add(input_layer)
        hidden_layer_1 = Dense(128, activation='relu')
        model.add(hidden_layer_1)
        hidden_layer_2 = Dropout(0.3)
        model.add(hidden_layer_2)
        outer_layer = Dense(10, activation='softmax')
        model.add(outer_layer)
        return model

    def create_loss(self, y_true, y_pred):
        loss = k.sum(k.log(y_true) - k.log(y_pred))
        return loss

    def compile_model(self):
        self.model.compile(loss=self.create_loss, optimizer='adam', metrics=['accuracy'])

    def fit_model(self):
        self.model.fit(self.x_train, self.y_train, epochs=5)

    def return_score(self):
        score = self.model.evaluate(self.x_test, self.y_test)
        print('accuracy', score[1])


mnist = keras.datasets.mnist
x = Model(mnist.load_data())
x.return_score()

