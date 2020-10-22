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
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam


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
        #self.step(self.x_train, self.y_train)

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

    def step(self, y_true, y_pred):
        with tf.GradientTape() as tape:
            pred = self.model(y_true)
            loss = sparse_categorical_crossentropy(y_pred, pred)

        grads = tape.gradient(loss, self.model.trainable_variables)
        opt = Adam
        opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return opt

    def create_loss(self, y_true, y_pred):
        loss = k.sum(k.log(y_true) - k.log(y_pred))
        return loss

    def compile_model(self):
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.step, metrics=['accuracy'])

    def fit_model(self):
        self.model.fit(self.x_train, self.y_train, epochs=5)

    def return_score(self):
        score = self.model.evaluate(self.x_test, self.y_test)
        print('accuracy', score[1])


class Loss_function():

    @staticmethod
    def loss(y_true, y_pred):
        loss = k.sum(k.log(y_true) - k.log(y_pred))
        return loss


# class optimizer:
#
#     def __init__(self):
#         self.model = Model()
#
#     def loss(self, y_true, y_pred):
#         with tf.GradientTape() as tape:
#             loss = sparse_categorical_crossentropy(y_true, y_pred)
#
#         opt = Adam
#         grads = tape.gradient(loss, self.model.trainable_variables)
#         opt.apply_grdients(zip(grads, self.model.trainable_variables))
#         return opt
        
        
mnist = keras.datasets.mnist
x = Model(mnist.load_data())
x.return_score()

