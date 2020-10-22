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
import math

class Model():

    def __init__(self, data):

        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.x_train, self.x_test = self.x_train/255.0, self.x_test/255.0
        self.y_train = k.cast(self.y_train, 'float32')
        self.y_test = k.cast(self.y_test, 'float32')
        self.batch_size = 128
        self.epochs = 20
        # Model
        self.model = self.build_model()
        self.opt = Adam()
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

    def step(self, x_true, y_true):
        with tf.GradientTape() as tape:
            pred = self.model(x_true)
            loss = sparse_categorical_crossentropy(y_true, pred)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def create_loss(self, y_true, y_pred):
        loss = k.sum(k.log(y_true) - k.log(y_pred))
        return loss

    def compile_model(self):
        bat_per_epoch = math.floor(len(self.x_train) / self.batch_size)
        for epoch in range(self.epochs):
            print('=', end='')
            for i in range(bat_per_epoch):
                n = i * self.batch_size
                self.step(self.x_train[n:n + self.batch_size], self.y_train[n:n + self.batch_size])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.opt, metrics=['accuracy'])

    def fit_model(self):
        self.model.fit(self.x_train, self.y_train, epochs=self.epochs)

    def return_score(self):
        score = self.model.evaluate(self.x_test, self.y_test)
        print('accuracy', score[1])


class Loss_function:

    @staticmethod
    def loss(y_true, y_pred):
        loss = k.sum(k.log(y_true) - k.log(y_pred))
        return loss


mnist = keras.datasets.mnist
x = Model(mnist.load_data())
x.return_score()
