#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/9/24 15:12
# @Author:  Mecthew
import keras
from keras import optimizers
from keras.layers import (Activation, Flatten, Conv2D,
                          MaxPooling2D, BatchNormalization, Reshape)
from keras.layers import (Dense, Dropout)
from keras.models import Sequential

from my_classifier import Classifier
from tools import ohe2cat, log, pad_seq


class Cnn2DModel(Classifier):
    def __init__(self):
        # clear_session()
        log("new {}".format(self.__class__.__name__))
        self._model = None
        self.is_init = False

    def preprocess_data(self, x, val_x=None, feature_length=None):
        x = pad_seq(x, pad_len=feature_length)

        if val_x is not None:
            val_x = pad_seq(val_x, pad_len=feature_length)
            return x, val_x

        return x

    def init_model(self,
                   input_shape,
                   num_classes,
                   max_layer_num=5,
                   **kwargs):
        model = Sequential()
        model.add(Reshape((input_shape[0], input_shape[1], 1)))
        min_size = min(input_shape[:2])
        for i in range(max_layer_num):
            if i == 0:
                model.add(
                    Conv2D(
                        64,
                        3,
                        input_shape=input_shape,
                        padding='same'))
            else:
                model.add(Conv2D(64, 3, padding='same'))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            min_size //= 2
            if min_size < 2:
                break

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dropout(rate=0.5))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6)

        optimizer = optimizers.Adam()
        # optimizer = optimizers.SGD(lr=1e-3, decay=2e-4, momentum=0.9, clipvalue=5)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        # model.summary()
        self.is_init = True
        self._model = model

    def fit(self, train_x, train_y, validation_data_fit, epochs, **kwargs):
        val_x, val_y = validation_data_fit

        callbacks = [keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3)]

        self._model.fit(train_x, ohe2cat(train_y),
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(val_x, ohe2cat(val_y)),
                        verbose=1,  # Logs once per epoch.
                        batch_size=32,
                        shuffle=True)

    def predict(self, x_test, batch_size=32):

        return self._model.predict(x_test, batch_size=batch_size)
