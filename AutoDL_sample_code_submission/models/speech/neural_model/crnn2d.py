#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/9/26 21:29
# @Author:  Mecthew
import keras
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.layers import (Input, Dense, Dropout, Convolution2D,
                          MaxPooling2D, ELU, Reshape, CuDNNGRU)
from keras.layers.normalization import BatchNormalization
from keras.models import Model as TFModel

from my_classifier import Classifier
from tools import ohe2cat, log, pad_seq


class Crnn2dModel(Classifier):
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
                   **kwargs):
        freq_axis = 2
        channel_axis = 3
        channel_size = 128
        min_size = min(input_shape[:2])
        feature_input = Input(shape=input_shape)
        # x = ZeroPadding2D(padding=(0, 37))(feature_input)
        # x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(x)

        x = Reshape((input_shape[0], input_shape[1], 1))(feature_input)
        # Conv block 1
        x = Convolution2D(64, 3, strides=(1, 1), padding='same', name='conv1')(x)
        x = BatchNormalization(axis=channel_axis, name='bn1')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
        x = Dropout(0.1, name='dropout1')(x)

        # Conv block 2
        x = Convolution2D(channel_size, 3, strides=(1, 1), padding='same', name='conv2')(x)
        x = BatchNormalization(axis=channel_axis, name='bn2')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)
        x = Dropout(0.1, name='dropout2')(x)

        # Conv block 3
        x = Convolution2D(channel_size, 3, strides=(1, 1), padding='same', name='conv3')(x)
        x = BatchNormalization(axis=channel_axis, name='bn3')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)
        x = Dropout(0.1, name='dropout3')(x)

        if min_size // 24 >= 4:
            # Conv block 4
            x = Convolution2D(
                channel_size,
                3,
                strides=(1, 1),
                padding='same',
                name='conv4')(x)
            x = BatchNormalization(axis=channel_axis, name='bn4')(x)
            x = ELU()(x)
            x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)
            x = Dropout(0.1, name='dropout4')(x)

        x = Reshape((-1, channel_size))(x)

        gru_units = 128
        if num_classes > gru_units:
            gru_units = int(num_classes * 1.5)
        # GRU block 1, 2, output
        x = CuDNNGRU(gru_units, return_sequences=True, name='gru1')(x)
        x = CuDNNGRU(gru_units, return_sequences=False, name='gru2')(x)
        # x = Dense(max(int(num_classes*1.5), 128), activation='relu', name='dense1')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax', name='output')(x)

        model = TFModel(inputs=feature_input, outputs=outputs)
        optimizer = optimizers.Adam(
            # learning_rate=1e-3,
            lr=1e-3,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            decay=1e-4,
            amsgrad=True)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'])
        model.summary()
        self._model = model
        self.is_init = True

    def fit(self, train_x, train_y, validation_data_fit, epochs, **kwargs):
        val_x, val_y = validation_data_fit

        patience = 2
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience)]
        self._model.fit(train_x, ohe2cat(train_y),
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(val_x, ohe2cat(val_y)),
                        verbose=1,  # Logs once per epoch.
                        batch_size=32,
                        shuffle=True)

    def predict(self, x_test, batch_size=32):

        return self._model.predict(x_test, batch_size=batch_size)
