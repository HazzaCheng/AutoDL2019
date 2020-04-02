#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/10/15 22:44
# @Author:  Mecthew
import keras
import tensorflow as tf
from keras import optimizers
from keras.layers import (SpatialDropout1D, Input, Bidirectional, GlobalMaxPool1D, GlobalAvgPool1D,
                          Concatenate,
                          Dense, Dropout, CuDNNLSTM)
from keras.models import Model as TFModel

from models.speech.neural_model.attention import Attention
from my_classifier import Classifier
from tools import log, ohe2cat, pad_seq


class AttentionGruModel(Classifier):
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
        inputs = Input(shape=input_shape)
        # bnorm_1 = BatchNormalization(axis=-1)(inputs)
        x = Bidirectional(CuDNNLSTM(96, name='blstm1',
                                    return_sequences=True),
                          merge_mode='concat')(inputs)
        # activation_1 = Activation('tanh')(lstm_1)
        x = SpatialDropout1D(0.1)(x)
        x = Attention(8, 16)([x, x, x])
        x1 = GlobalMaxPool1D()(x)
        x2 = GlobalAvgPool1D()(x)
        x = Concatenate(axis=-1)([x1, x2])
        x = Dense(units=128, activation='elu')(x)
        x = Dense(units=64, activation='elu')(x)
        x = Dropout(rate=0.4)(x)
        outputs = Dense(units=num_classes, activation='softmax')(x)

        model = TFModel(inputs=inputs, outputs=outputs)
        optimizer = optimizers.Adam(
            # learning_rate=1e-3,
            lr=1e-3,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            decay=0.0002,
            amsgrad=True)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
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
