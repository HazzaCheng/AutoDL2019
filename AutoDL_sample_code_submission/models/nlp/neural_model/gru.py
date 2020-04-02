#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2020/1/17 19:09
# @Author:  Mecthew
import math

import keras
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.layers import (Input, Embedding, Activation, Flatten, Conv1D, Dense, Dropout, CuDNNGRU,
                          GlobalMaxPooling1D, BatchNormalization, Reshape, Concatenate, PReLU, SpatialDropout1D)

from my_classifier import Classifier
from tools import ohe2cat, log
import numpy as np
from models.nlp.neural_model.focal_loss import binary_focal_loss


class Gru(Classifier):
    def __init__(self):
        log("new {}".format(self.__class__.__name__))
        self._model = None
        self.is_init = False
        self._num_classes = None
        self._is_multilabel = False
        self._use_step_decay = False
        self._cur_model_run_loop = 0
        self._initial_lr = 0.016

    def step_decay(self, epoch):
        epoch = self._cur_model_run_loop
        initial_lrate = self._initial_lr
        drop = 0.65
        epochs_drop = 1.0 #2.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    def preprocess_data(self, x, val_x=None):
        if val_x is not None:
            return x, val_x
        else:
            return x

    def init_model(self,
                   embedding_matrix,
                   seq_len,
                   num_features,
                   num_classes,
                   is_multilabel,
                   is_balanced,
                   classes_ratio,
                   filters=100,
                   emb_size=300,
                   use_step_decay=False,
                   **kwargs):
        self._use_step_decay = use_step_decay
        self._num_classes = num_classes
        self._is_multilabel = is_multilabel
        if num_classes == 2 or is_multilabel:
            loss = 'binary_crossentropy' if is_balanced or is_multilabel else binary_focal_loss(gamma=2, alpha=(
                        1 - classes_ratio[1]))
            output_activation = 'sigmoid'
            if is_multilabel:
                output_units = self._num_classes
            else:
                output_units = 1
        else:
            loss = 'sparse_categorical_crossentropy'
            output_activation = 'softmax'
            output_units = num_classes

        trainable = True
        inputs = Input(name='inputs', shape=(seq_len,))
        if embedding_matrix is None:
            x = Embedding(input_dim=num_features,
                          output_dim=emb_size,
                          input_length=seq_len,
                          trainable=trainable)(inputs)
        else:
            x = Embedding(input_dim=num_features,
                          output_dim=emb_size,
                          input_length=seq_len,
                          trainable=trainable,
                          embeddings_initializer=keras.initializers.Constant(
                              embedding_matrix))(inputs)

        # QMC
        # x = CuDNNGRU(128, return_sequences=True)(x)
        # x = Activation('tanh')(x)
        # x = SpatialDropout1D(0.4)(x)
        # x = GlobalMaxPooling1D()(x)
        #
        # x = Dense(128, activation='softplus')(x)  #
        # x = Dropout(0.5)(x)  # 0
        # x = BatchNormalization()(x)

        # DB
        x = CuDNNGRU(128, return_sequences=True)(x)
        x = GlobalMaxPooling1D()(x)

        x = Dense(128)(x)
        x = PReLU()(x)
        x = Dropout(0.35)(x)
        x = BatchNormalization()(x)

        output = Dense(output_units, activation=output_activation)(x)
        model = keras.models.Model(inputs=inputs, outputs=output)

        optimizer = optimizers.Adam()
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.summary()
        self.is_init = True
        self._model = model

    def fit(self, train_x, train_y, validation_data_fit, epochs, cur_model_run_loop, batch_size=32, **kwargs):
        val_x, val_y = validation_data_fit
        callbacks = []

        if self._use_step_decay:
            lrate = LearningRateScheduler(self.step_decay)
            callbacks.append(lrate)
        if self._is_multilabel:
            train_y = train_y
            val_y = val_y
        else:
            train_y = ohe2cat(train_y)
            val_y = ohe2cat(val_y)

        if epochs > 1:
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
            callbacks.append(early_stop)
            self._model.fit(train_x, train_y,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=(val_x, val_y),
                            verbose=1,  # Logs once per epoch.
                            batch_size=batch_size,
                            shuffle=True)
        else:
            self._model.fit(train_x, train_y,
                            epochs=1,
                            callbacks=callbacks,
                            verbose=1,  # Logs once per epoch.
                            batch_size=batch_size,
                            shuffle=True)

    def predict(self, x_test, batch_size=32):
        predicts = self._model.predict(x_test, batch_size=batch_size)
        if self._num_classes == 2 and predicts.shape[-1] == 1:
            predicts = predicts.reshape(-1)
            another = 1 - predicts
            return np.concatenate([another.reshape(-1, 1), predicts.reshape(-1, 1)], axis=-1)
        else:
            return predicts