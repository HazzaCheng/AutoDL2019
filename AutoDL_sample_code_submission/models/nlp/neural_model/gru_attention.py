#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2020/1/17 19:18
# @Author:  Mecthew
import keras
from keras import optimizers
from keras.layers import (Input, Embedding, Activation, Flatten, Conv1D, Dense, Dropout, CuDNNGRU,
                          GlobalMaxPooling1D, BatchNormalization, Reshape, Concatenate, PReLU, SpatialDropout1D)
from models.nlp.neural_model.attention import Attention
from my_classifier import Classifier
from tools import ohe2cat, log
import numpy as np


class GruAttention(Classifier):
    def __init__(self):
        log("new {}".format(self.__class__.__name__))
        self._model = None
        self.is_init = False
        self._num_classes = None

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
                   filters=100,
                   emb_size=300,
                   **kwargs):
        self._num_classes = num_classes
        if num_classes == 2:
            loss = 'binary_crossentropy'
            output_activation = 'sigmoid'
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

        x = CuDNNGRU(128, return_sequences=True)(x)
        x = Activation('tanh')(x)
        x = SpatialDropout1D(0.4)(x)
        x = Attention(8, 16)([x, x, x])
        x = GlobalMaxPooling1D()(x)

        x = Dense(128, activation='softplus')(x)  #
        x = Dropout(0.5)(x)  # 0
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
        predicts = self._model.predict(x_test, batch_size=batch_size)
        if self._num_classes == 2 and predicts.shape[-1] == 1:
            predicts = predicts.reshape(-1)
            another = 1 - predicts
            return np.concatenate([another.reshape(-1, 1), predicts.reshape(-1, 1)], axis=-1)
        else:
            return predicts