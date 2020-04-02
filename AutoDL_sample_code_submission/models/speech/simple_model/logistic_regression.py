#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/10/5 10:35
# @Author:  Mecthew

import numpy as np
from sklearn.linear_model import logistic
from sklearn.preprocessing import StandardScaler

from my_classifier import Classifier
from tools import timeit, ohe2cat, log


# Consider use LR as the first model because it can reach high point at
# first loop
class LogisticRegression(Classifier):
    def __init__(self):
        # TODO: init model, consider use CalibratedClassifierCV
        log("new {}".format(self.__class__.__name__))

        self._existed_classes = None
        self._num_classes = None

        self._model = None
        self.is_init = False

    def init_model(self,
                   kernel,
                   num_classes,
                   max_iter=200,
                   C=1.0,
                   **kwargs):
        self._num_classes = num_classes
        if num_classes <= 5:
            class_weight = None
        else:
            class_weight = "balanced"
        self._model = logistic.LogisticRegression(
            C=C, max_iter=max_iter, solver='liblinear', multi_class='auto', class_weight=class_weight)
        self.is_init = True

    @timeit
    def preprocess_data(self, x, *args, **kwargs):
        x_feas = []
        for i in range(len(x)):
            fea = np.mean(x[i], axis=0).reshape(-1)
            fea_std = np.std(x[i], axis=0).reshape(-1)
            x_feas.append(np.concatenate([fea, fea_std], axis=-1))
        x_feas = np.asarray(x_feas)
        scaler = StandardScaler()
        x = scaler.fit_transform(x_feas[:, :])

        return x

    def fit(self, x_train, y_train, *args, **kwargs):
        print("=== lr fit {}".format(y_train.shape))
        self._existed_classes = set(ohe2cat(y_train))
        print("=== lr class {}".format(len(self._existed_classes)))
        self._model.fit(x_train, ohe2cat(y_train))

    def predict(self, x_test, batch_size=32, *args, **kwargs):
        preds = self._model.predict_proba(x_test)

        if self._num_classes != len(self._existed_classes):
            preds = self._make_predictions(preds)

        return preds

    def _make_predictions(self, preds):
        new_preds = np.zeros((preds.shape[0], self._num_classes))
        n = len(self._existed_classes)
        r = n / self._num_classes
        fake_pred = (1 - r) / (self._num_classes - n)

        cnt = 0
        for i in range(self._num_classes):
            if i in self._existed_classes:
                new_preds[:, i] = preds[:, cnt]
                cnt += 1
            else:
                new_preds[:, i] = fake_pred

        return new_preds
