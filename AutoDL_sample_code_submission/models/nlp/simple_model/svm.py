#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2020/1/16 22:07
# @Author:  Mecthew
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler

from my_classifier import Classifier
from tools import timeit, ohe2cat, log
from sklearn.calibration import CalibratedClassifierCV


# Consider use LR as the first model because it can reach high point at
# first loop
class Svm(Classifier):
    def __init__(self):
        # TODO: init model, consider use CalibratedClassifierCV
        log("new {}".format(self.__class__.__name__))
        self._model = None
        self.is_init = False
        self._existed_classes = None
        self._num_classes = None

    def init_model(self,
                   num_classes,
                   kernel,
                   max_iter=500,
                   C=1.0,
                   **kwargs):
        # self._model = SVC(C=C, kernel=kernel, max_iter=max_iter, probability=True)
        self._model = CalibratedClassifierCV(LinearSVC(C=C, max_iter=max_iter, random_state=0))
        self.is_init = True
        self._num_classes = num_classes

    @timeit
    def preprocess_data(self, x, *args, **kwargs):
        return x

    @timeit
    def fit(self, x_train, y_train, *args, **kwargs):
        # sscaler = StandardScaler()
        # x_train = sscaler.fit_transform(x_train[:, :])
        self._existed_classes = set(ohe2cat(y_train))
        print("=== svm class {}".format(len(self._existed_classes)))
        self._model.fit(x_train, ohe2cat(y_train))

    def predict(self, x_test, batch_size=32, *args, **kwargs):
        # sscaler = StandardScaler()
        # x_test = sscaler.fit_transform(x_test[:, :])
        predicts = self._model.predict_proba(x_test)
        if len(predicts.shape) == 1:
            new_preds = np.zeros(shape=(predicts.shape[0], self._num_classes))
            new_preds[:, 1] = predicts
            new_preds[:, 0] = 1 - predicts
            return new_preds
        elif len(self._existed_classes) != self._num_classes:
            return self._make_predictions(predicts)
        else:
            return predicts

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
