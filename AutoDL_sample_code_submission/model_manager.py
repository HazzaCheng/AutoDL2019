#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-01-28
from abc import abstractmethod

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from Configurations import CLASS_NUM
from tools import timeit


def auc_metric(solution, prediction):
    solution = np.array(solution, dtype=np.float)
    prediction = np.array(prediction, dtype=np.float)
    if solution.sum(axis=0).min() == 0:
        return 0
    auc = roc_auc_score(solution, prediction, average='macro')
    return np.mean(auc * 2 - 1)


def acc_metric(solution, prediction):
    if solution.sum(axis=0).min() == 0:
        return np.nan
    acc = accuracy_score(solution, prediction)
    return acc


class BaseModelManager:
    def __init__(self,
                 meta,
                 data_manager,
                 is_multilabel,
                 keep_num=5,
                 each_model_keep_num=3,
                 each_model_top_k=2,
                 patience=3,
                 auc_threshold=0.5,
                 **kwargs):
        self.metadata = meta
        self._num_classes = self.metadata[CLASS_NUM]
        self._data_manager = data_manager
        self._is_multilabel = is_multilabel

        self._keep_num = keep_num
        self._each_model_keep_num = each_model_keep_num
        self._each_model_top_k = each_model_top_k
        self._patience = patience
        self._not_rise_num = 0

        self._input_shape = None
        self._model = None
        self._model_name = None
        self._models = {}
        self._model_lib = None
        self._last_model_name = None
        self._cur_model_run_loop = 0
        self._model_num = 0
        self._round_num = 0

        self._val_set = None
        self._test_x = None

        self._use_new_data = False
        self._is_reset_model = False

        self._k_best_predicts = [-1] * self._keep_num
        self._k_best_auc = [-1.1] * self._keep_num
        self._each_model_best_predict = {}
        self._each_model_best_auc = {}
        self._cur_model_max_auc = -1
        self._all_model_best_auc = -1
        self._auc_threshold = auc_threshold

        self._max_simple_model_run_loop = None
        self._round0_max_model_run_loop = None
        self._max_model_run_loop = None
        self._neural_model_train_epochs = -1

        # model kind
        # 0: simple model 1: simple neural model 2: complex neural model
        self._model_kind = 0

        self._feature_name = None
        # simple model
        self._simple_model = None
        self._simple_model_feature = None
        # simple neural model
        self._simple_neural_model_sequences = None
        self._simple_neural_model_idx = -1
        self._simple_neural_model_feature = None
        self._simple_neural_model_feature_idx = -1
        # complex neural model
        self._complex_neural_model = None
        self._complex_neural_model_round = 0
        self._is_complex_neural_model_init = False
        self._use_complex_neural_model_predict = False

    @abstractmethod
    def _get_or_create_model(self):
        pass

    @abstractmethod
    def _pre_select_model_and_feature(self, train_loop_num):
        pass

    def _get_each_model_top_k_predicts(self):
        predicts = []
        for k, v in self._each_model_best_auc.items():
            if k == self._simple_model:
                continue
            k_predicts = np.array(self._each_model_best_predict[k])
            temp = [(auc, k_predicts[i]) for i, auc in enumerate(v)
                    if auc > max(self._auc_threshold, self._k_best_auc[0] - 0.1)]
            temp.sort(key=lambda x: x[0], reverse=True)
            predicts.extend(temp[:self._each_model_top_k])

        if len(predicts) == 0:
            return [], []

        predicts = sorted(predicts, key=lambda x: x[0], reverse=True)[
                   :self._each_model_keep_num]
        top_k_aucs = [predicts[i][0] for i in range(len(predicts))]
        top_k_predicts = [predicts[i][1] for i in range(len(predicts))]

        return top_k_aucs, top_k_predicts

    @abstractmethod
    def _predict_ensemble(self):
        pass

    @abstractmethod
    @timeit
    def fit(self, train_loop_num=1, remaining_time_budget=None):
        pass

    @abstractmethod
    @timeit
    def predict(self, test_x):
        pass
