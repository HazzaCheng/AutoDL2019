#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-12-10
import random
from functools import partial

import numpy as np
from sklearn.preprocessing import StandardScaler

from Configurations import MAX_VALID_PER_CLASS, SPEECH_SIMPLE_MODEL_EACH_LOOP_SAMPLE_NUM, \
    SPEECH_NEURAL_MODEL_EACH_LOOP_SAMPLE_NUM, SPEECH_NEURAL_MODEL_SAMPLE_MAX_RUN_LOOP, \
    SPEECH_SIMPLE_MODEL_SAMPLE_MAX_RUN_LOOP, \
    SPEECH_SIMPLE_MODEL_MAX_SAMPLE_NUM, MIN_VALID_PER_CLASS, MAX_VALID_SET_SIZE
from data_manager import BaseDataManager
from features.speech.speech_features import MAX_FRAME_NUM
from tools import log, get_max_length, pad_seq


class SpeechDataManager(BaseDataManager):
    def __init__(self, metadata, train_dataset, is_multilabel, dtype=None):
        super().__init__(metadata, train_dataset, is_multilabel, dtype)
        self._incremental_train_x_index = []

        self._pre_x = {}
        self._pre_train_x, self._pre_train_y = None, None
        self._pre_val_x, self._pre_val_y = None, None

        self._raw_data_max_length = get_max_length(self._all_x)

        self._feature_length = None
        self._spec_len = None
        self._feature_pre_func = None

    def get_raw_data(self):
        if self._val_x_index is None:
            all_index, sample_nums = np.arange(len(self._all_y)).tolist(), len(self._all_y)
            train_index = random.sample(all_index, int(sample_nums*0.9))
            val_index = list(set(all_index).difference(set(train_index)))
            self._train_x_index, self._val_x_index = train_index, val_index
        return self._all_x, self._all_y, self._all_x[self._val_x_index], self._all_y[self._val_x_index]

    def get_raw_data_max_length(self):
        return self._raw_data_max_length

    def get_raw_data_split_length(self):
        return self._raw_data_split_length

    def set_raw_data_split_length(self, l):
        self._raw_data_split_length = l

    def get_feature_pre_func(self):
        return self._feature_pre_func

    def set_feature_pre_func(self, func):
        self._feature_pre_func = func

    def get_feature_length(self):
        return self._feature_length

    def set_spec_len(self, l):
        self._spec_len = l

    def _fake_train_test_split(self, ratio=0.9):
        if self._is_multilabel:
            all_index, sample_nums = np.arange(len(self._all_y)).tolist(), len(self._all_y)
            train_index = random.sample(all_index, int(sample_nums*0.9))
            val_index = list(set(all_index).difference(set(train_index)))
            return train_index, val_index

        all_index, train_index, val_index = [], [], []
        max_val_per_class = min(MAX_VALID_PER_CLASS, int(MAX_VALID_SET_SIZE / self._num_classes))
        for i in range(self._num_classes):
            all_index.append(
                list(np.where(self._all_y[:, i] == 1)[0]))
        for i in range(self._num_classes):
            if len(all_index[i]) <= 0:
                continue
            tmp = random.sample(all_index[i],
                                max(MIN_VALID_PER_CLASS, int(len(all_index[i]) * (1 - ratio))))
            if len(tmp) > max_val_per_class:
                tmp = tmp[:max_val_per_class]
            val_index += tmp
            train_index += list(all_index[i])
        return train_index, val_index

    def _get_preprocess_data(self, sample_index, model_kind, pre_func):
        need_pre = set([i for i in sample_index if i not in self._pre_x])
        raw_data = [self._all_x[i] for i in need_pre]
        if model_kind == 0 or model_kind == 1:
            raw_data = [sample[:self._raw_data_split_length] for sample in raw_data]
        pre_data = pre_func(raw_data)
        if model_kind == 0 or model_kind == 1:
            if self._feature_length is None:
                self._feature_length = get_max_length(pre_data)
                self._feature_length = min(MAX_FRAME_NUM, self._feature_length)
                # pre_data = pad_seq(pre_data, pad_len=self._feature_length)
        log("Total {}, update {}".format(len(sample_index), len(need_pre)))

        if len(need_pre) > 0:
            if model_kind == 0:
                x_feas = []
                for i in range(len(pre_data)):
                    fea = np.mean(pre_data[i], axis=0).reshape(-1)
                    fea_std = np.std(pre_data[i], axis=0).reshape(-1)
                    x_feas.append(np.concatenate([fea, fea_std], axis=-1))
                x_feas = np.asarray(x_feas)
                scaler = StandardScaler()
                pre_data = scaler.fit_transform(x_feas[:, :])
            elif model_kind == 1:
                pre_data = pad_seq(pre_data, pad_len=self._feature_length)

            cnt = 0
            for i in need_pre:
                self._pre_x[i] = pre_data[cnt]
                cnt += 1

        x = [self._pre_x[i] for i in sample_index]
        y = [self._all_y[i] for i in sample_index]

        return x, y

    def get_train_data(self,
                       train_loop_num,
                       use_new_data,
                       model_kind=0,
                       is_need_incremental_sample=False):
        # split the valid dataset
        if train_loop_num == 1 and self._is_multilabel:
            self._train_x_index, self._val_x_index = self._train_test_split(ratio=0.9)
            self._init_each_class_index()
            self._pre_train_x = self._pre_train_y = self._pre_val_x = self._pre_val_y = None
        elif self._val_x_index is None or train_loop_num <= 3:
            if train_loop_num == 1 or train_loop_num == 2 or train_loop_num == 3:
                self._train_x_index, self._val_x_index = self._fake_train_test_split(ratio=0.9)
            else:
                self._train_x_index, self._val_x_index = self._train_test_split(ratio=0.9)
            self._init_each_class_index()
            self._pre_train_x = self._pre_train_y = self._pre_val_x = self._pre_val_y = None

        if use_new_data:
            if is_need_incremental_sample and model_kind == 1:
                self._init_even_class_index()
                self._incremental_train_x_index = []
            self._pre_train_x = self._pre_train_y = self._pre_val_x = self._pre_val_y = None
            self._raw_data_split_length = None
            self._feature_length = None
            self._pre_x = {}

        # incremental sampling
        if is_need_incremental_sample:
            if model_kind == 0:
                if train_loop_num == SPEECH_SIMPLE_MODEL_SAMPLE_MAX_RUN_LOOP:
                    sample_num = \
                        min(sum([len(self._each_class_index[i]) for i in range(self._num_classes)]),
                            SPEECH_SIMPLE_MODEL_MAX_SAMPLE_NUM)
                else:
                    sample_num = SPEECH_SIMPLE_MODEL_EACH_LOOP_SAMPLE_NUM * train_loop_num
                train_samples = self._get_samples_from_each_class(sample_num=sample_num)
            else:
                if train_loop_num == SPEECH_NEURAL_MODEL_SAMPLE_MAX_RUN_LOOP + SPEECH_SIMPLE_MODEL_SAMPLE_MAX_RUN_LOOP:
                    sample_num = sum([len(self._even_class_index[i]) for i in range(self._num_classes)])
                elif train_loop_num < SPEECH_NEURAL_MODEL_SAMPLE_MAX_RUN_LOOP + SPEECH_SIMPLE_MODEL_SAMPLE_MAX_RUN_LOOP:
                    sample_num = max(int(len(self._train_x_index) * (1 / SPEECH_NEURAL_MODEL_SAMPLE_MAX_RUN_LOOP)),
                                     SPEECH_NEURAL_MODEL_EACH_LOOP_SAMPLE_NUM)
                else:
                    sample_num = len(self._train_x_index)
                train_samples = self._get_samples_from_even_class(sample_num=sample_num)
                self._incremental_train_x_index.extend(train_samples)
                train_samples = self._incremental_train_x_index
        else:
            # sample_num = len(self._train_y)
            sample_num = SPEECH_NEURAL_MODEL_EACH_LOOP_SAMPLE_NUM
            train_samples = self._get_samples_from_each_class(sample_num)
        log("Train loop num {}, sample {}".format(train_loop_num, len(train_samples)))

        pre_func = partial(self._feature_pre_func)
        if len(train_samples) > 0:
            self._pre_train_x, self._pre_train_y = self._get_preprocess_data(train_samples, model_kind, pre_func)

        if self._pre_val_x is None:
            self._pre_val_x, self._pre_val_y = self._get_preprocess_data(self._val_x_index, model_kind, pre_func)
        # TODO need dtype?
        return np.asarray(self._pre_train_x), np.asarray(self._pre_train_y), \
               np.asarray(self._pre_val_x), np.asarray(self._pre_val_y)
