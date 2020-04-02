#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-01-28
import random
from abc import abstractmethod

import numpy as np

from Configurations import CLASS_NUM, MIN_VALID_PER_CLASS, MAX_VALID_PER_CLASS, MAX_VALID_SET_SIZE
from tools import get_max_length, log, timeit


class BaseDataManager:
    def __init__(self, metadata, train_dataset, is_multilabel, dtype='object'):
        self._metadata = metadata
        self._dtype = dtype
        self._is_multilabel = is_multilabel
        self._all_x, self._all_y = train_dataset
        self._all_x, self._all_y = np.array(self._all_x, dtype=self._dtype), np.array(self._all_y, dtype=np.int32)

        self._train_x_index = None
        self._val_x_index = None

        self._each_class_index = []
        self._even_class_index = []
        self._max_class_num, self._min_class_num = 0, 0

        self._num_classes = self._metadata[CLASS_NUM]

        self._raw_data_max_length = None
        self._raw_data_split_length = None

    def add_data(self, dataset):
        if dataset:
            x, y = dataset
            self._all_x = np.concatenate((self._all_x, np.array(x, dtype=self._dtype)), axis=0)
            self._all_y = np.concatenate((self._all_y, np.array(y, dtype=np.int32)), axis=0)
        # self._init_each_class_index()

    @timeit
    def _init_each_class_index(self):
        train_y = self._all_y[self._train_x_index]
        each_class_count = np.sum(np.array(train_y, dtype=np.int32), axis=0)
        self._max_class_num, self._min_class_num = \
            int(np.max(each_class_count)), \
            int(np.min(each_class_count))
        log("Raw train data: train_num(without val) {}; class_num {} ; max_class_num {}; min_class_num {}; "
            .format(len(self._train_x_index), self._num_classes, self._max_class_num, self._min_class_num))

        self._each_class_index = [[] for _ in range(self._num_classes)]
        for i in self._train_x_index:
            c = np.argmax(self._all_y[i])
            self._each_class_index[c].append(i)
        # for i in range(self._num_classes):
        #     self._each_class_index.append(list(np.where(train_y[:, i] == 1)[0]))

    @timeit
    def _init_even_class_index(self):
        train_y = self._all_y[self._train_x_index]
        self._even_class_index = []
        sample_per_class = max(int(len(train_y) / self._num_classes), 1)
        for i in range(self._num_classes):
            class_cnt = len(self._each_class_index[i])
            tmp = []
            if class_cnt == 0:
                pass
            elif class_cnt < sample_per_class:
                tmp = self._each_class_index[i] * \
                      int(sample_per_class / class_cnt)
                tmp += random.sample(
                    self._each_class_index[i],
                    sample_per_class - len(tmp))
            else:
                tmp += random.sample(
                    self._each_class_index[i], sample_per_class)
            random.shuffle(tmp)
            self._even_class_index.append(tmp)

    def _train_test_split(self, ratio=0.9):
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
            differ_set = set(all_index[i]).difference(set(tmp))
            # avoid some classes only have one sample
            if len(differ_set) == 0:
                differ_set = set(tmp)
            train_index += list(differ_set)
        return train_index, val_index

    def _get_samples_from_even_class(self, sample_num):
        per_class_num = max(int(sample_num / self._num_classes), 1)

        sample_indexs = []
        for i in range(self._num_classes):
            selected = self._even_class_index[i][:per_class_num]
            rest = self._even_class_index[i][per_class_num:]
            self._even_class_index[i] = rest
            sample_indexs += selected

        random.shuffle(sample_indexs)

        return sample_indexs

    def _get_samples_from_each_class(self, sample_num):
        per_class_num = max(int(sample_num / self._num_classes), 1)
        per_class_num = min(per_class_num, int(len(self._train_x_index) / self._num_classes))

        sample_indexs = []
        for i in range(self._num_classes):
            class_cnt = len(self._each_class_index[i])
            tmp = []
            if class_cnt == 0:
                pass
            elif class_cnt < per_class_num:
                tmp = self._each_class_index[i] * \
                      int(per_class_num / class_cnt)
                tmp += random.sample(
                    self._each_class_index[i],
                    per_class_num - len(tmp))
            else:
                tmp += random.sample(self._each_class_index[i], per_class_num)
            random.shuffle(tmp)
            sample_indexs.extend(tmp)
        random.shuffle(sample_indexs)
        return sample_indexs

    @timeit
    @abstractmethod
    def _get_preprocess_data(self, *args, **kwargs):
        pass

    @timeit
    @abstractmethod
    def get_train_data(self, *args, **kwargs):
        pass
