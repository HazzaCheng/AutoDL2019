#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-01-28
import random
import time

from sklearn.model_selection import train_test_split

from data_manager import BaseDataManager
from tools import get_mean_length, log, ohe2cat, timeit
from Configurations import MAX_VALID_PER_CLASS, NLP_TFIDF_TRAIN_NUM
import numpy as np

from features.nlp import ac
from features.nlp.nlp_features import vectorize_texts


class NlpDataManager(BaseDataManager):
    def __init__(self, metadata, train_dataset, is_multilabel, stopwords=None, vocabulary=None, dtype='object'):
        super().__init__(metadata, train_dataset, is_multilabel, dtype)

        self.x_train, self.x_val, self.y_train, self.y_val = None, None, None, None

        self._data_lang = metadata['language']

        self._raw_data_max_length = 1200
        self._raw_data_split_length = get_mean_length(self._all_x)
        self._max_doc_len = 6000

        self._word_index = None
        self._max_length = None
        self._seq_len = None
        self._num_features = None

        self._classes_ratio = None
        self._is_balance = self._check_is_blance()
        self._tfidf_vectorizer = None
        self._vocabulary = vocabulary
        self._stopwords = stopwords
        self._pre_tfidf = {}

    def get_sample_num(self):
        if self._all_y is None:
            return 0
        else:
            return self._all_y.shape[0]

    def get_word_index(self):
        return self._word_index

    def get_num_features(self):
        return self._num_features

    def get_is_balance(self):
        return self._is_balance

    def get_classes_ratio(self):
        return self._classes_ratio

    def get_seq_len(self):
        return self._seq_len

    def reset_dataset(self, dataset):
        if dataset:
            x, y = dataset
            self._all_x = np.array(x, dtype=self._dtype)
            self._all_y = np.array(y, dtype=self._dtype)

    @timeit
    def _check_is_blance(self):
        vals = np.sum(self._all_y, axis=0)
        # min_val = min(vals[0], vals[1])
        # ratio = min_val / (vals[0] + vals[1])
        min_val = np.min(vals)
        ratio = min_val / np.sum(vals)
        self._classes_ratio = vals / np.sum(vals, dtype=np.float)

        print('vals: ', vals)
        print('min_val:', min_val)
        print('ratio:', ratio)

        return ratio >= 0.01

    @timeit
    def _get_preprocess_data(self, data, data_lang, deal_seg, sentence_len):
        data = np.array(data, dtype='object')

        s1 = time.time()
        t1 = time.time()
        # For TF-IDF
        if deal_seg == 0:
            res, self._tfidf_vectorizer = vectorize_texts(lang=self._data_lang, x=data,
                                                          tfidf_vectorizer=self._tfidf_vectorizer,
                                                          stopwords=self._stopwords)
            return res, None, None, None
        # char level
        elif deal_seg == 1:
            if data_lang == 'ZH':
                data = ac.clean_text_zh_seg1(data, sentence_len)
            elif data_lang == 'EN':
                data = ac.clean_text_en_seg1(data, sentence_len)
        # word level
        elif deal_seg == 2:
            if data_lang == 'ZH':
                data = ac.clean_text_zh_seg2(data, sentence_len)
            elif data_lang == 'EN':
                data = ac.clean_text_en_seg2(data, sentence_len)

        t2 = time.time()

        num_sentence = len(data)
        t = np.array(data, dtype='object')
        MAX_VOCAB_SIZE, MAX_SEQ_LENGTH, word2index, text_lens = ac.bulid_index(t, num_sentence)

        print('*****************************DataNum:', num_sentence)
        print('*****************************DataLen:', np.mean(text_lens))

        t3 = time.time()
        max_length = MAX_SEQ_LENGTH
        res = ac.texts_to_sequences_and_pad(t, num_sentence, word2index, max_length, text_lens, 0)
        num_features = min(len(word2index) + 1, MAX_VOCAB_SIZE)
        t4 = time.time()
        print('###clean ', t2 - t1, 's')
        print('###build', t3 - t2, 's')
        print('###seq', t4 - t3, 's')

        s2 = time.time()
        print('###init data tot use time ', s2 - s1, 's')

        log("process train data; data num {}; data_lang {}; sentence_len {}; max_length {}; deal_seg {}"
            .format(len(data), data_lang, sentence_len, max_length, deal_seg))

        return res, word2index, num_features, max_length

    @timeit
    def get_train_data(self, train_loop_num, use_new_data, model_kind, deal_seg):
        is_simple_model = model_kind == 0 and use_new_data

        if use_new_data:
            # self._raw_data_split_length = get_mean_length(self._all_x)
            self._is_balance = self._check_is_blance()
            self.x_train, self.x_val, self.y_train, self.y_val = None, None, None, None

        #  fast model need sample at each loop
        if model_kind == -1:
            train_index, val_index = self._tfidf_train_val_split(x=self._all_x,
                                                                 y=self._all_y,
                                                                 val_ratio=0.1,
                                                                 max_train_num_per_class=NLP_TFIDF_TRAIN_NUM // self._num_classes)
            self.x_train, self.y_train, self.x_val, self.y_val = self._get_preprocess_tfidf(train_index, val_index)
            return self.x_train, self.x_val, self.y_train, self.y_val
        # simple model need sample
        elif is_simple_model:
            self._pre_tfidf = None  # release dict space
            t1 = time.time()
            self._raw_data_split_length = min(self._raw_data_split_length, self._raw_data_max_length)

            len_mean_for_compute = max(100, self._raw_data_split_length)
            sample_num = int(-90.8 * len_mean_for_compute + 128960)
            MAX_SAMPLE_ROW = 100000
            MIN_SAMPLE_ROW = 16666
            sample_num = min(sample_num, MAX_SAMPLE_ROW)
            sample_num = max(sample_num, MIN_SAMPLE_ROW)
            sample_num = min(sample_num, len(self._all_y))
            log("Use sample data; sample num {}; cut length {};".format(sample_num, self._raw_data_split_length))

            t10 = time.time()
            index = [i for i in range(len(self._all_y))]
            np.random.shuffle(index)
            index = index[0: sample_num]
            x_train = self._all_x[index]
            y_train = self._all_y[index]

            t11 = time.time()
            sentence_len = self._raw_data_split_length
            x_train, word_index, num_features, max_length = self._get_preprocess_data(x_train,
                                                                                      self._data_lang,
                                                                                      deal_seg,
                                                                                      sentence_len)
            t2 = time.time()
            self._word_index = word_index
            self._max_length = max_length
            self._num_features = num_features
            self.x_train, self.y_train, self.x_val, self.y_val = self._nlp_train_val_split(x=x_train, y=y_train,
                                                                                           val_ratio=0.1)
            self._seq_len = len(x_train[0])
            t3 = time.time()
            log("before preprocess {}s; preprocess data {}s; train_test_split {}s".format(t11 - t10, t2 - t11, t3 - t2))
            log("vocab_size {} in char".format(self._num_features))
        elif use_new_data:  # 使用fast model时每一轮都得重新切分
            self._pre_tfidf = None  # release dict space
            # without cut
            self._raw_data_split_length = self._max_doc_len
            sentence_len = self._raw_data_split_length

            x_train, word_index, num_features, max_length = self._get_preprocess_data(self._all_x,
                                                                                      self._data_lang,
                                                                                      deal_seg,
                                                                                      sentence_len)
            self._word_index = word_index
            self._max_length = max_length
            self._num_features = num_features
            self.x_train, self.y_train, self.x_val, self.y_val = self._nlp_train_val_split(x=x_train, y=self._all_y,
                                                                                           val_ratio=0.1)
            self._seq_len = len(x_train[0])
            log("vocab_size {} in word".format(self._num_features))
        return self.x_train, self.x_val, self.y_train, self.y_val

    @timeit
    def preprocess_test_data(self, data, deal_seg):
        data = np.array(data, dtype='object')

        data_lang = self._data_lang
        sentence_len = self._raw_data_split_length
        log("cut_len {}, max_len {}".format(sentence_len, self._max_length))
        # For TF-IDF
        if deal_seg == 0:
            res, self._tfidf_vectorizer = vectorize_texts(lang=self._data_lang, x=data,
                                                          tfidf_vectorizer=self._tfidf_vectorizer,
                                                          stopwords=self._stopwords)
            print("tfidf shape: {}".format(np.array(res).shape))
            return res
        elif deal_seg == 1:
            if data_lang == 'ZH':
                data = ac.clean_text_zh_seg1(data, sentence_len)
            elif data_lang == 'EN':
                data = ac.clean_text_en_seg1(data, sentence_len)
        elif deal_seg == 2:
            if data_lang == 'ZH':
                data = ac.clean_text_zh_seg2(data, sentence_len)
            elif data_lang == 'EN':
                data = ac.clean_text_en_seg2(data, sentence_len)
        num_sentence = len(data)
        t = np.array(data, dtype='object')
        word2index = self._word_index
        max_length = self._max_length
        res = ac.texts_to_sequences_and_pad(t, num_sentence, word2index, max_length, None, 1)

        log("process test data; data num {}; data_lang {}; sentence_len {}; max_length {}; deal_seg {}"
            .format(len(data), data_lang, sentence_len, self._max_length, deal_seg))

        return res

    @timeit
    def _get_preprocess_tfidf(self, train_index, val_index):
        train_need_pre_idx = [idx for idx in train_index if idx not in self._pre_tfidf]
        res, self._tfidf_vectorizer = vectorize_texts(lang=self._data_lang, x=self._all_x[train_need_pre_idx],
                                                      tfidf_vectorizer=self._tfidf_vectorizer,
                                                      stopwords=self._stopwords)
        for idx, pre_x in zip(train_need_pre_idx, res):
            self._pre_tfidf[idx] = pre_x

        val_need_pre_idx = [idx for idx in val_index if idx not in self._pre_tfidf]
        res, self._tfidf_vectorizer = vectorize_texts(lang=self._data_lang, x=self._all_x[val_need_pre_idx],
                                                      tfidf_vectorizer=self._tfidf_vectorizer,
                                                      stopwords=self._stopwords)
        for idx, pre_x in zip(val_need_pre_idx, res):
            self._pre_tfidf[idx] = pre_x

        x_train, y_train = [self._pre_tfidf[idx] for idx in train_index], self._all_y[train_index]
        x_val, y_val = [self._pre_tfidf[idx] for idx in val_index], self._all_y[val_index]
        return np.array(x_train, dtype=np.float), np.array(y_train, dtype=np.int),\
               np.array(x_val, dtype=np.float), np.array(y_val, dtype=np.int)

    def _nlp_train_val_split(self, x, y, val_ratio=0.1):
        if self._is_multilabel:
            all_index, sample_nums = np.arange(len(y)).tolist(), len(y)
            train_index = random.sample(all_index, int(sample_nums * (1 - val_ratio)))
            val_index = list(set(all_index).difference(set(train_index)))
            return x[train_index], y[train_index], x[val_index], y[val_index]

        all_index, train_index, val_index = [], [], []
        max_val_per_class = 200
        for i in range(self._num_classes):
            all_index.append(
                list(np.where(y[:, i] == 1)[0]))
        for i in range(self._num_classes):
            if len(all_index[i]) <= 0:
                continue
            tmp = random.sample(all_index[i],
                                max(1, int(len(all_index[i]) * val_ratio)))
            if len(tmp) > max_val_per_class:
                tmp = tmp[:max_val_per_class]
            val_index += tmp
            differ_set = set(all_index[i]).difference(set(tmp))
            # avoid some classes only have one sample
            if len(differ_set) == 0:
                differ_set = set(tmp)
            train_index += list(differ_set)
        x_train, y_train, x_val, y_val = x[train_index], y[train_index], x[val_index], y[val_index]
        print("train class {}; val class {}".format(np.sum(y_train, axis=0), np.sum(y_val, axis=0)))
        return x_train, y_train, x_val, y_val

    def _tfidf_train_val_split(self, x, y, val_ratio=0.1, max_train_num_per_class=1000):
        if self._is_multilabel:
            all_index, sample_nums = np.arange(len(y)).tolist(), len(y)
            train_index = random.sample(all_index, int(sample_nums * (1 - val_ratio)))
            val_index = list(set(all_index).difference(set(train_index)))
            return train_index, val_index

        all_index, train_index, val_index = [], [], []
        max_val_per_class = 200
        # max_train_num_per_class = min(NLP_MAX_TRAIN_PER_CLASS, max_train_num_per_class)
        for i in range(self._num_classes):
            all_index.append(
                list(np.where(y[:, i] == 1)[0]))
        for i in range(self._num_classes):
            if len(all_index[i]) <= 0:
                continue
            tmp = random.sample(all_index[i],
                                max(1, int(len(all_index[i]) * val_ratio)))
            if len(tmp) > max_val_per_class:
                tmp = tmp[:max_val_per_class]
            val_index += tmp
            differ_set = set(all_index[i]).difference(set(tmp))
            # avoid some classes only have one sample
            if len(differ_set) == 0:
                differ_set = set(tmp)
            diff_list = list(differ_set)
            if diff_list.__len__() > max_train_num_per_class:
                begin_idx = np.random.randint(0, diff_list.__len__() - max_train_num_per_class)
                diff_list = diff_list[begin_idx: begin_idx + max_train_num_per_class]
            train_index += diff_list
        x_train, y_train, x_val, y_val = x[train_index], y[train_index], x[val_index], y[val_index]
        print("train class {}; val class {}".format(np.sum(y_train, axis=0), np.sum(y_val, axis=0)))
        return train_index, val_index
