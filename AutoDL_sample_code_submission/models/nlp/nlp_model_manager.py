#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-01-28
import gc

from Configurations import NLP_SIMPLE_NEURAl_MODEL_SEQUENCES, NLP_FEATURES_SEQUENCES, NLP_FAST_MODEL_RUN_LOOP
from features.nlp.nlp_features import get_embedding
from model_manager import BaseModelManager, auc_metric
from models.nlp import NLP_CNN, NLP_TEXT_CNN, NLP_GRU, NLP_GRU_ATTENTION, NLP_SVM, NLP_LR, NLP_MODEL_LIB, NLP_TF_MODEL
from tools import log
import keras.backend as K
import numpy as np


class NlpModelManager(BaseModelManager):
    def __init__(self,
                 meta,
                 data_manager,
                 is_multilabel,
                 is_use_fast_model,
                 is_use_simple_model,
                 keep_num=5,
                 each_model_keep_num=3,
                 each_model_top_k=2,
                 patience=3,
                 auc_threshold=0.5,
                 **kwargs):
        super().__init__(meta, data_manager, is_multilabel,
                         keep_num, each_model_keep_num, each_model_top_k, patience,
                         auc_threshold, **kwargs)

        self._model_kind = -1
        self._model_lib = NLP_MODEL_LIB

        self._pretrained_embedding_matrix = kwargs['pretrained_embedding_matrix'] \
            if 'pretrained_embedding_matrix' in kwargs else None
        self._is_use_pretrained_embedding = False
        self._embedding_dim = 64
        self._embedding_matrix = None

        self._is_batch_size_adaptive = True
        self._is_skip_gru = False
        self.is_read_rest_1 = False
        self.is_read_rest_2 = False

        self._max_simple_model_run_loop = 4
        self._max_model_run_loop = 8

        self._each_model_best_predict = [0] * 100
        self._best_predict = None
        self._valid_model_idx = None

        # segment type, 0 for tfidf, 1 for char level, 2 for word level
        self._seg_type = -1

        # for skipping model
        self.MAX_SEQ_LEN = 1600

        # first model，svm or lr
        self._fast_model = NLP_LR
        self._fast_model_feature = NLP_FEATURES_SEQUENCES['fast_model_feature']
        self._fast_model_run_loop = NLP_FAST_MODEL_RUN_LOOP
        self._is_use_fast_model = is_use_fast_model and not self._is_multilabel

        # simple model
        self._simple_model = None  # 这里用tf model 或者 text cnn
        self._simple_model_feature = NLP_FEATURES_SEQUENCES['simple_model_feature']
        self._is_use_simple_model = is_use_simple_model or self._is_multilabel
        # simple neural model
        self._simple_neural_model_sequences = NLP_SIMPLE_NEURAl_MODEL_SEQUENCES
        self._simple_neural_model_feature = NLP_FEATURES_SEQUENCES['simple_neural_model_feature']

    def _get_or_create_model(self):
        # use new model and not reset model, have to initialize the model
        if not self._model.is_init:
            log("get new model {}".format(self._model_name))
            # init model parameters
            if self._model_name in [NLP_CNN, NLP_TEXT_CNN, NLP_GRU, NLP_GRU_ATTENTION]:
                kwargs = {
                    'embedding_matrix': self._embedding_matrix,  # Can be None
                    'seq_len': self._data_manager.get_seq_len(),
                    'num_features': self._data_manager.get_num_features(),
                    'num_classes': self._num_classes,
                    'emb_size': self._embedding_dim,
                    'use_step_decay': True,
                    'is_multilabel': self._is_multilabel,
                    'is_balanced': self._data_manager.get_is_balance(),
                    'classes_ratio': self._data_manager.get_classes_ratio()
                }
            elif self._model_name == NLP_TF_MODEL:
                kwargs = {
                    'seq_len': self._data_manager.get_seq_len(),
                    'num_features': self._data_manager.get_num_features(),
                    'num_classes': self._num_classes,
                    'emb_size': self._embedding_dim,
                }
            elif self._model_name == NLP_SVM:
                kwargs = {
                    'num_classes': self._num_classes,
                    'kernel': 'linear',
                    'max_iter': 100
                }
            elif self._model_name == NLP_LR:
                kwargs = {
                    'num_classes': self._num_classes,
                    'C': 10,
                    'kernel': 'liblinear',
                    'max_iter': 100,
                    'is_balanced': self._data_manager.get_is_balance()
                }
            else:
                raise Exception("No such model!")
            if not self._model.is_init:
                self._model.init_model(**kwargs)
        log("This train loop use {}, last train loop use {}".format(self._model_name, self._last_model_name))

    def _pre_select_model_and_feature(self, train_loop_num):
        use_new_feature = False
        use_new_model = False
        self._use_new_data = False
        self._last_model_name = self._model_name

        # select first or simple model
        if self._model_kind == -1 and (train_loop_num == 1 or
                                       (self._cur_model_run_loop >= self._fast_model_run_loop and self._is_use_simple_model)):
            if train_loop_num == 1 and self._is_use_fast_model:
                log("use fast model {}".format(self._fast_model))
                self._model_name = self._fast_model
            else:
                if self._num_classes == 2 and self._data_manager.get_is_balance() and not self._is_multilabel:
                    log("simple model use tf model")
                    self._simple_model = NLP_TF_MODEL
                else:
                    log("simple model use cnn model")
                    self._simple_model = NLP_CNN
                self._model_name = self._simple_model
                self._model_kind = 0

            self.delete_model()
            self._model_num += 1
            self._cur_model_run_loop = 0
            self._not_rise_num = 0
            use_new_feature = True
            use_new_model = True

        # select neural model
        if self._not_rise_num == self._patience or \
                (self._model_kind == 0 and self._cur_model_run_loop >= self._max_simple_model_run_loop) or \
                (self._model_kind == 1 and self._cur_model_run_loop >= self._max_model_run_loop) or \
                (self._model_kind == -1 and self._is_use_fast_model and not self._is_use_simple_model
                 and self._cur_model_run_loop >= self._fast_model_run_loop):
            # clear session
            self.delete_model()
            # self._is_skip_gru = self._simple_neural_model_idx + 1 >= len(self._simple_neural_model_sequences) - 1\
            #                     and (self._data_manager.get_seq_len() > self.MAX_SEQ_LEN
            #                          or self._num_classes == 2 and not self._data_manager.get_is_balance())
            self._is_skip_gru = False

            use_new_model = True
            if self._model_kind == 0 or self._model_kind == -1:
                log("cur_model_run_loop: {}; patience: {}; stop simple model {}"
                    .format(self._cur_model_run_loop, self._not_rise_num, self._model_name))
                self._model_kind = 1
                self._all_model_best_auc = 0.02
                self._simple_neural_model_feature_idx += 1
                self._simple_neural_model_idx = -1
                use_new_feature = True
                # to avoid self._round_num increase twice
                if not self._is_skip_gru:
                    self._round_num += 1

            # must ensure get_seq_len() has been initialized
            if self._simple_neural_model_idx >= len(self._simple_neural_model_sequences) - 1 or self._is_skip_gru:
                self._round_num += 1
                self._simple_neural_model_feature_idx += 1
                self._simple_neural_model_idx = -1
                use_new_feature = True

            self._simple_neural_model_idx += 1
            self._model_name = self._simple_neural_model_sequences[self._simple_neural_model_idx]
            self._model_num += 1
            if self._valid_model_idx is None:
                self._valid_model_idx = self._model_num

            log("round{}; cur_model_run_loop: {}; patience: {}; change model from {} to {}; model_num {}"
                .format(self._round_num, self._cur_model_run_loop, self._not_rise_num,
                        self._last_model_name, self._model_name, self._model_num))

            self._cur_model_run_loop = 0
            self._not_rise_num = 0

        if use_new_feature:
            # all features are used, finish training
            if self._simple_neural_model_feature_idx >= len(self._simple_neural_model_feature):
                return True
            model_feature = None
            if self._model_kind == 0:
                model_feature = self._simple_model_feature
            elif self._model_kind == 1:
                model_feature = self._simple_neural_model_feature[self._simple_neural_model_feature_idx]
            elif self._model_kind == -1:
                model_feature = self._fast_model_feature

            self._seg_type = model_feature['level']
            self._use_new_data = model_feature['is_use_new_data']
            if self._use_new_data:
                self._test_x = None
            self._embedding_dim = model_feature['embedding_dim']
            self._is_use_pretrained_embedding = model_feature['is_pretrained_embedding']

            feature_name = model_feature['name']
            log("round num {}; feature_name {}; seg_type {};"
                " is_use_pretrained_embedding {}; embedding_dim {}; is_use_new_data {}"
                .format(self._round_num,
                        feature_name,
                        self._seg_type,
                        self._is_use_pretrained_embedding,
                        self._embedding_dim,
                        self._use_new_data))

        # TODO predicts
        if use_new_model:
            self._cur_model_max_auc = -1
            self._model = self._model_lib[self._model_name]()

        return False

    def _get_batch_size(self, batch_size):
        seq_len = self._data_manager.get_seq_len()
        emb_size = self._embedding_dim
        out_size = 128

        N = 7633305600 / (seq_len * emb_size * out_size)

        batch_size = min(batch_size, N)
        batch_size = max(batch_size, 4)

        return int(batch_size)

    def delete_model(self):
        if self._model:
            del self._model
            gc.collect()
            K.clear_session()
            self._model = None

    def _predict_ensemble(self):
        # size = self._model_num + 1
        size = self._model_num
        if self._valid_model_idx:
            return np.mean(self._each_model_best_predict[self._valid_model_idx:size], axis=0)
        else:
            raise Exception("self._valid_model_idx not initialized")
        # donot ensemble sample data model
        # if not self._is_use_fast_model:
        #     return np.mean(self._each_model_best_predict[2:size], axis=0)
        # else:
        #     return np.mean(self._each_model_best_predict[3:size], axis=0)

    def fit(self, train_loop_num=1, remaining_time_budget=None):
        is_done = self._pre_select_model_and_feature(train_loop_num)
        self._cur_model_run_loop += 1
        if is_done:
            return True
        # skip the first loop and use x_test to fit tfidf_vectorizer
        if train_loop_num == 1 and self._is_use_fast_model:
            return False

        train_x, val_x, train_y, val_y = self._data_manager.get_train_data(train_loop_num=train_loop_num,
                                                                           use_new_data=self._use_new_data,
                                                                           model_kind=self._model_kind,
                                                                           deal_seg=self._seg_type)

        self._val_set = (val_x, val_y)
        self._input_shape = train_x.shape

        # num_features is computed after get_train_data
        if self._is_use_pretrained_embedding:
            self._embedding_matrix = get_embedding(self._pretrained_embedding_matrix,
                                                   self._data_manager.get_num_features(),
                                                   self._data_manager.get_word_index())
        else:
            self._embedding_matrix = None
        self._get_or_create_model()

        batch_size = 32
        if self._is_batch_size_adaptive and self._model_kind != -1:
            # change batch size
            sample_nums = len(train_y)
            e = self._cur_model_run_loop
            batch_size = 64
            # 根据样本数和epoch数调整batch size
            if sample_nums > 10000:
                if e < 4:
                    batch_size = (6 - e) * 32 * int(sample_nums / 10000)  # 256
                else:
                    batch_size = 16 * int(sample_nums / 6000)  # 256  10000

                batch_size = min(batch_size, 2048)
                batch_size = max(batch_size, 32)

            if e == 1 and sample_nums > 5000:
                batch_size = max(batch_size, 128)
            batch_size = self._get_batch_size(batch_size)

        log("fit {} for {} times; train_x: {}; train_y: {}; val_x: {}; val_y: {}; batch_size {}".format(
            self._model_name,
            self._cur_model_run_loop,
            train_x.shape,
            train_y.shape,
            val_x.shape,
            val_y.shape,
            batch_size))

        self._model.fit(train_x, train_y,
                        validation_data_fit=(val_x, val_y),
                        epochs=1,
                        cur_model_run_loop=self._cur_model_run_loop,
                        batch_size=batch_size)

        return False

    def predict(self, test_x):
        is_best = False

        # ensemble previous model's best predictions, donot ensemble sample data model
        if self._cur_model_run_loop == 1 and (self._valid_model_idx is not None and self._model_num > self._valid_model_idx):
            log("model_num {}; ensemble predicts".format(self._model_num))
            self._best_predict = self._predict_ensemble()

        if self._cur_model_run_loop == 1:
            # batch_size = 32
            # if self._is_batch_size_adaptive:
            #     batch_size = self._get_batch_size(1024)
            # cur_auc = auc_metric(np.array(y_val), self._model.predict(x_val, batch_size=batch_size))
            # log("{} first auc {}".format(self._model_name, cur_auc))
            cur_auc = 0.001
        else:
            x_val, y_val = self._val_set
            batch_size = 32
            if self._is_batch_size_adaptive and self._model_kind != -1:
                batch_size = self._get_batch_size(1024)
            cur_auc = auc_metric(np.array(y_val), self._model.predict(x_val, batch_size=batch_size))

        if cur_auc > self._cur_model_max_auc:
            self._not_rise_num = 0
            self._cur_model_max_auc = cur_auc
            if cur_auc > self._all_model_best_auc:
                is_best = True
                self._all_model_best_auc = cur_auc
        else:
            self._not_rise_num += 1

        # judge whether to read rest data
        if (self._model_kind == 0 and (self._cur_model_run_loop >= self._max_simple_model_run_loop or self._not_rise_num >= self._patience)) \
                or (self._model_kind == -1 and not self._is_use_simple_model and self._cur_model_run_loop >= self._fast_model_run_loop):
            self.is_read_rest_1 = True
        if self._model_kind == 1 \
                and (self._cur_model_run_loop >= self._max_model_run_loop or self._not_rise_num >= self._patience) \
                and (not self._is_skip_gru and self._simple_neural_model_idx >= len(self._simple_neural_model_sequences) - 1
                    or self._is_skip_gru and self._simple_neural_model_idx >= len(self._simple_neural_model_sequences) - 2):
            self.is_read_rest_2 = True

        log("val aucs {}, best auc {}".format(cur_auc, self._all_model_best_auc))
        if self._not_rise_num > 0 and self._model_kind >= 0:
            # not rise, return previous best predictions
            log("cur_run_loop {}; cur_auc {} cur_model_max_auc {} all_model_best_auc {};"
                " {} not rise for {} times, don't predict"
                .format(self._cur_model_run_loop,
                        cur_auc, self._cur_model_max_auc, self._all_model_best_auc,
                        self._model_name, self._not_rise_num))
            return self._best_predict

        if self._test_x is None:
            self._test_x = self._data_manager.preprocess_test_data(test_x, self._seg_type)
        x_test = self._test_x
        if self._model_kind == -1 and self._cur_model_run_loop == 1:  # 第一轮使用x_test来fit tfidf_vectorizer，predict随机生成
            return np.random.rand(len(x_test), self._num_classes)

        batch_size = 32
        if self._is_batch_size_adaptive and self._model_kind != -1:
            batch_size = 32 * int(len(x_test) / 2000)
            batch_size = min(batch_size, 2048)
            batch_size = max(batch_size, 32)

            batch_size = self._get_batch_size(batch_size)

        preds = self._model.predict(x_test, batch_size=batch_size)

        log("cur_run_loop {}; cur_auc {} cur_model_max_auc {} all_model_best_auc {};"
            " {} rise, predict, batch_size {}"
            .format(self._cur_model_run_loop,
                    cur_auc, self._cur_model_max_auc, self._all_model_best_auc,
                    self._model_name, batch_size))

        self._each_model_best_predict[self._model_num] = preds

        if self._model_kind < 0:
            self._best_predict = preds
        else:
            if is_best:
                self._best_predict = preds
            else:
                preds = self._best_predict

        return preds
