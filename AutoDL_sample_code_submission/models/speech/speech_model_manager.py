#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-22
import gc
import numpy as np
from keras import backend as K

from Configurations import CLASS_NUM, SPEECH_SIMPLE_MODEL_SAMPLE_MAX_RUN_LOOP, SPEECH_SIMPLE_NEURAl_MODEL_SEQUENCES, \
    SPEECH_SIMPLE_MODEL, SPEECH_FEATURES_SEQUENCES, SPEECH_MODEL_MAX_RUN_LOOP, SPEECH_ROUND0_MODEL_MAX_RUN_LOOP
from features.speech.speech_features import get_specified_feature_func, get_features_data, AUDIO_SAMPLE_RATE
from model_manager import auc_metric, BaseModelManager
from models.speech import *
from tools import log


class SpeechModelManager(BaseModelManager):
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
        super().__init__(meta, data_manager, is_multilabel,
                         keep_num, each_model_keep_num, each_model_top_k, patience,
                         auc_threshold, **kwargs)
        self._model_lib = SPEECH_MODEL_LIB

        self._max_simple_model_run_loop = SPEECH_SIMPLE_MODEL_SAMPLE_MAX_RUN_LOOP
        self._round0_max_model_run_loop = SPEECH_ROUND0_MODEL_MAX_RUN_LOOP
        self._max_model_run_loop = SPEECH_MODEL_MAX_RUN_LOOP

        # simple model
        self._simple_model = SPEECH_SIMPLE_MODEL
        self._simple_model_feature = SPEECH_FEATURES_SEQUENCES['simple_model_feature']
        # simple neural model
        self._simple_neural_model_sequences = SPEECH_SIMPLE_NEURAl_MODEL_SEQUENCES
        self._simple_neural_model_feature = SPEECH_FEATURES_SEQUENCES['simple_neural_model_feature']
        # complex neural model
        self._complex_neural_model = THIN_RESNET34_MODEL
        self._complex_neural_model_round = 0

        self._lr_preds = None
        self._lr_auc = -1
        self._ts = 0.15
        self._keep_num = 3

    def _get_or_create_model(self):
        # use new model and not reset model, have to initialize the model
        if self._model_kind != 2:
            if not self._model.is_init:
                log("get new model {}".format(self._model_name))
                # init model parameters
                if self._model_name == CNN2D_MODEL:
                    kwargs = {
                        'input_shape': self._input_shape[1:],
                        'num_classes': self.metadata[CLASS_NUM],
                        'max_layer_num': 10
                    }
                elif self._model_name in [LSTM_MODEL, BILSTM_MODEL, CRNN2D_MODEL, ATT_GRU_MODEL]:
                    kwargs = {
                        'input_shape': self._input_shape[1:],
                        'num_classes': self.metadata[CLASS_NUM],
                        'is_multilabel': self._is_multilabel
                    }
                elif self._model_name == SVM_MODEL:
                    kwargs = {
                        'kernel': 'linear',
                        'max_iter': 1000
                    }
                elif self._model_name == LR_MODEL:
                    kwargs = {
                        'kernel': 'liblinear',
                        'num_classes': self.metadata[CLASS_NUM],
                        'max_iter': 100
                    }
                else:
                    raise Exception("No such model!")
                self._model.init_model(**kwargs)
        else:
            if self._model_name == THIN_RESNET34_MODEL and not self._is_complex_neural_model_init:
                self._model = self._model_lib[self._model_name](self.metadata, self._is_multilabel)
                self._is_complex_neural_model_init = True
            else:
                self._last_model_name = self._model_name

        log("This train loop use {}, last train loop use {}".format(self._model_name, self._last_model_name))

    def _pre_select_model_and_feature(self, train_loop_num):
        self._last_model_name = self._model_name

        if train_loop_num == 1 or self._model_name is None and not self._is_multilabel:
            self._model_name = self._simple_model
            self._model_kind = 0
            self._each_model_best_auc[self._model_name] = [-1]
            self._each_model_best_predict[self._model_name] = [-1]
            self._use_new_data = True
        else:
            self._use_new_data = False

        if self._not_rise_num == self._patience \
                or (self._model_kind == 0 and self._cur_model_run_loop >= self._max_simple_model_run_loop) \
                or (self._round_num == 0 and self._cur_model_run_loop >= self._round0_max_model_run_loop) \
                or (self._model_kind == 1 and self._cur_model_run_loop >= self._max_model_run_loop) \
                or (train_loop_num == 1 and self._is_multilabel):
            self._not_rise_num = 0
            if self._model_kind == 0:
                self._model_kind = 1

            # add round num
            if self._model_kind == 1 and self._simple_neural_model_idx == len(self._simple_neural_model_sequences) - 1:
                self._round_num += 1
                self._simple_neural_model_idx = -1

            if self._round_num == self._complex_neural_model_round:
                self._model_kind = 2
                self._is_reset_model = True

            if self._model_kind == 2:
                self._model_name = self._complex_neural_model
                self._cur_model_run_loop = 0
            elif self._model_kind == 1:
                self._simple_neural_model_idx += 1
                if self._simple_neural_model_idx == 0 and self._round_num > 0:
                    # sort model sequences by auc, desc
                    self._simple_neural_model_sequences = list(sorted(self._simple_neural_model_sequences,
                                                                      key=lambda x: self._each_model_best_auc[x][-1],
                                                                      reverse=True))
                    log("round {} start, model sequences {}".format(self._round_num,
                                                                    self._simple_neural_model_sequences[
                                                                    self._simple_neural_model_idx:]))
                self._model_name = self._simple_neural_model_sequences[self._simple_neural_model_idx]
                self._model_num += 1
                log("change model from {} to {}, loop_num: {}".format(self._last_model_name, self._model_name,
                                                                      self._cur_model_run_loop))
                self._use_new_data = self._simple_neural_model_idx == 0 and (
                        self._round_num == 0 or self._round_num > 1)
                self._is_reset_model = self._simple_neural_model_idx == 0 and self._round_num > 1
                self._cur_model_run_loop = 0

            if self._use_new_data:
                self._test_x = None

            if (self._round_num == 0 or self._round_num == self._complex_neural_model_round) \
                    and self._cur_model_run_loop == 0:
                self._each_model_best_auc[self._model_name] = [-1]
                self._each_model_best_predict[self._model_name] = [-1]
                self._cur_model_max_auc = -1
            elif self._round_num == 1 and self._round_num < self._complex_neural_model_round \
                    and self._cur_model_run_loop == 0:
                self._cur_model_max_auc = self._each_model_best_auc[self._model_name][-1]
            elif self._round_num >= 2 \
                    and self._cur_model_run_loop == 0:
                self._each_model_best_auc[self._model_name] += [-1]
                self._each_model_best_predict[self._model_name] += [-1]
                self._cur_model_max_auc = -1

            if self._is_reset_model:
                log("new round {}".format(self._round_num))
                # clear all models
                self._models.clear()
                del self._model
                self._model = None
                gc.collect()
                K.clear_session()

        # change epochs
        if self._round_num == 0:
            self._neural_model_train_epochs = 5
        elif self._round_num == 1:
            self._neural_model_train_epochs = 10
        else:
            self._neural_model_train_epochs = 15

        # change split length and feature
        if not self._model_kind == 2 and \
                self._cur_model_run_loop == 0 and \
                (self._model_kind == 0 or (self._simple_neural_model_idx == 0 and self._round_num != 1)):
            if self._model_kind == 0:
                feature_name = self._simple_model_feature['name']
                max_duration = self._simple_model_feature['max_duration']
                kwargs = self._simple_model_feature['kwargs']
            else:
                self._simple_neural_model_feature_idx += 1
                if self._simple_neural_model_feature_idx == len(self._simple_neural_model_feature):
                    self._simple_neural_model_feature_idx = 0
                    # return True
                feature_name = self._simple_neural_model_feature[self._simple_neural_model_feature_idx]['name']
                max_duration = self._simple_neural_model_feature[self._simple_neural_model_feature_idx]['max_duration']
                kwargs = self._simple_neural_model_feature[self._simple_neural_model_feature_idx]['kwargs']
            split_length = min(max_duration * AUDIO_SAMPLE_RATE, self._data_manager.get_raw_data_max_length())
            self._data_manager.set_raw_data_split_length(split_length)
            feature_pre_func = get_specified_feature_func(feature_name, **kwargs)
            self._data_manager.set_feature_pre_func(feature_pre_func)
            self._feature_name = feature_name
            log("round {}; feature name: {}; max_duration {}; split length {}; kwargs {}"
                .format(self._round_num, feature_name, max_duration, split_length, kwargs))

        if self._model_kind != 2 and (
                self._model_name != self._last_model_name or self._model is None or self._is_reset_model):
            if self._model_name in self._models:
                self._model = self._models[self._model_name]
            else:
                self._model = self._model_lib[self._model_name]()
                self._models[self._model_name] = self._model

    def _get_each_model_top_k_predicts(self):
        predicts = []
        for k, v in self._each_model_best_auc.items():
            if k == self._simple_model:
                continue
            k_predicts = np.asarray(self._each_model_best_predict[k])
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

    def _predict_ensemble(self):
        selected_k_best = [self._k_best_predicts[i]
                           for i, a in enumerate(self._k_best_auc) if a > 0.0]
        each_model_k_aucs, selected_each_model_k_best = self._get_each_model_top_k_predicts()

        if self._round_num >= 2:
            selected = selected_k_best + selected_each_model_k_best
        else:
            selected = selected_k_best

        log("model_num: {} Select k best {} predicts which have auc {}, "
            "each model {} best which have auc {}, "
            "and each previous model's best predict which have auc "
            "{}"
            .format(self._model_num, self._keep_num, self._k_best_auc,
                    self._each_model_keep_num, each_model_k_aucs,
                    ["({}:{})".format(k, v) for k, v in self._each_model_best_auc.items()]))
        return np.mean(selected, axis=0)

    def fit(self, train_loop_num=1, remaining_time_budget=None):
        flag = False
        # select model first, in order to use preprocess data method
        if self._model_kind != 2:
            self._pre_select_model_and_feature(train_loop_num)
        self._cur_model_run_loop += 1

        # simple model and first neural model need sampling data
        # if self._model_kind == 0 or self._model_num == 1:
        if self._model_kind == 0 or \
                (self._model_kind == 1 and self._round_num == 0 and self._model_num == 1):
            train_x, train_y, val_x, val_y = self._data_manager.get_train_data(train_loop_num=train_loop_num,
                                                                               use_new_data=self._use_new_data,
                                                                               is_need_incremental_sample=True,
                                                                               model_kind=self._model_kind)
        elif self._model_kind == 1:
            train_x, train_y, val_x, val_y = self._data_manager.get_train_data(train_loop_num=train_loop_num,
                                                                               use_new_data=self._use_new_data,
                                                                               is_need_incremental_sample=False,
                                                                               model_kind=self._model_kind)
        else:
            train_x, train_y, val_x, val_y = self._data_manager.get_raw_data()
        self._val_set = (val_x, val_y)
        self._input_shape = train_x.shape
        log("fit {} for {} times; train_x: {}; train_y: {}; val_x: {}; val_y: {};".format(
            self._model_name,
            self._cur_model_run_loop,
            train_x.shape,
            train_y.shape,
            val_x.shape,
            val_y.shape))
        # init model really
        self._get_or_create_model()
        if self._model_kind != 2:
            self._model.fit(train_x, train_y,
                            validation_data_fit=(val_x, val_y),
                            epochs=self._neural_model_train_epochs,
                            run_loop=self._cur_model_run_loop,
                            spec_len=None)
        else:
            flag = self._model.train((train_x, train_y), remaining_time_budget=remaining_time_budget)

        return flag

    def predict(self, test_x):
        if self._use_complex_neural_model_predict:
            return self._model.test(test_x)
        x_val, y_val = self._val_set
        if self._model_kind != 2:
            auc = auc_metric(y_val, self._model.predict(x_val))
            if self._model_kind == 0:
                self._lr_auc = auc
        else:
            print(x_val.shape, x_val.dtype)
            auc = auc_metric(y_val, self._model.test(x_val, True))
            simple_model_max_auc = self._each_model_best_auc[LR_MODEL][-1]
            # simple_model_mean_auc = np.sum([i for i in self._k_best_auc if i > 0]) / np.sum([1 for i in self._k_best_auc if i > 0])
            simple_model_last_auc = self._lr_auc
            simple_model_auc = simple_model_last_auc
            print("simple model auc: {};"
                  " simple model max auc: {};"
                  " simple model last auc: {};"
                  " complex neural model auc: {}".format(simple_model_auc,
                                                         simple_model_max_auc,
                                                         simple_model_last_auc,
                                                         auc))
            if auc >= simple_model_auc:
                self._use_complex_neural_model_predict = True
                log("use_complex_neural_model_predict")
                return self._model.test(test_x)
        need_predict = False
        if auc > self._cur_model_max_auc:
            log("cur_max_auc {}; cur_auc {}; {} auc rise for {} times"
                .format(self._cur_model_max_auc, auc, self._model_name, self._cur_model_run_loop))
            self._cur_model_max_auc = auc
            if self._round_num == 0:
                self._not_rise_num = max(0, self._not_rise_num - 1)
            else:
                self._not_rise_num = 0
            if auc > self._lr_auc - self._ts:
                need_predict = True
        else:
            self._not_rise_num += 1
            log("cur_max_auc {}; cur_auc {}; {} auc not rise for {} times"
                .format(self._cur_model_max_auc, auc, self._model_name, self._not_rise_num))

        if self._model_kind != 0 and \
                max(self._k_best_auc[-1], self._lr_auc - self._ts) >= auc and not need_predict:
            log("not predict")
        else:
            log("new predict")
            if self._model_kind != 2:
                if self._test_x is None:
                    self._test_x = get_features_data(x=test_x,
                                                     feature_func=self._data_manager.get_feature_pre_func(),
                                                     model_kind=self._model_kind,
                                                     split_length=self._data_manager.get_raw_data_split_length(),
                                                     feature_length=self._data_manager.get_feature_length())
                y_pred = self._model.predict(self._test_x, batch_size=8)
                if auc <= 0 and self._model_kind == 0:
                    return y_pred
            else:
                y_pred = self._model.test(test_x)
            if self._k_best_auc[-1] < auc and auc > self._lr_auc - self._ts:
                self._k_best_predicts[-1] = y_pred
                self._k_best_auc[-1] = auc
            if self._each_model_best_auc[self._model_name][-1] < auc:
                self._each_model_best_predict[self._model_name][-1] = y_pred
                self._each_model_best_auc[self._model_name][-1] = auc

            i = 0
            for a, p in sorted(
                    zip(self._k_best_auc, self._k_best_predicts), key=lambda x: x[0], reverse=True):
                self._k_best_auc[i] = a
                self._k_best_predicts[i] = p
                i += 1

            if self._model_kind == 0:
                self._k_best_predicts = [-1] * self._keep_num
                self._k_best_auc = [-1.1] * self._keep_num
                self._k_best_predicts[0] = y_pred
                self._k_best_auc[0] = self._lr_auc
                self._lr_preds = y_pred
                log("aucauc {}".format(self._lr_auc))
                return y_pred

        self._use_new_data = False
        self._is_reset_model = False

        return self._predict_ensemble()
