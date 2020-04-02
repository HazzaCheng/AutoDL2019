# -*- coding: utf-8 -*-
from __future__ import absolute_import
import random
import abc
import tensorflow as tf
import torchvision as tv
import torch
import numpy as np
import copy

from .api import Model
from .others import *
from ...import skeleton

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from ...architectures import *
from ..utils.tools import *
from ...constant import *
from ...TF_COMMAND import *

LOGGER = get_logger(__name__)


class LogicModel(Model):
    def __init__(self, metadata, session=None):
        super(LogicModel, self).__init__(metadata)
        LOGGER.info('--------- Model.metadata ----------')
        LOGGER.info('path: %s', self.metadata.get_dataset_name())
        LOGGER.info('shape:  %s', self.metadata.get_tensor_size(0))
        LOGGER.info('size: %s', self.metadata.size())
        LOGGER.info('num_class:  %s', self.metadata.get_output_size())

        test_metadata_filename = self.metadata.get_dataset_name().replace('train', 'test') + '/metadata.textproto'
        self.num_test = [int(line.split(':')[1]) for line in open(test_metadata_filename, 'r').readlines()[:3] if 'sample_count' in line][0]
        LOGGER.info('num_test:  %d', self.num_test)

        self.timers = {
            'train': skeleton.utils.Timer(),
            'test': skeleton.utils.Timer()
        }
        self.info = {
            'dataset': {
                'path': self.metadata.get_dataset_name(),
                'shape': self.metadata.get_tensor_size(0),
                'size': self.metadata.size(),
                'num_class': self.metadata.get_output_size()
            },
            'loop': {
                'epoch': 0,
                'test': 0,
                'best_score': {
                    'valid' : {},
                    'train' : {}
                }
            },
            'condition': {
                'first': {
                    'train': True,
                    'valid': True,
                    'test': True
                }
            },
            'terminate': False
        }

        # TODO: adaptive logic for hyper parameter
        self.hyper_params = {
            'optimizer': {
                'lr': 0.025,
            },
            'dataset': {
                'train_info_sample': 256,
                'cv_valid_ratio': 0.1,
                'max_valid_count': 256,

                'max_size': 64,
                'base': 16,  # input size should be multipliers of 16
                'max_times': 4,

                'enough_count': {
                    'image': 10000,
                    'video': 1000
                },

                'batch_size': 32,
                'steps_per_epoch': 30,
                'max_epoch': 400,  # initial value
                'batch_size_test': 256,
            },
            'checkpoints': {
                'keep': 20
            },
            'conditions': {
                'score_type': 'auc',
                'early_epoch': 1,
                'skip_valid_score_threshold': 0.90,  # if bigger then 1.0 is not use
                'skip_valid_after_test': min(10, max(3, int(self.info['dataset']['size'] // 1000))),
                'test_after_at_least_seconds': 1,
                'test_after_at_least_seconds_max': 90,
                'test_after_at_least_seconds_step': 2,

                'threshold_valid_score_diff': 0.001,
                'threshold_valid_best_score': 0.993,
                'max_inner_loop_ratio': 0.2,
                'min_lr': 1e-6,
                'use_fast_auto_aug': True
            }
        }

        self._model_name = " "
        self._last_model_name = " "

        self._mean_training_time = 0.0
        self._mean_all_data_training_time = 0.0

        self.checkpoints = {}
        # self._kbest_pre_result = {}
        self._each_model_kbest = {}
        self._global_kbest = []

        self.dataloaders = {
            'train': None,
            'valid': None,
            'test': None
        }
        # self._model_manager = ModelManager()

        self.is_skip_valid = True

        self._K_num = K_NUM_FIRST
        self._each_model_keep_num = EACH_MODEL_KEEP_NUM
        self._each_model_top_k = EACH_MODEL_TOP_K

        self.last_train_max_epoch = LAST_TRAIN_MAX_EPOCH
        self.pre_train_num_threshold = PRE_TRAIN_NUM_THRESHOLD
        self.pre_train_score_threshold = PRE_TRAIN_SCORE_THRESHOLD
        self.decrease_ratio = DECREASE_RATIO
        self.train_score_threshold_list = TRAIN_SCORE_THRESHOLD_LIST

        self._loaded_data_num = 0
        self.valid_score_lastk = []
        self._use_all_data_times = 0
        self._pre_predict_ensemble = None

        self._timer_num_process_data = 0
        self._timer_num_train = 0
        self._process_data_time = 0.0
        self._each_train_time = 0.0
        self._estimate_per_training_time = 0.0

        # self.model_sequence = [RESNET18_MODEL, RESNET34_MODEL, MOBILENET_MODEL]
        # self.model_sequence = [RESNET18_MODEL, SERESNEXT50_MODEL if self.num_test <= 10000 else DENSENET121_MODEL]
        self.model_sequence = [RESNET18_MODEL, SERESNEXT50_MODEL]

        self._inner_epoch_num = 1
        self._pre_train_loss = 9999.9
        self._cur_train_loss = 9999.9
        self._pre_train_score = 0
        self._cur_train_score = 0
        self._pre_train_state = None
        self._cur_epoch_coef = 1
        self._larger_data = False
        self._is_use_all_data = False
        self._use_pre_result = False

        self._is_use_ckp_ensemble = True if self.num_test < 10000 else False
        self._is_use_second_model = True

        self._pre_steps_per_epoch = 30
        self.basic_steps_num = 30 if self.info['dataset']['size'] < 30000 else 50
        self._model_id = 0
        self._model_round = 0
        self._model_seq_round = 0
        self._each_model_test_num = [0] * len(self.model_sequence)
        self._cur_model_train_times = 0
        self._each_model_use_time = [0] * len(self.model_sequence)
        self.complexmodel_overfit_time = 0
        self._global_step = 0
        self._last_k_queue = {
            'train' : [],
            'valid' : []
        }

        self._model_lib = CV_MODEL_LIB
        self._models = {}
        self._models_pred = {}

        self.model = None
        self.model_pred = None

        self._first_train_loss = 999.9
        self._second_train_loss = 999.9

        LOGGER.info('[init] build')

        self.build()

        LOGGER.info('[init] session')

        LOGGER.info('[init] done')

    def __repr__(self):
        return '\n---------[{0}]---------\ninfo:{1}\nparams:{2}\n---------- ---------'.format(
            self.__class__.__name__,
            self.info, self.hyper_params
        )

    def build(self):
        raise NotImplementedError

    def update_model(self):
        # call after to scan train sample
        pass

    def epoch_train(self, epoch, train):
        raise NotImplementedError

    def epoch_valid(self, epoch, valid):
        raise NotImplementedError

    def skip_valid(self, epoch):
        raise NotImplementedError

    def prediction(self, dataloader):
        raise NotImplementedError

    def adapt(self):
        raise NotImplementedError

    def select_model(self):
        raise NotImplementedError

    def select_model_video(self):
        raise NotImplementedError

    def is_change_model(self):
        raise NotImplementedError

    def get_model_name(self):
        raise NotImplementedError

    def is_multiclass(self):
        return self.info['dataset']['sample']['is_multiclass']

    def is_video(self):
        return self.info['dataset']['sample']['is_video']

    @timeit
    def build_or_get_train_dataloader(self, dataset):
        if not self.info['condition']['first']['train']:
            return self.build_or_get_dataloader('train')

        num_images = self.info['dataset']['size']

        # split train/valid
        num_valids = int(min(num_images * self.hyper_params['dataset']['cv_valid_ratio'], self.hyper_params['dataset']['max_valid_count']))
        num_trains = num_images - num_valids
        LOGGER.info('[cv_fold] num_trains:%d num_valids:%d', num_trains, num_valids)

        LOGGER.info('[%s] scan before', 'sample')
        num_samples = self.hyper_params['dataset']['train_info_sample']
        sample = dataset.take(num_samples).prefetch(buffer_size=num_samples)
        train = skeleton.data.TFDataset(self.session, sample, num_samples)
        self.info['dataset']['sample'] = train.scan(samples=num_samples)

        del train
        del sample
        LOGGER.info('[%s] scan after', 'sample')

        is_multi_class = "Multi_Class" if self.is_multiclass() else "One_Class"
        log("is_multi_class: {}".format(is_multi_class))

        # input_shape = [min(s, self.hyper_params['dataset']['max_size']) for s in self.info['dataset']['shape']]
        times, height, width, channels = self.info['dataset']['sample']['example']['shape']
        values = self.info['dataset']['sample']['example']['value']
        aspect_ratio = width / height

        # fit image area to 64x64
        if aspect_ratio > 2 or 1. / aspect_ratio > 2:
            self.hyper_params['dataset']['max_size'] *= 2
        size = [min(s, self.hyper_params['dataset']['max_size']) for s in [height, width]]

        # keep aspect ratio
        if aspect_ratio > 1:
            size[0] = size[1] / aspect_ratio
        else:
            size[1] = size[0] * aspect_ratio

        # too small image use original image
        if width <= 32 and height <= 32:
            input_shape = [times, height, width, channels]
        else:
            fit_size_fn = lambda x: int(x / self.hyper_params['dataset']['base'] + 0.8) * self.hyper_params['dataset']['base']
            size = list(map(fit_size_fn, size))
            min_times = min(times, self.hyper_params['dataset']['max_times'])
            # min_times = int(times / 2)
            input_shape = [fit_size_fn(min_times) if min_times > self.hyper_params['dataset']['base'] else min_times] + size + [channels]
            # input_shape = [1] + size + [channels]

        if self.is_video():
            self.hyper_params['dataset']['batch_size'] = int(self.hyper_params['dataset']['batch_size'] // 2)
            # self.hyper_params['dataset']['batch_size_test'] = int(self.hyper_params['dataset']['batch_size_test'] // 2)
        LOGGER.info('[input_shape] origin:%s aspect_ratio:%f target:%s', [times, height, width, channels], aspect_ratio, input_shape)

        self.hyper_params['dataset']['input'] = input_shape

        num_class = self.info['dataset']['num_class']
        batch_size = self.hyper_params['dataset']['batch_size']
        if num_class > batch_size / 2 and not self.is_video():
            self.hyper_params['dataset']['batch_size'] = batch_size * 2
        batch_size = self.hyper_params['dataset']['batch_size']

        preprocessor1 = get_tf_resize(input_shape[1], input_shape[2], times=input_shape[0], min_value=values['min'], max_value=values['max'])

        dataset = dataset.map(
            lambda *x: (preprocessor1(x[0]), x[1]),
            num_parallel_calls=TF_COMMAND
        )
        # dataset = dataset.prefetch(buffer_size=batch_size * 3)
        # dataset = dataset.shuffle(buffer_size=num_valids * 4, reshuffle_each_iteration=False)

        must_shuffle = self.info['dataset']['sample']['label']['zero_count'] / self.info['dataset']['num_class'] >= 0.5
        enough_count = self.hyper_params['dataset']['enough_count']['video'] if self.is_video() else self.hyper_params['dataset']['enough_count']['image']
        if must_shuffle or num_images < enough_count:
            dataset = dataset.shuffle(buffer_size=min(enough_count, num_images), reshuffle_each_iteration=False)
            LOGGER.info('[dataset] shuffle before split train/valid')

        train = dataset.skip(num_valids)
        valid = dataset.take(num_valids)

        self.datasets = {
            'train': train,
            'valid': valid,
            'num_trains': num_trains,
            'num_valids': num_valids
        }

        return self.build_or_get_dataloader('train', self.datasets['train'], num_trains)

    def cur_loop_data_num(self, mode):
        data_num = self.basic_steps_num
        # if self.is_video() and mode == "train":
        if mode == "train":
            # if self.info['loop']['test'] == 0:
            if self.info['loop']['test'] < 5:
                if self.is_video():
                    self.basic_steps_num = 30
                else:
                    self.basic_steps_num = 50
                data_num = self.basic_steps_num
            else:
                # if self.get_model_name() == RESNET18_MODEL:
                if self._model_round == 0:
                    self.basic_steps_num = 30
                    data_num = self.basic_steps_num
                else:
                    self.basic_steps_num = 30
                    data_num = self.basic_steps_num
                    self._use_all_data_times += 1
                    self._is_use_all_data = True
                    # if self.is_change_model() and self._each_model_use_time[self._model_id] == 1:
                    #     # data_num = int(self.datasets['num_trains'] * 0.3 // self.hyper_params['dataset']['batch_size'])
                    #     # data_num = max(data_num, self.basic_steps_num)
                    #     # log("change_model data_num: {}  num_trains: {}  batch_size: {}".format(data_num,
                    #     #                                                                  self.datasets['num_trains'],
                    #     #                                                                  self.hyper_params['dataset'][
                    #     #                                                                      'batch_size']))
                    #     data_num = self.basic_steps_num
                    #     self._use_all_data_times += 1
                    #     self._is_use_all_data = True
                    # else:
                    #     data_num = min(self.basic_steps_num,
                    #                    int(self.datasets['num_trains'] // self.hyper_params['dataset']['batch_size']))
                # data_num = 30 + (self._cur_epoch_coef - 1) * 10

        return data_num

    @timeit
    def build_or_get_dataloader(self, mode, dataset=None, num_items=0):
        if mode == 'train':
            self._pre_steps_per_epoch = copy.deepcopy(self.hyper_params['dataset']['steps_per_epoch'])
            self.hyper_params['dataset']['steps_per_epoch'] = self.cur_loop_data_num(mode)

            log("pre: {},  now_steps: {}".format(self._pre_steps_per_epoch, self.hyper_params['dataset']['steps_per_epoch']))

            if mode in self.dataloaders and self.dataloaders[mode] is not None:
                if self._pre_steps_per_epoch != self.hyper_params['dataset']['steps_per_epoch']:
                    self.dataloaders[mode].steps = self.hyper_params['dataset']['steps_per_epoch']
                return self.dataloaders[mode]

        else:
            if mode in self.dataloaders and self.dataloaders[mode] is not None:
                return self.dataloaders[mode]

        enough_count = self.hyper_params['dataset']['enough_count']['video'] if self.is_video() else self.hyper_params['dataset']['enough_count']['image']

        LOGGER.debug('[dataloader] %s build start', mode)
        values = self.info['dataset']['sample']['example']['value']

        if mode == 'train':
            batch_size = self.hyper_params['dataset']['batch_size']
            # input_shape = self.hyper_params['dataset']['input']
            preprocessor = get_tf_to_tensor(is_random_flip=True)
            # dataset = dataset.prefetch(buffer_size=batch_size * 3)

            if num_items < enough_count:
                dataset = dataset.cache()

            # dataset = dataset.apply(
            #     tf.data.experimental.shuffle_and_repeat(buffer_size=min(enough_count, num_items))
            # )

            dataset = dataset.repeat()                  # repeat indefinitely
            # dataset = dataset.shuffle(self.info['dataset']['size'], reshuffle_each_iteration=True).repeat()
            dataset = dataset.map(
                lambda *x: (preprocessor(x[0]), x[1]),
                num_parallel_calls=TF_COMMAND
            )
            dataset = dataset.prefetch(buffer_size=batch_size * 8)

            dataset = skeleton.data.TFDataset(self.session, dataset, num_items)

            transform = tv.transforms.Compose([
                # skeleton.data.Cutout(int(input_shape[1] // 4), int(input_shape[2] // 4))
            ])
            dataset = skeleton.data.TransformDataset(dataset, transform, index=0)

            self.dataloaders['train'] = skeleton.data.FixedSizeDataLoader(
                dataset,
                steps=self.hyper_params['dataset']['steps_per_epoch'],
                batch_size=batch_size,
                shuffle=False, drop_last=True, num_workers=0, pin_memory=False
            )
        elif mode in ['valid', 'test']:
            batch_size = self.hyper_params['dataset']['batch_size_test']
            input_shape = self.hyper_params['dataset']['input']

            preprocessor2 = get_tf_to_tensor(is_random_flip=False)
            if mode == 'valid':
                preprocessor = preprocessor2
            else:
                preprocessor1 = get_tf_resize(input_shape[1], input_shape[2], times=input_shape[0], min_value=values['min'], max_value=values['max'])
                preprocessor = lambda *tensor: preprocessor2(preprocessor1(*tensor))

            # batch_size = 500
            tf_dataset = dataset.apply(
                tf.data.experimental.map_and_batch(
                    map_func=lambda *x: (preprocessor(x[0]), x[1]),
                    batch_size=batch_size,
                    drop_remainder=False,
                    num_parallel_calls=TF_COMMAND
                )
            ).prefetch(buffer_size=8)

            dataset = skeleton.data.TFDataset(self.session, tf_dataset, num_items)

            LOGGER.info('[%s] scan before', mode)
            self.info['dataset'][mode], tensors = dataset.scan(
                with_tensors=True, is_batch=True,
                device=self.device, half=self.is_half
            )
            tensors = [torch.cat(t, dim=0) for t in zip(*tensors)]
            LOGGER.info('[%s] scan after', mode)

            del tf_dataset
            del dataset
            dataset = skeleton.data.prefetch_dataset(tensors)

            if 'valid' == mode:
                transform = tv.transforms.Compose([
                ])
                dataset = skeleton.data.TransformDataset(dataset, transform, index=0)

            self.dataloaders[mode] = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.hyper_params['dataset']['batch_size_test'],
                shuffle=False, drop_last=False, num_workers=0, pin_memory=False
            )

            self.info['condition']['first'][mode] = False

        LOGGER.debug('[dataloader] %s build end', mode)
        return self.dataloaders[mode]

    def update_condition(self, metrics=None):
        self.info['condition']['first']['train'] = False
        self.info['loop']['epoch'] += 1

        metrics.update({'epoch': self.info['loop']['epoch']})
        self.checkpoints[self._model_name].append(metrics)

        indices = np.argsort(np.array([v['valid']['score'] for v in self.checkpoints[self._model_name]] if len(self.checkpoints[self._model_name]) > 0 else [0]))
        indices = sorted(indices[::-1][:self.hyper_params['checkpoints']['keep']])
        self.checkpoints[self._model_name] = [self.checkpoints[self._model_name][i] for i in indices]

        # if metrics['valid']['score'] < 0.01:
        #     return
        #
        # if len(self.valid_score_lastk) < 10:
        #     self.valid_score_lastk.append(round(metrics['valid']['score'], 3))
        #     return
        #
        # temp_valid_score = []
        # for i in range(len(self.valid_score_lastk)):
        #     if i == 0:
        #         continue
        #     temp_valid_score.append(self.valid_score_lastk[i])
        # temp_valid_score.append(round(metrics['valid']['score'], 3))
        # self.valid_score_lastk = temp_valid_score

    def break_train_loop_condition(self, remaining_time_budget=None, total_epoch=1):
        consume = total_epoch * self.timers['train'].step_time

        best_idx = 'best_idx'
        best_epoch = 'best_epoch'
        best_loss = 'best_loss'
        best_score = 'best_score'

        valid_metric = {}
        train_metric = {}

        valid_metric[best_idx] = np.argmax(np.array([c['valid']['score'] for c in self.checkpoints[self._model_name]]))
        valid_metric[best_epoch] = self.checkpoints[self._model_name][valid_metric[best_idx]]['epoch']
        valid_metric[best_loss] = self.checkpoints[self._model_name][valid_metric[best_idx]]['valid']['loss']
        valid_metric[best_score] = self.checkpoints[self._model_name][valid_metric[best_idx]]['valid']['score']

        train_metric[best_idx] = np.argmax(np.array([c['train']['score'] for c in self.checkpoints[self._model_name]]))
        train_metric[best_epoch] = self.checkpoints[self._model_name][train_metric[best_idx]]['epoch']
        train_metric[best_loss] = self.checkpoints[self._model_name][train_metric[best_idx]]['train']['loss']
        train_metric[best_score] = self.checkpoints[self._model_name][train_metric[best_idx]]['train']['score']

        lr = self.optimizer[self._model_name].get_learning_rate()
        LOGGER.debug('[CONDITION] best (epoch:%04d loss:%.2f score:%.2f) lr:%.8f time delta:%.2f',
                     valid_metric[best_epoch], valid_metric[best_loss], valid_metric[best_score], lr, consume)

        if self.info['loop']['epoch'] <= self.hyper_params['conditions']['early_epoch']:
            LOGGER.info('[BREAK] early %d epoch', self.hyper_params['conditions']['early_epoch'])
            return True

        if valid_metric[best_score] > self.hyper_params['conditions']['threshold_valid_best_score']:
            LOGGER.info('[BREAK] achieve best score %f', valid_metric[best_score])
            return True

        if self.num_test > 10000:
            if total_epoch < 2:
                return False

        if self._model_round > 0 and total_epoch < 2:
            return False

        if self._model_round > 0 and self._global_step > 20:
            LOGGER.info('[BREAK] too many epochs for complex models')
            return True

        if self.is_video():
            if self._global_step > 20:
                LOGGER.info('[BREAK] too many epochs for complex models')
                return True

        best_score_ratio = 1.001
        # best_score_ratio = 1.105
        # if self.info['loop']['test'] <= 10 and best_score <= 0.55:
        #     # best_score_ratio = 1.255
        #     best_score_ratio = 1.105
        # else:
        #     best_score_ratio -= 0.035
        #     if best_score_ratio < 1.001:
        #         best_score_ratio = 1.001

        # if self.is_video() and self.info['loop']['test'] <= 5 and best_score <= 0.55:
        #     if inner_epoch <= 3:
        #         return False

        if self.info['loop']['test'] < 4:
            # LOGGER.info('pre_train_loss : %f    cur_train_loss : %f ', self._pre_train_loss,
            #                                                            self._cur_train_loss)
            loop_test = self.info['loop']['test']
            if total_epoch >= 5:
                LOGGER.info('[BREAK] pre3 round cannot train more than 5 epoches')
                return True
            log("cur_train_score: {}".format(self._cur_train_score))
            log("cur_train_loss: {}".format(self._cur_train_loss))

            if loop_test == 1 and \
               self._cur_train_score < self.train_score_threshold_list[loop_test]:
                return False
            # if loop_test > 1 and \
            #    self._cur_train_loss > self._pre_train_loss * self.decrease_ratio[loop_test]:
            #     return False

        # if self.is_change_model():
        if self.is_skip_valid is False:
            if valid_metric[best_score] <= self._global_kbest[-1]['valid']['score']:
                return False
        else:
            if train_metric[best_score] <= self._global_kbest[-1]['train']['score']:
                return False

        if self._model_round > 0:
            if self.is_skip_valid is True:
                return False

        if consume > self.hyper_params['conditions']['test_after_at_least_seconds'] and \
            ((self.is_skip_valid is False and valid_metric[best_score] > self.info['loop']['best_score']['valid'][self._model_name] * best_score_ratio) or \
            (self.is_skip_valid is True and train_metric[best_score] > self.info['loop']['best_score']['train'][self._model_name] * best_score_ratio)):

            if self.is_skip_valid is False and valid_metric[best_score] > self.info['loop']['best_score']['valid'][self._model_name] * best_score_ratio:
                metric = valid_metric
            else:
                metric = train_metric

            if self.checkpoints[self._model_name][metric[best_idx]]['epoch'] > self.info['loop']['epoch'] - total_epoch:
                # best_score > self.info['loop']['best_score'] * 1.011:
                # increase hyper param
                self.hyper_params['conditions']['test_after_at_least_seconds'] = min(
                    self.hyper_params['conditions']['test_after_at_least_seconds_max'],
                    self.hyper_params['conditions']['test_after_at_least_seconds'] + self.hyper_params['conditions']['test_after_at_least_seconds_step']
                )
                if self.is_skip_valid is False and valid_metric[best_score] > self.info['loop']['best_score']['valid'][self._model_name] * best_score_ratio:
                    self.info['loop']['best_score']['valid'][self._model_name] = valid_metric[best_score]
                else:
                    self.info['loop']['best_score']['train'][self._model_name] = train_metric[best_score]
                LOGGER.info('[BREAK] found best model (score:%f)', metric[best_score])
                return True

        if lr < self.hyper_params['conditions']['min_lr']:
            LOGGER.info('[BREAK] too small lr (lr:%f < %f)', lr, self.hyper_params['conditions']['min_lr'])
            return True

        early_term_budget = 3 * 60
        expected_more_time = (self.timers['test'].step_time + (self.timers['train'].step_time * 2)) * 1.5
        if remaining_time_budget is not None and \
                remaining_time_budget - early_term_budget < expected_more_time:
            LOGGER.info('[BREAK] not enough time to train (remain:%f need:%f)', remaining_time_budget,
                        expected_more_time)
            return True

        if self.info['loop']['epoch'] >= 20 and \
            total_epoch > self.hyper_params['dataset']['max_epoch'] * self.hyper_params['conditions']['max_inner_loop_ratio']:
            LOGGER.info('[BREAK] cannot found best model in too many epoch')
            return True

        if valid_metric[best_score] < 0.01:
            if total_epoch >= 8:
                LOGGER.info('[BREAK] train over 8 epoch in no valid test')
                return True

        # if self._model_round > 0 and self._each_model_use_time[self._model_id] == 1 and \
        #         self._cur_model_train_times == 1:
        #     LOGGER.info('[BREAK] if not resnet18 and new model first train then break in fixed inner epoch')
        #     return True

        if len(self.valid_score_lastk) >= 10:
            log("valid_score_lastk : {}".format(self.valid_score_lastk))
            flag = False
            for i in range(len(self.valid_score_lastk) - 1):
                if self.valid_score_lastk[i] < self.valid_score_lastk[i+1]:
                    flag = True
                    break
            if flag is False:
                LOGGER.info('[BREAK] valid score has not increased for at least 5 epoches')
                return True

        return False

    def terminate_train_loop_condition(self, remaining_time_budget=None, inner_epoch=0):
        early_term_budget = 3 * 60

        meaning_time = 0.0
        if self.is_change_model() is False or (self._model_round == 0):
            self._mean_training_time = (self._mean_training_time * self.info['loop']['test'] + self.timers['train'].step_time) / (self.info['loop']['test'] + 1)

        if self.is_change_model() and (self._model_round > 0) and (self._each_model_use_time[self._model_id] == 1):
            self._mean_all_data_training_time = (self._mean_all_data_training_time * (self._use_all_data_times - 1) +
                                                 self.timers['train'].step_time) / self._use_all_data_times
            log("use_all_data_time___ step_time: {}  mean_time: {}".format(self.timers['train'].step_time,
                                                                       self._mean_all_data_training_time))
        meaning_time = self._mean_training_time

        if self._model_round == 0 or self.is_change_model() is False:
            # expected_more_time = (self.timers['test'].step_time + (self.timers['train'].step_time * 2)) * 1.5
            expected_more_time = (self.timers['test'].step_time + (meaning_time * 2)) * 1.5
            if remaining_time_budget is not None and \
                remaining_time_budget - early_term_budget < expected_more_time:
                LOGGER.info('[TERMINATE] not enough time to train (remain:%f need:%f)', remaining_time_budget, expected_more_time)
                self.info['terminate'] = True
                self.done_training = True
                return True

        if self.info['loop']['test'] < 5:
            return False

        best_idx = np.argmax(np.array([c['valid']['score'] for c in self.checkpoints[self._model_name]]))
        best_score = self.checkpoints[self._model_name][best_idx]['valid']['score']
        if best_score > self.hyper_params['conditions']['threshold_valid_best_score']:
            if self._model_seq_round == 1:
                if self._model_round == 0 and remaining_time_budget > 840:
                    LOGGER.info('[TERMINATE] achieve best score but remaining time enough to train new model %f', best_score)
                    self.info['terminate'] = False
                    return False
                else:
                    if self._model_round == 1 and remaining_time_budget > 720:
                        LOGGER.info('[TERMINATE] achieve best score but remaining time enough to train new model %f', best_score)
                        self.info['terminate'] = False
                        return False

            LOGGER.info('[TERMINATE] achieve best score %f', best_score)
            done = True if self.info['terminate'] else False
            self.info['terminate'] = True
            self.done_training = done if self.is_video() else True
            return True

        scores = [c['valid']['score'] for c in self.checkpoints[self._model_name]]
        diff = (max(scores) - min(scores)) * (1 - max(scores))
        threshold = self.hyper_params['conditions']['threshold_valid_score_diff']
        if 1e-8 < diff and diff < threshold and self.info['loop']['epoch'] >= 20:
            if self._model_seq_round == 1:
                if self._model_round == 0 and remaining_time_budget > 480:
                    LOGGER.info('[TERMINATE] too small score change (diff:%f < %f) but remaining time enough to train new model', diff, threshold)
                    self.info['terminate'] = False
                    return False
                else:
                    if self._model_round == 1 and remaining_time_budget > 420:
                        LOGGER.info('[TERMINATE] too small score change (diff:%f < %f) but remaining time enough to train new model', diff, threshold)
                        self.info['terminate'] = False
                        return False
            LOGGER.info('[TERMINATE] too small score change (diff:%f < %f)', diff, threshold)
            done = True if self.info['terminate'] else False
            self.info['terminate'] = True
            self.done_training = done
            return True

        if self.optimizer[self._model_name].get_learning_rate() < self.hyper_params['conditions']['min_lr']:
            LOGGER.info('[TERMINATE] lr=%f', self.optimizer[self._model_name].get_learning_rate())
            done = True if self.info['terminate'] else False
            self.info['terminate'] = True
            self.done_training = done
            return True

        if self.info['loop']['epoch'] >= 20 and \
            inner_epoch > self.hyper_params['dataset']['max_epoch'] * self.hyper_params['conditions']['max_inner_loop_ratio']:
            LOGGER.info('[TERMINATE] cannot found best model in too many epoch')
            done = True if self.info['terminate'] else False
            self.info['terminate'] = True
            self.done_training = done
            return True

        return False

    def get_total_time(self):
        return sum([self.timers[key].total_time for key in self.timers.keys()])

    @timeit
    def train(self, dataset, remaining_time_budget=None):
        LOGGER.debug(self)
        LOGGER.debug('[train] [%02d] budget:%f', self.info['loop']['epoch'], remaining_time_budget)
        self.timers['train']('outer_start', exclude_total=True, reset_step=True)

        self._is_use_all_data = False

        if self.info['loop']['test'] >= 1:
            if self.is_video():
                self.select_model_video()
                if self.info['loop']['test'] == 0:
                    del self._models[self.model_sequence[1]]
                    del self._models_pred[self.model_sequence[1]]
                    del self.model_sequence[1]
            else:
                self.select_model()
                log("select_model_resnet!")

        train_dataloader = self.build_or_get_train_dataloader(dataset)

        if self.info['loop']['test'] == 0:
            if self.is_video():
                self.select_model_video()
                if self.info['loop']['test'] == 0:
                    del self._models[self.model_sequence[1]]
                    del self._models_pred[self.model_sequence[1]]
                    del self.model_sequence[1]
            else:
                # if self.info['loop']['test'] == 0:
                #     self.select_model()
                #     log("select_model_resnet!")
                self.select_model()
                log("select_model_resnet!")

        if self._is_use_all_data:
            if remaining_time_budget is not None:
                if self._model_seq_round == 1 and self._model_id == 1:
                    if remaining_time_budget < self._estimate_per_training_time * self.hyper_params['dataset']['steps_per_epoch'] * 2.5:
                        LOGGER.info('[TERMINATE] not enough time to train new model(remain:%f need:%f)',
                                    remaining_time_budget,
                                    self._estimate_per_training_time * self.hyper_params['dataset']['steps_per_epoch'] * 2.5)
                        self.info['terminate'] = True
                        self.done_training = True
                        self._use_pre_result = True
                        return
                else:
                    if remaining_time_budget < self._mean_all_data_training_time * 1.5:
                        LOGGER.info('[TERMINATE] not enough time to train new model(remain:%f need:%f)', remaining_time_budget,
                                    remaining_time_budget * 1.5)
                        self.info['terminate'] = True
                        self.done_training = True
                        self._use_pre_result = True
                        return

        # if self.info['condition']['first']['train']:
        #     self.update_model()
        #     LOGGER.info(self)

        if self.is_change_model():
             if self._each_model_test_num[self._model_id] == 0:
                 self.update_model()
                 self.checkpoints[self._model_name] = []
                 self.info['loop']['best_score']['train'][self._model_name] = 0.0
                 self.info['loop']['best_score']['valid'][self._model_name] = 0.0
                 self._each_model_kbest[self._model_name] = []
             self._last_k_queue['train'] = []
             self._last_k_queue['valid'] = []

             self._cur_model_train_times = 0

             if self._model_round > 0:
                 self.hyper_params['conditions']['skip_valid_score_threshold'] = 0.80
                 self.hyper_params['conditions']['test_after_at_least_seconds'] = 1
                 self.hyper_params['conditions']['skip_valid_after_test'] = min(3, max(3, int(
                     self.info['dataset']['size'] // 1000)))

             LOGGER.info(self)

        log("!!!!!! this round model !!!!!! : {}".format(self._model_name))

        self.timers['train']('build_dataset')

        log("loop_test: {}, steps_per_epoch: {}".format(self.info['loop']['test'],
                                                        self.hyper_params['dataset']['steps_per_epoch']))

        self.valid_score_lastk = []

        if self.info['loop']['test'] < 2:
            self._inner_epoch_num = 1
        else:
            # if self.get_model_name() != RESNET18_MODEL
            if self._model_round > 0:
                if self._each_model_test_num[self._model_id] == 0:
                     self._inner_epoch_num = 2
                else:
                    if self.is_skip_valid is False:
                        self._inner_epoch_num = 1
            else:
                if len(self._global_kbest) > 0 and \
                        (self._global_kbest[0]['valid']['score'] > 0.1 or
                         self._global_kbest[-1]['train']['score'] >= 0.9):
                    self._inner_epoch_num = 1

        self._cur_model_train_times += 1

        self._global_step = 0
        total_epoch = 0
        while True:
            total_epoch += 1
            remaining_time_budget -= self.timers['train'].step_time

            self.timers['train']('start', reset_step=True)
            train_metrics = self.epoch_train(self.info['loop']['epoch'], train_dataloader)
            self.timers['train']('train')

            last_k_ckp = 20
            if self._model_round > 0:
                last_k_ckp = 4

            train_score = np.min([c['train']['score'] for c in self.checkpoints[self._model_name][-last_k_ckp:] + [{'train': train_metrics}]])
            # if train_score is too low, the valid is not necessary, the training should be continued
            # if test_loop gets larger, the valid is a must
            log("_model_round:{} _is_change_model:{}".format(self._model_round, self.is_change_model()))

            if train_score > self.hyper_params['conditions']['skip_valid_score_threshold'] or \
                self._each_model_test_num[self._model_id] >= self.hyper_params['conditions']['skip_valid_after_test']\
                or ((self._model_round > 0) and (self.is_change_model() == False)):

                is_first = self.info['condition']['first']['valid']
                valid_dataloader = self.build_or_get_dataloader('valid', self.datasets['valid'], self.datasets['num_valids'])
                self.timers['train']('valid_dataset', exclude_step=is_first)

                valid_metrics = self.epoch_valid(self.info['loop']['epoch'], valid_dataloader)
                self.is_skip_valid = False
                if self._model_round == 0:
                    self._K_num = K_NUM_SECOND_VIDEO if self.is_video() else K_NUM_SECOND_IMAGE
                    # self._is_use_ckp_ensemble = False
            else:
                # skip valid: set loss inf and score 1e-4, making it useless
                valid_metrics = self.skip_valid(self.info['loop']['epoch'])
                self.is_skip_valid = True
            self.timers['train']('valid')

            metrics = {
                'epoch': self.info['loop']['epoch'],
                # 'model': self.model.state_dict(),
                'model': copy.deepcopy(self.model.state_dict()),
                'model_prediction': None,
                'train': train_metrics,
                'valid': valid_metrics,
            }

            self._cur_train_score = train_metrics['score']

            self.update_condition(metrics)

            self.timers['train']('adapt', exclude_step=True)

            self._loaded_data_num += self.dataloaders['train'].steps * self.hyper_params['dataset']['batch_size']

            LOGGER.info(
                '[train] [%02d] time(budge:%.2f, total:%.2f, step:%.2f) loss:(train:%.3f, valid:%.3f) score:(train:%.3f valid:%.3f) lr:%f',
                self.info['loop']['epoch'], remaining_time_budget, self.get_total_time(), self.timers['train'].step_time,
                metrics['train']['loss'], metrics['valid']['loss'], metrics['train']['score'], metrics['valid']['score'],
                self.optimizer[self._model_name].get_learning_rate()
            )
            LOGGER.debug('[train] [%02d] Timer:%s', self.info['loop']['epoch'], self.timers['train'])

            self.hyper_params['dataset']['max_epoch'] = self.info['loop']['epoch'] + remaining_time_budget // self.timers['train'].step_time
            LOGGER.info('[ESTIMATE] max_epoch: %d', self.hyper_params['dataset']['max_epoch'])

            self._cur_train_loss = metrics['train']['loss']

            if self.break_train_loop_condition(remaining_time_budget, total_epoch):
                break

            self.timers['train']('end')

        self._pre_train_loss = metrics['train']['loss']
        self._pre_train_score = metrics['train']['score']
        self._pre_train_state = copy.deepcopy(metrics)
        log("_pre_train_loss : {}".format(self._pre_train_loss))

        if self.info['loop']['test'] == 0:
            self._first_train_loss = metrics['train']['loss']

        remaining_time_budget -= self.timers['train'].step_time
        self.terminate_train_loop_condition(remaining_time_budget, total_epoch)

        if not self.done_training:
            self.adapt(remaining_time_budget)

        self.timers['train']('outer_end')
        LOGGER.info(
            '[train] [%02d] time(budge:%.2f, total:%.2f, step:%.2f) loss:(train:%.3f, valid:%.3f) score:(train:%.3f valid:%.3f) lr:%f',
            self.info['loop']['epoch'], remaining_time_budget, self.get_total_time(), self.timers['train'].step_time,
            metrics['train']['loss'], metrics['valid']['loss'], metrics['train']['score'], metrics['valid']['score'],
            self.optimizer[self._model_name].get_learning_rate()
        )
        LOGGER.info('[train] [%02d] Timer:%s', self.info['loop']['epoch'], self.timers['train'])

    @timeit
    def test(self, dataset, remaining_time_budget=None):
        self.timers['test']('start', exclude_total=True, reset_step=True)
        is_first = self.info['condition']['first']['test']
        self.info['loop']['test'] += 1
        self._each_model_test_num[self._model_id] += 1

        dataloader = self.build_or_get_dataloader('test', dataset, self.num_test)
        self.timers['test']('build_dataset', reset_step=is_first)

        if self._model_round == 1 and self.is_change_model():
            self._K_num = K_NUM_THIRD
            del self._global_kbest[-1]
            del self._global_kbest[-1]
            log("change_model_time, global_kbest_len: {}".format(len(self._global_kbest)))

        if self._use_pre_result or ((self.complexmodel_overfit_time >= 4) and (self.is_video() is False)) or \
                ((self.complexmodel_overfit_time >= 9) and (self.is_video() is True)):
            rv = self._pre_predict_ensemble
            self.done_training = True
        else:
            rv = self.prediction(dataloader)
        self.timers['test']('end')

        LOGGER.info(
            '[test ] [%02d] test:%02d time(budge:%.2f, total:%.2f, step:%.2f)',
            self.info['loop']['epoch'], self.info['loop']['test'], remaining_time_budget, self.get_total_time(), self.timers['test'].step_time,
        )
        LOGGER.debug('[test ] [%02d] Timer:%s', self.info['loop']['epoch'], self.timers['test'])

        if self.info['loop']['test'] == 2:
            if self.timers['test'].step_time >= 3:
                self._is_use_ckp_ensemble = False

        if self.is_video():
            self._is_use_ckp_ensemble = False

        if self.is_video() is False:
            if self.info['loop']['test'] == 4:
                if self.timers['test'].step_time >= 7:
                    del self._models[self.model_sequence[1]]
                    del self._models_pred[self.model_sequence[1]]
                    del self.model_sequence[1]
                    self._is_use_second_model = False
                    log("no use second model!")

        log("model_seq_len: {}".format(len(self.model_sequence)))

        return rv
