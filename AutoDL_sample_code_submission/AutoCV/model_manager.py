# -*- coding: utf-8 -*-
from __future__ import absolute_import

import gc
import numpy as np
import tensorflow as tf
import os
import gc
import torch

from .import skeleton
from .architectures import *
from .skeleton.projects import get_logger, LogicModel
from .skeleton.projects.others import NBAC, AUC
from .skeleton.utils.tools import *
from .model import Model
from .architectures import *

LOGGER = get_logger(__name__)


class ModelManager(Model):
    def __init__(self, metadata):
        super(ModelManager, self).__init__(metadata)

        # self._model = None
        # self._model_pred = None

        self.change_model_valid_threshold = 0.020

    def build_model(self, model_name, to_device=True):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        in_channels = self.info['dataset']['shape'][-1]
        num_class = self.info['dataset']['num_class']
        # torch.cuda.synchronize()

        LOGGER.info('[init] Model_{}'.format(model_name))
        # Network = ResNet18  # ResNet18  ResNet34 # BasicNet, SENet18, ResNet18, MobileNetv2, EfficientNetB0
        Network = self._model_lib[model_name]
        self._models[model_name] = Network(in_channels, num_class)
        self._models_pred[model_name] = Network(in_channels, num_class).eval()  # test: eval will hold BN and Dropout unchanged
        # torch.cuda.synchronize()

        LOGGER.info('[init] Model_{} weight initialize'.format(model_name))
        if model_name in [RESNET18_MODEL, RESNET34_MODEL, EFFICIENT_MODEL, MOBILENET_MODEL,
                          DENSENET121_MODEL, DENSENET201_MODEL,
                          SERESNEXT50_MODEL, SERESNEXT101_MODEL]:

            # model_path = os.path.join(base_dir, 'models')
            model_path = os.path.join(base_dir, 'models')
            LOGGER.info('model path: %s', model_path)
            self._models[model_name].init(model_dir=model_path, gain=1.0)
        else:
            self._models[model_name].init(gain=1.0)
        # torch.cuda.synchronize()

        # if model_name == RESNET18_MODEL:
        #     model = ResNet18(3, 5)
        #     model.load_state_dict(torch.load(os.path.join(base_dir, "meta_models/meta-lr-ResNet18-8.pkl")))
        #     print("use meta model")
        #     self._models_pred[model_name].copy_params(model.meta_parameters())

        if to_device:
            LOGGER.info('[init] Model_{} copy to device'.format(model_name))
            self._models[model_name] = self._models[model_name].to(device=self.device, non_blocking=True)  # .half()
            self._models_pred[model_name] = self._models_pred[model_name].to(device=self.device, non_blocking=True)  # .half()
            self.is_half = self._models[model_name]._half
            self.model = self._models[model_name]
            self.model_pred = self._models_pred[model_name]
            # torch.cuda.synchronize()

        LOGGER.info('[init] Model_{} done.'.format(model_name))

    @timeit
    def init_opt(self):
        steps_per_epoch = self.hyper_params['dataset']['steps_per_epoch']
        batch_size = self.hyper_params['dataset']['batch_size']

        params = [p for p in self.model.parameters() if p.requires_grad]
        params_fc = [p for n, p in self.model.named_parameters() if
                     p.requires_grad and 'fc' == n[:2] or 'conv1d' == n[:6]]

        init_lr = self.hyper_params['optimizer']['lr']
        warmup_multiplier = 2.0
        lr_multiplier = max(0.5, batch_size / 32)

        # lr_multiplier = max(1.0, batch_size / 32)
        # skeleton.optim.gradual_warm_up(
        #     skeleton.optim.get_reduce_on_plateau_scheduler(
        #         init_lr * lr_multiplier / warmup_multiplier,
        #         patience=10, factor=.5, metric_name='train_loss'
        #     ),
        #     warm_up_epoch=5,
        #     multiplier=warmup_multiplier
        # )

        scheduler_lr = skeleton.optim.get_change_scale(
            skeleton.optim.gradual_warm_up(
                skeleton.optim.get_reduce_on_plateau_scheduler(
                    init_lr * lr_multiplier / warmup_multiplier,
                    patience=10, factor=.5, metric_name='train_loss'
                ),
                warm_up_epoch=5,
                multiplier=warmup_multiplier
            ),
            init_scale=1.0
        )

        # self.optimizer_fc = skeleton.optim.ScheduledOptimizer(
        #     params_fc,
        #     torch.optim.SGD,
        #     # skeleton.optim.SGDW,
        #     steps_per_epoch=steps_per_epoch,
        #     clip_grad_max_norm=None,
        #     lr=scheduler_lr,
        #     momentum=0.9,
        #     weight_decay=0.00025,
        #     nesterov=True
        # )

        self.optimizer[self._model_name] = skeleton.optim.ScheduledOptimizer(
            params,
            torch.optim.SGD,
            # skeleton.optim.SGDW,
            steps_per_epoch=steps_per_epoch,
            clip_grad_max_norm=None,
            lr=scheduler_lr,
            momentum=0.9,
            weight_decay=0.00025,
            nesterov=True
        )
        LOGGER.info('[optimizer] %s (batch_size:%d)', self.optimizer[self._model_name]._optimizer.__class__.__name__, batch_size)

    @timeit
    def update_model(self):
        num_class = self.info['dataset']['num_class']

        epsilon = min(0.1, max(0.001, 0.001 * pow(num_class / 10, 2)))
        if self.is_multiclass():
            self.model.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
            # self.model.loss_fn = skeleton.nn.BinaryCrossEntropyLabelSmooth(num_class, epsilon=epsilon, reduction='none')
            self.tau = 8.0
            LOGGER.info('[update_model] %s (tau:%f, epsilon:%f)', self.model.loss_fn.__class__.__name__, self.tau, epsilon)
        else:
            self.model.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            # self.model.loss_fn = skeleton.nn.CrossEntropyLabelSmooth(num_class, epsilon=epsilon)
            self.tau = 8.0
            LOGGER.info('[update_model] %s (tau:%f, epsilon:%f)', self.model.loss_fn.__class__.__name__, self.tau, epsilon)
        self.model_pred.loss_fn = self.model.loss_fn

        if self.is_video():
            # not use fast auto aug
            self.hyper_params['conditions']['use_fast_auto_aug'] = False
            times = self.hyper_params['dataset']['input'][0]
            log("times: {}".format(times))
            self.model.set_video(times=times)
            self.model_pred.set_video(times=times)

        self.init_opt()
        LOGGER.info('[update] done.')

    def is_change_model(self):
        return self._model_name != self._last_model_name

    def get_model_name(self):
        return self._model_name

    def select_model(self):
        self._last_model_name = self._model_name

        if len(self._last_k_queue['valid']) == self._last_k_num:
            log("diff_last_k_queue: {}".format(self._last_k_queue['valid'][-1] - self._last_k_queue['valid'][0]))

        if (len(self._last_k_queue['valid']) == self._last_k_num) and \
                (self._last_k_queue['valid'][-1] - self._last_k_queue['valid'][0] <=
                 max(self.change_model_valid_threshold - self.complexmodel_overfit_time * 0.005, 0.010) or
                 # self.change_model_valid_threshold or
                    self._last_k_queue['train'][-1] > 0.995):

            if self._is_use_second_model is False:
                self.complexmodel_overfit_time += 1
                self.init_opt()
                log("Over fit {} times !!!".format(self.complexmodel_overfit_time))
                return

            if self._model_id == 0:
                # self._model_id = (self._model_id + 1) % len(self.model_sequence)
                self._is_use_ckp_ensemble = False

                # delete previous model
                del self._models[self._model_name]
                del self._models_pred[self._model_name]
                del self.optimizer[self._model_name]
                del self.model
                del self.model_pred
                self.model = None
                self.model_pred = None
                torch.cuda.empty_cache()
                gc.collect()

                self._model_id += 1
                self._model_round += 1
                self._model_name = self.model_sequence[self._model_id]

                LOGGER.info('[init] Model_{} copy to device'.format(self._model_name))
                self._models[self._model_name] = self._models[self._model_name].to(device=self.device, non_blocking=True)  # .half()
                self._models_pred[self._model_name] = self._models_pred[self._model_name].to(device=self.device, non_blocking=True)  # .half()
                self.model = self._models[self._model_name]
                self.model_pred = self._models_pred[self._model_name]
                self.is_half = self._models[self._model_name]._half
                # torch.cuda.synchronize()
            else:
                self.complexmodel_overfit_time += 1
                self.init_opt()
                # else:
                #     self.init_opt()
            log("Over fit {} times !!!".format(self.complexmodel_overfit_time))
        else:
            self._model_name = self.model_sequence[self._model_id]

        if self.is_change_model():
            self._each_model_use_time[self._model_id] += 1
            if self._model_id == 0:
                self._model_seq_round += 1

        # if self._model_name in self._models:
        #     self._model = self._models[self._model_name]
        # else:
        #     self.build_model(self._model_name)
        #
        # self.model = self._models[self._model_name]
        # self.model_pred = self._models_pred[self._model_name]

    def select_model_video(self):
        self._last_model_name = self._model_name
        is_over_fit = False

        if len(self._last_k_queue['valid']) == self._last_k_num:
            log("diff_last_k_queue: {}".format(self._last_k_queue['valid'][-1] - self._last_k_queue['valid'][0]))

        if (len(self._last_k_queue['valid']) == self._last_k_num) and \
                (self._last_k_queue['valid'][-1] - self._last_k_queue['valid'][0] <=
                 max(self.change_model_valid_threshold - self.complexmodel_overfit_time * 0.005, 0.015) or
                 # self.change_model_valid_threshold or
                 self._last_k_queue['train'][-1] > 0.995):
            is_over_fit = True
            self.complexmodel_overfit_time += 1
            if (self.complexmodel_overfit_time == 1) or (self.complexmodel_overfit_time >= 3 and self.complexmodel_overfit_time % 3 == 0):
                self.init_opt()
            # else:
            #     self.init_opt()
            log("Over fit {} times !!!".format(self.complexmodel_overfit_time))

        self._model_name = self.model_sequence[self._model_id]