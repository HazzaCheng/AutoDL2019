# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import threading
import random
import abc
import tensorflow as tf
import torch
import torchvision as tv
import numpy as np
import sys
import copy

from .import skeleton
from .architectures.resnet import ResNet18, ResNet34
from .architectures.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2
from .architectures.squeezenet import SqueezeneNet11
from .architectures.mobilenet import MobileNetv2
from .skeleton.projects import LogicModel, get_logger
from .skeleton.projects.others import NBAC, AUC
from .skeleton.utils.tools import *
from .architectures import *
from keras.utils import to_categorical

torch.backends.cudnn.benchmark = True
threads = [
    threading.Thread(target=lambda: torch.cuda.synchronize()),
    threading.Thread(target=lambda: tf.Session())
]
[t.start() for t in threads]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LOGGER = get_logger(__name__)


def set_random_seed_all(seed, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_random_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Model(LogicModel):
    def __init__(self, metadata):
        # set_random_seed_all(0xC0FFEE)
        super(Model, self).__init__(metadata)
        self.use_test_time_augmentation = False
        self.update_transforms = False

        self.optimizer = {}
        self.optimizer_fc = None
        self.tau = None

        self._kbest_checkpoints = {}
        self._last_k_num = 3

    def select_model(self):
        raise NotImplementedError

    def select_model_video(self):
        raise NotImplementedError

    def is_change_model(self):
        raise NotImplementedError

    def get_model_name(self):
        raise NotImplementedError

    def build_model(self, model_name, to_device):
        raise NotImplementedError

    @timeit
    def build(self):
        # torch.cuda.synchronize()

        LOGGER.info('[init] session')
        [t.join() for t in threads]

        self.device = torch.device('cuda', 0)
        self.session = tf.Session()

        for i in range(len(self.model_sequence)):
            model_name = self.model_sequence[i]
            if i == 0:
                # self.build_model(model_name, model_dir='meta_models/meta-lr-ResNet18-8.pkl')
                self.build_model(model_name)
            else:
                self.build_model(model_name, to_device=False)

            self.info['loop']['best_score']['train'][model_name] = 0.0
            self.info['loop']['best_score']['valid'][model_name] = 0.0

    def adapt(self, remaining_time_budget=None):
        epoch = self.info['loop']['epoch']
        input_shape = self.hyper_params['dataset']['input']
        height, width = input_shape[:2]
        batch_size = self.hyper_params['dataset']['batch_size']

        train_score = np.average([c['train']['score'] for c in self.checkpoints[self._model_name][-5:]])
        valid_score = np.average([c['valid']['score'] for c in self.checkpoints[self._model_name][-5:]])
        LOGGER.info('[adapt] [%04d/%04d] train:%.3f valid:%.3f',
                    epoch, self.hyper_params['dataset']['max_epoch'],
                    train_score, valid_score)

        self.use_test_time_augmentation = self.info['loop']['test'] > 1

        if self.hyper_params['conditions']['use_fast_auto_aug']:
            self.hyper_params['conditions']['use_fast_auto_aug'] = valid_score < 0.995

        '''
        if self.hyper_params['conditions']['use_fast_auto_aug'] and \
                (train_score > 0.995 or self.info['terminate']) and \
                remaining_time_budget > 120 and \
                valid_score > 0.01 and \
                self.dataloaders['valid'] is not None and \
                not self.update_transforms:

            LOGGER.info('[adapt] use all data after this time')
            self.update_transforms = True
            self.info['terminate'] = True

            self._is_use_all_data = True

            self.hyper_params['optimizer']['lr'] /= 2.0
            self.init_opt()
            self.hyper_params['conditions']['max_inner_loop_ratio'] *= 3
            self.hyper_params['conditions']['threshold_valid_score_diff'] = 0.00001
            self.hyper_params['conditions']['min_lr'] = 1e-8
        '''

        '''
        # Adapt Apply Fast auto aug
        if self.hyper_params['conditions']['use_fast_auto_aug'] and \
                (train_score > 0.995 or self.info['terminate']) and \
                remaining_time_budget > 120 and \
                valid_score > 0.01 and \
                self.dataloaders['valid'] is not None and \
                not self.update_transforms:
            LOGGER.info('[adapt] search fast auto aug policy')
            self.update_transforms = True
            self.info['terminate'] = True

            original_valid_policy = self.dataloaders['valid'].dataset.transform.transforms
            policy = skeleton.data.augmentations.autoaug_policy()

            num_policy_search = 100
            num_sub_policy = 3
            # num_select_policy = 5
            num_select_policy = 1
            searched_policy = []
            for policy_search in range(num_policy_search):
                selected_idx = np.random.choice(list(range(len(policy))), num_sub_policy)
                selected_policy = [policy[i] for i in selected_idx]

                self.dataloaders['valid'].dataset.transform.transforms = original_valid_policy + [
                    lambda t: t.cpu().float() if isinstance(t, torch.Tensor) else torch.Tensor(t),
                    tv.transforms.ToPILImage(),
                    skeleton.data.augmentations.Augmentation(
                        selected_policy
                    ),
                    tv.transforms.ToTensor(),
                    lambda t: t.to(device=self.device) #.half()
                ]

                metrics = []
                for policy_eval in range(num_sub_policy * 2):
                    valid_dataloader = self.build_or_get_dataloader('valid', self.datasets['valid'], self.datasets['num_valids'])
                    # original_valid_batch_size = valid_dataloader.batch_sampler.batch_size
                    # valid_dataloader.batch_sampler.batch_size = batch_size

                    valid_metrics = self.epoch_valid(self.info['loop']['epoch'], valid_dataloader, reduction='max')

                    # valid_dataloader.batch_sampler.batch_size = original_valid_batch_size
                    metrics.append(valid_metrics)
                loss = np.max([m['loss'] for m in metrics])
                score = np.max([m['score'] for m in metrics])
                LOGGER.info('[adapt] [FAA] [%02d/%02d] score: %f, loss: %f, selected_policy: %s',
                            policy_search, num_policy_search, score, loss, selected_policy)

                searched_policy.append({
                    'loss': loss,
                    'score': score,
                    'policy': selected_policy
                })

            flatten = lambda l: [item for sublist in l for item in sublist]

            # filtered valid score
            searched_policy = [p for p in searched_policy if p['score'] > valid_score]

            if len(searched_policy) > 0:
                policy_sorted_index = np.argsort([p['score'] for p in searched_policy])[::-1][:num_select_policy]
                # policy_sorted_index = np.argsort([p['loss'] for p in searched_policy])[:num_select_policy]
                policy = flatten([searched_policy[idx]['policy'] for idx in policy_sorted_index])
                policy = skeleton.data.augmentations.remove_duplicates(policy)

                LOGGER.info('[adapt] [FAA] scores: %s', [searched_policy[idx]['score'] for idx in policy_sorted_index])

                original_train_policy = self.dataloaders['train'].dataset.transform.transforms
                self.dataloaders['train'].dataset.transform.transforms = original_train_policy + [
                    lambda t: t.cpu().float() if isinstance(t, torch.Tensor) else torch.Tensor(t),
                    tv.transforms.ToPILImage(),
                    skeleton.data.augmentations.Augmentation(
                        policy
                    ),
                    tv.transforms.ToTensor(),
                    lambda t: t.to(device=self.device) #.half()
                ]

            self.dataloaders['valid'].dataset.transform.transforms = original_valid_policy

            # reset optimizer pararms
            # self.model.init()
            self.hyper_params['optimizer']['lr'] /= 2.0
            self.init_opt()
            self.hyper_params['conditions']['max_inner_loop_ratio'] *= 3
            self.hyper_params['conditions']['threshold_valid_score_diff'] = 0.00001
            self.hyper_params['conditions']['min_lr'] = 1e-8
        '''
    def activation(self, logits):
        if self.is_multiclass():
            logits = torch.sigmoid(logits)
            prediction = (logits > 0.5).to(logits.dtype)
        else:
            logits = torch.softmax(logits, dim=-1)
            _, k = logits.max(-1)
            prediction = torch.zeros(logits.shape, dtype=logits.dtype, device=logits.device).scatter_(-1, k.view(-1, 1), 1.0)
        return logits, prediction

    @timeit
    def process_data(self, train):
        X = []
        y = []
        y_origin = []
        class_set = [0] * self.info['dataset']['num_class']

        for step, (examples, labels) in enumerate(train):
            if examples.shape[0] == 1:
                examples = examples[0]
                labels = labels[0]
            original_labels = labels
            if not self.is_multiclass():
                labels = labels.argmax(dim=-1)

            X.append(examples)
            y.append(labels)
            y_origin.append(original_labels)
            # labels_numpy = labels.numpy()

            # if self.is_multiclass():
            #     for label_i in labels_numpy:
            #         for i in range(label_i.shape[0]):
            #             class_set[i] += 1 if label_i[i] == 1 else 0
            # else:
            #     for v in labels_numpy:
            #         class_set[v] += 1

        # indices = np.argsort(
        #     np.array([-v for v in class_set]))
        # class_set_new = [class_set[i] for i in indices]
        # log("class_______id: {}".format(indices))
        # log("each_class_num: {}".format(class_set_new))

        return X, y, y_origin

    @timeit
    def epoch_train(self, epoch, train, model=None, optimizer=None):
        model = model if model is not None else self.model
        # if epoch < 0:
        #     optimizer = optimizer if optimizer is not None else self.optimizer_fc
        # else:
        #     optimizer = optimizer if optimizer is not None else self.optimizer
        optimizer = optimizer if optimizer is not None else self.optimizer[self._model_name]

        # batch_size = self.hyper_params['dataset']['batch_size']
        model.train()
        model.zero_grad()

        num_steps = len(train)

        start_time = time.time()
        X, y, y_origin = self.process_data(train)

        if self._timer_num_process_data < 5:
            self._timer_num_process_data += 1
            self._process_data_time += time.time() - start_time

        inner_epoch = 0

        while inner_epoch < self._inner_epoch_num:

            LOGGER.debug('-----------------------------------------inner_epoch {}----------------------------------------------'.format(inner_epoch + 1))

            self._global_step += 1

            start_time = time.time()

            model.train()

            metrics = []
            for step in range(len(y)):
                examples = X[step]
                labels = y[step]
                original_labels = y_origin[step]

            # for step, (examples, labels) in enumerate(train):
            #     if examples.shape[0] == 1:
            #         examples = examples[0]
            #         labels = labels[0]
            #     original_labels = labels
            #     if not self.is_multiclass():
            #         labels = labels.argmax(dim=-1)
                skeleton.nn.MoveToHook.to((examples, labels), self.device, self.is_half)
                logits, loss = model(examples, labels, tau=self.tau, reduction='avg')
                loss = loss.sum()
                loss.backward()

                max_epoch = self.hyper_params['dataset']['max_epoch']
                optimizer.update(maximum_epoch=max_epoch)
                optimizer.step()
                model.zero_grad()

                logits, prediction = self.activation(logits.float())
                tpr, tnr, nbac = NBAC(prediction, original_labels.float())
                auc = AUC(logits, original_labels.float())

                score = auc if self.hyper_params['conditions']['score_type'] == 'auc' else float(nbac.detach().float())
                metrics.append({
                    'loss': loss.detach().float().cpu(),
                    'score': score,
                })

                LOGGER.debug(
                    '[train] [%02d_%02d] [%03d/%03d] loss:%.6f AUC:%.3f NBAC:%.3f tpr:%.3f tnr:%.3f, lr:%.8f',
                    epoch, inner_epoch + 1, step, num_steps, loss, auc, nbac, tpr, tnr,
                    optimizer.get_learning_rate()
                )

            train_loss = np.average([m['loss'] for m in metrics])
            train_score = np.average([m['score'] for m in metrics])
            optimizer.update(train_loss=train_loss)

            LOGGER.info(
                '[train] loss:(train:%.3f) score:(train:%.3f)',
                train_loss, train_score,
            )

            if self.info['loop']['test'] == 1:
                if train_loss > self._first_train_loss * 0.9:
                    self._inner_epoch_num = 2

            inner_epoch += 1

            if self._timer_num_train < 5:
                self._timer_num_train += 1
                self._each_train_time += time.time() - start_time

        log("_timer_num_process_data: {}  _timer_num_train : {}".format(self._timer_num_process_data, self._timer_num_train))
        if self._timer_num_process_data == 5 and self._timer_num_train == 5:
            self._process_data_time = self._process_data_time / self._timer_num_process_data / self.hyper_params['dataset']['steps_per_epoch']
            self._each_train_time = self._each_train_time / self._timer_num_train / self.hyper_params['dataset']['steps_per_epoch']
            self._estimate_per_training_time = self._process_data_time + self._each_train_time
            log("estimate_per_training_time : {}".format(self._estimate_per_training_time))

            self._timer_num_process_data = 6
            self._timer_num_train = 6

        return {
            'loss': train_loss,
            'score': train_score,
        }

    @timeit
    def epoch_valid(self, epoch, valid, reduction='avg'):
        test_time_augmentation = False
        self.model.eval()
        num_steps = len(valid)
        metrics = []
        tau = self.tau

        with torch.no_grad():
            for step, (examples, labels) in enumerate(valid):
                original_labels = labels
                if not self.is_multiclass():
                    labels = labels.argmax(dim=-1)

                batch_size = examples.size(0)

                # Test-Time Augment flip
                if self.use_test_time_augmentation and test_time_augmentation:
                    examples = torch.cat([examples, torch.flip(examples, dims=[-1])], dim=0)
                    labels = torch.cat([labels, labels], dim=0)

                # skeleton.nn.MoveToHook.to((examples, labels), self.device, self.is_half)
                logits, loss = self.model(examples, labels, tau=tau, reduction=reduction)

                # avergae
                if self.use_test_time_augmentation and test_time_augmentation:
                    logits1, logits2 = torch.split(logits, batch_size, dim=0)
                    logits = (logits1 + logits2) / 2.0

                logits, prediction = self.activation(logits.float())
                tpr, tnr, nbac = NBAC(prediction, original_labels.float())
                if reduction == 'avg':
                    auc = AUC(logits, original_labels.float())
                else:
                    auc = max([AUC(logits[i:i+16], original_labels[i:i+16].float()) for i in range(int(len(logits)) // 16)])

                score = auc if self.hyper_params['conditions']['score_type'] == 'auc' else float(nbac.detach().float())
                metrics.append({
                    'loss': loss.detach().float().cpu(),
                    'score': score,
                })

                LOGGER.debug(
                    '[valid] [%02d] [%03d/%03d] loss:%.6f AUC:%.3f NBAC:%.3f tpr:%.3f tnr:%.3f, lr:%.8f',
                    epoch, step, num_steps, loss, auc, nbac, tpr, tnr,
                    self.optimizer[self._model_name].get_learning_rate()
                )
            if reduction == 'avg':
                valid_loss = np.average([m['loss'] for m in metrics])
                valid_score = np.average([m['score'] for m in metrics])
            elif reduction in ['min', 'max']:
                valid_loss = np.min([m['loss'] for m in metrics])
                valid_score = np.max([m['score'] for m in metrics])
            else:
                raise Exception('not support reduction method: %s' % reduction)
        self.optimizer[self._model_name].update(valid_loss=np.average(valid_loss))

        return {
            'loss': valid_loss,
            'score': valid_score,
        }

    def skip_valid(self, epoch):
        LOGGER.debug('[valid] skip')
        return {
            'loss': 99.9,
            'score': epoch * 1e-4,
        }

    def swap(self, kbest_temp, i0, j0):
        temp = kbest_temp[i0]
        kbest_temp[i0] = kbest_temp[j0]
        kbest_temp[j0] = temp
        return kbest_temp

    def mysort(self, kbest_temp):
        _len = len(kbest_temp)
        valid_threshold = 0.1
        for i in range(_len - 1):
            for j in range(i+1, _len):
                if kbest_temp[j]['valid']['score'] > valid_threshold and \
                   kbest_temp[i]['valid']['score'] > valid_threshold:
                    if kbest_temp[j]['valid']['score'] > kbest_temp[i]['valid']['score']:
                        kbest_temp = self.swap(kbest_temp, i, j)
                elif kbest_temp[j]['valid']['score'] <= valid_threshold and \
                    kbest_temp[i]['valid']['score'] <= valid_threshold:
                    if kbest_temp[j]['train']['score'] > kbest_temp[i]['train']['score']:
                        kbest_temp = self.swap(kbest_temp, i, j)
                else:
                    if kbest_temp[j]['valid']['score'] > valid_threshold:
                        kbest_temp = self.swap(kbest_temp, i, j)
        return kbest_temp

    def kbest_list_create(self, model_state, pred):
        K = self._K_num
        new_ckp = model_state
        new_ckp['model_prediction'] = pred
        if len(self._global_kbest) < K:
            self._global_kbest.append(new_ckp)
        else:
            valid_score = new_ckp['valid']['score']
            train_score = new_ckp['train']['score']

            if valid_score > 0.01:
                best_score = valid_score
                if best_score > self._global_kbest[-1]['valid']['score']:
                    self._global_kbest[-1] = new_ckp
            else:
                best_score = train_score
                if best_score > self._global_kbest[-1]['train']['score']:
                    self._global_kbest[-1] = new_ckp
        self._global_kbest = self.mysort(self._global_kbest)
        return new_ckp

    @timeit
    def find_kbest_checkpoint(self):
        K = min(3, len(self.checkpoints[self._model_name]))
        indices = np.argsort(
            np.array([v['valid']['score'] for v in self.checkpoints[self._model_name]] if len(self.checkpoints[self._model_name]) > 0 else [0]))
        indices = sorted(indices[::-1][:K])
        self._kbest_checkpoints[self._model_name] = [self.checkpoints[self._model_name][i] for i in indices]

    @timeit
    def get_each_model_top_k_predicts(self):
        predicts = []
        for k, v in self._each_model_kbest.items():
            temp = []
            for i in range(len(v)):
                temp.append((v[i]['valid']['score'], v[i]['model_prediction']))
            temp.sort(key=lambda x: x[0], reverse=True)
            predicts.extend(temp[:self._each_model_top_k])

        if len(predicts) == 0:
            return [], []

        predicts = sorted(predicts, key=lambda x: x[0], reverse=True)[
                   :self._each_model_keep_num]
        top_k_aucs = [predicts[i][0] for i in range(len(predicts))]
        top_k_predicts = [predicts[i][1] for i in range(len(predicts))]

        return top_k_aucs, top_k_predicts

    @timeit
    def prediction(self, dataloader, model=None, test_time_augmentation=True, detach=True, num_step=None):
        # if self.is_change_model() and self._model_round > 0 and self._each_model_test_num[self._model_id] == 1:
        #     selected_k_best_score = [self._global_kbest[i]['valid']['score'] for i in range(len(self._global_kbest))]
        #     selected_k_best = [self._global_kbest[i]['model_prediction'] for i in range(len(self._global_kbest))]
        #     predictions_ensemble = selected_k_best
        #     log("selected_k_best_score: {}".format(selected_k_best_score))
        #     predictions_ensemble = np.array(predictions_ensemble)
        #     predictions_ensemble = np.mean(predictions_ensemble, axis=0)
        #     self._pre_predict_ensemble = predictions_ensemble
        #     return predictions_ensemble

        new_state = None
        if self._pre_train_state['valid']['score'] <= 0.1:
            pred_cur = self._prediction(dataloader, model_state=self._pre_train_state['model'])
            new_state = self.kbest_list_create(model_state=self._pre_train_state, pred=pred_cur)
        else:
            best_idx = np.argmax(np.array([c['valid']['score'] for c in self.checkpoints[self._model_name]]))
            pred_cur = self._prediction(dataloader, model_state=self.checkpoints[self._model_name][best_idx]['model'])
            new_state = self.kbest_list_create(model_state=self.checkpoints[self._model_name][best_idx], pred=pred_cur)

        log("_global_kbest_train_score: {}".format([self._global_kbest[i]['train']['score']
                                                    for i in range(len(self._global_kbest))]))
        log("_global_kbest_valid_score: {}".format([self._global_kbest[i]['valid']['score']
                                                    for i in range(len(self._global_kbest))]))

        if self.is_change_model():
            if self._each_model_use_time[self._model_id] == 1:
                self._each_model_kbest[self._model_name] = []
            self._each_model_kbest[self._model_name].append(new_state)
            # log("new_state: {}".format(new_state))

        score = None
        pre_score = None
        if new_state['valid']['score'] >= 0.01:
            score = new_state['valid']['score']
            pre_score = self._each_model_kbest[self._model_name][-1]['valid']['score']
        else:
            score = new_state['train']['score']
            pre_score = self._each_model_kbest[self._model_name][-1]['train']['score']
        if score > pre_score * 1.001:
            self._each_model_kbest[self._model_name][-1] = new_state

        if self.info['loop']['test'] <= 3:
            return pred_cur

        self._kbest_checkpoints[self._model_name] = []
        self.find_kbest_checkpoint()

        if self._pre_train_state['valid']['score'] >= 0.01:
            cur_valid_score_list = [v['valid']['score'] for v in self._kbest_checkpoints[self._model_name]]
            cur_train_score_list = [v['train']['score'] for v in self._kbest_checkpoints[self._model_name]]
            # if self._model_name in self._kbest_checkpoints:
            #     cur_valid_score_list.extend([v['valid']['score'] for v in self._kbest_checkpoints[self._model_name]])

            cur_valid_score_mean = np.array(cur_valid_score_list)
            log("cur_valid_score_mean_list: {}".format(cur_valid_score_mean))
            cur_valid_score_mean = np.mean(cur_valid_score_mean)
            log("cur_valid_score_mean: {}".format(cur_valid_score_mean))

            cur_train_score_mean = np.array(cur_train_score_list)
            log("cur_train_score_mean_list: {}".format(cur_train_score_mean))
            cur_train_score_mean = np.mean(cur_train_score_mean)
            log("cur_train_score_mean: {}".format(cur_train_score_mean))

            if len(self._last_k_queue['valid']) < self._last_k_num:
                self._last_k_queue['train'].append(cur_train_score_mean)
                self._last_k_queue['valid'].append(cur_valid_score_mean)
            else:
                self._last_k_queue['train'][0] = self._last_k_queue['train'][1]
                self._last_k_queue['train'][1] = self._last_k_queue['train'][2]
                self._last_k_queue['train'][2] = cur_train_score_mean

                self._last_k_queue['valid'][0] = self._last_k_queue['valid'][1]
                self._last_k_queue['valid'][1] = self._last_k_queue['valid'][2]
                self._last_k_queue['valid'][2] = cur_valid_score_mean

            log("_last_k_queue_train_score: {}".format(self._last_k_queue['train']))
            log("_last_k_queue_valid_score: {}".format(self._last_k_queue['valid']))

        predictions_ckp = []
        if self._is_use_ckp_ensemble:
            for idx in range(len(self._kbest_checkpoints[self._model_name]) - 1):
                if self._kbest_checkpoints[self._model_name][idx]['model_prediction'] is None:
                    pred_y = self._prediction(dataloader, model_state=self._kbest_checkpoints[self._model_name][idx]['model'])
                    self._kbest_checkpoints[self._model_name][idx]['model_prediction'] = pred_y
                    predictions_ckp.append(pred_y)
                else:
                    predictions_ckp.append(self._kbest_checkpoints[self._model_name][idx]['model_prediction'])
            # predictions_ckp.append(pred_cur)
            # predictions_ckp = np.array(predictions_ckp)
            # predictions_ckp = np.mean(predictions_ckp, axis=0)

        # predictions_ensemble = self._each_model_kbest[self._model_name]

        selected_k_best_score_train = [self._global_kbest[i]['train']['score'] for i in range(len(self._global_kbest))]
        selected_k_best_score = [self._global_kbest[i]['valid']['score'] for i in range(len(self._global_kbest))]
        selected_k_best = [self._global_kbest[i]['model_prediction'] for i in range(len(self._global_kbest))]

        if np.mean(selected_k_best_score_train) <= 0.70:
            return pred_cur

        # global_kbest + each_model_kbest
        # each_model_k_aucs = None
        # selected_each_model_k_best = None
        # if self._model_seq_round == 1 and self._model_id == 0:
        #     predictions_ensemble = selected_k_best
        # else:
        #     each_model_k_aucs, selected_each_model_k_best = self.get_each_model_top_k_predicts()
        #     predictions_ensemble = selected_k_best + selected_each_model_k_best

        predictions_ensemble = selected_k_best
        log("selected_k_best_score: {}".format(selected_k_best_score))
        # log('each_model_k_best_score: {}'.format(each_model_k_aucs))

        predictions_ensemble.extend(predictions_ckp)

        predictions_ensemble = np.array(predictions_ensemble)
        log("prediction_ensemble_shape: {}".format(predictions_ensemble.shape))

        predictions_ensemble = np.mean(predictions_ensemble, axis=0)

        self._pre_predict_ensemble = predictions_ensemble

        return predictions_ensemble

    @timeit
    def _prediction(self, dataloader, model_state, model=None, test_time_augmentation=True, detach=True, num_step=None):
        tau = self.tau
        if model is None:
            model = self.model_pred
            states = model_state
            model.load_state_dict(states)

        num_step = len(dataloader) if num_step is None else num_step

        model.eval()
        with torch.no_grad():
            predictions = []
            for step, (examples, labels) in zip(range(num_step), dataloader):
                batch_size = examples.size(0)

                # Test-Time Augment flip
                if self.use_test_time_augmentation and test_time_augmentation:
                    examples = torch.cat([examples, torch.flip(examples, dims=[-1])], dim=0)

                # skeleton.nn.MoveToHook.to((examples, labels), self.device, self.is_half)
                logits = model(examples, tau=tau)

                # avergae
                if self.use_test_time_augmentation and test_time_augmentation:
                    logits1, logits2 = torch.split(logits, batch_size, dim=0)
                    logits = (logits1 + logits2) / 2.0

                logits, prediction = self.activation(logits)

                if detach:
                    predictions.append(logits.detach().float().cpu().numpy())
                else:
                    predictions.append(logits)

            if detach:
                predictions = np.concatenate(predictions, axis=0).astype(np.float)
            else:
                predictions = torch.cat(predictions, dim=0)

        return predictions

    '''
    @timeit
    def prediction(self, dataloader, model=None, test_time_augmentation=True, detach=True, num_step=None):
        tau = self.tau
        if model is None:
            model = self.model_pred
            best_idx = np.argmax(np.array([c['valid']['score'] for c in self.checkpoints]))
            best_loss = self.checkpoints[best_idx]['valid']['loss']
            best_score = self.checkpoints[best_idx]['valid']['score']

            states = self.checkpoints[best_idx]['model']
            model.load_state_dict(states)
            LOGGER.info('best checkpoints at %d/%d (valid loss:%f score:%f) tau:%f',
                        best_idx + 1, len(self.checkpoints), best_loss, best_score, tau)

        num_step = len(dataloader) if num_step is None else num_step

        model.eval()
        with torch.no_grad():
            predictions = []
            for step, (examples, labels) in zip(range(num_step), dataloader):
                batch_size = examples.size(0)

                # Test-Time Augment flip
                if self.use_test_time_augmentation and test_time_augmentation:
                    examples = torch.cat([examples, torch.flip(examples, dims=[-1])], dim=0)

                # skeleton.nn.MoveToHook.to((examples, labels), self.device, self.is_half)
                logits = model(examples, tau=tau)

                # avergae
                if self.use_test_time_augmentation and test_time_augmentation:
                    logits1, logits2 = torch.split(logits, batch_size, dim=0)
                    logits = (logits1 + logits2) / 2.0

                logits, prediction = self.activation(logits)

                if detach:
                    predictions.append(logits.detach().float().cpu().numpy())
                else:
                    predictions.append(logits)

            if detach:
                predictions = np.concatenate(predictions, axis=0).astype(np.float)
            else:
                predictions = torch.cat(predictions, dim=0)
        
        return predictions
    '''