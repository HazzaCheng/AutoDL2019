import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import globals
from util.stages import Stage
from util.tools import batch_split, round_pow2
from tabular.feature_engineer.col_type import infer_type
from tabular.feature_engineer.env import NFSEnv
from tabular.feature_engineer.op import Op
from tabular.feature_engineer.agent import NFSAgent, RandomAgent

from tools import log, timeit
from classifiers.traditional_model import LgbClassifier, MultiClassifier, XgbClassifier


class RetrieveStage(Stage):
    @timeit
    def __init__(self, batch_size=256, **kwargs):
        super(RetrieveStage, self).__init__(**kwargs)
        sample_count = self.ctx.metadata.size()
        self._batch_size = min(batch_size, round_pow2(sample_count * 0.10))
        self.next_batch = None
        self._generator = None
        self._num_boost = 1
        self._kwargs = kwargs
        self._max_num_boost = 200
        log("Batch size: {} in retrieve stage".format(self._batch_size))

    @property
    def num_boost(self):
        num_boost = self._num_boost
        if self._num_boost < self._max_num_boost:
            self._num_boost *= 2
            # DEBUG
            # self._num_boost *= 1
        return num_boost

    @timeit
    def generate_data(self):
        try:
            instances, labels = self.ctx.sess.run(self.next_batch)
            instances = batch_split(instances)
            instances = [np.squeeze(instance) for instance in instances]
            labels = batch_split(labels)
            if self.ctx.is_first_train:
                # judge whether multi labels
                self.ctx.is_multi_labels = np.sum(labels, axis=-1).max() > 1
                log("multi? {}".format(self.ctx.is_multi_labels))

            if self.ctx.is_multi_labels:
                labels = np.asarray(labels)
            else:
                labels = np.where(np.asarray(labels) == 1)[1]
            return instances, labels
        except tf.errors.OutOfRangeError:
            self._need_transition = True
            self.ctx.is_ensemble = True
            self._num_boost = self._max_num_boost
            return None, None

    def transition(self):
        _, y = self.ctx.train_data
        import pandas as pd
        log("{}".format(pd.Series(y).value_counts()))
        if not self.ctx.is_multi_labels:
            self.ctx.stage = FeatureStage(**self._kwargs)
        # self.ctx.stage = ModelStage(**self._kwargs)

    @timeit
    def train(self, dataset, remain_time_budget=None):
        super(RetrieveStage, self).train(dataset, remain_time_budget)
        if self.ctx.is_first_train:
            self.next_batch = self.ctx.raw_dataset\
                .batch(self._batch_size)\
                .make_one_shot_iterator()\
                .get_next()
        log("Current boost num: {}".format(self._num_boost))
        x, y = self.generate_data()
        self.ctx.is_first_train = False
        self.ctx.update_train_data(x, y)

        x_all, y_all = self.ctx.train_data
        x_all = np.asarray(x_all, dtype=np.float)
        # if self._model is not None:
        #     return
        if self.ctx.is_multi_labels:
            self._model = MultiClassifier(
                num_boost_round=self.num_boost,
            )
        else:
            self._model = XgbClassifier(
                self.ctx.num_class,
                num_boost_round=self.num_boost,
                early_stopping_rounds=20
            )
        self._model.fit((x_all, y_all))

    @timeit
    def test(self, dataset, remain_time_budget=None):
        if self._need_transition:
            self.ctx.config["feature_importance"] = self._model.feature_importance
            log("feature importance:\n{}".format(self._model.feature_importance))
        super(RetrieveStage, self).test(dataset, remain_time_budget)
        x_test = self.ctx.test_data
        y_test_hat = self._model.predict(x_test)
        return y_test_hat


class FeatureStage(Stage):

    @timeit
    def __init__(self, **kwargs):
        super(FeatureStage, self).__init__(**kwargs)

        # hyper params
        self._num_batch = 1
        self._epochs = 1
        self._selected_feature_num = 20
        self._num_boost_round = 1

        x, y = self.ctx.train_data
        # self._col_types = infer_type(x)
        self._kwargs = kwargs

        self._selected_features = np.argsort(self.ctx.config["feature_importance"])[-self._selected_feature_num:]
        self._todo_x = x[:, self._selected_features]

        self._env = NFSEnv(
            (self._todo_x, y),
            self.ctx,
            max_order=5
        )
        self._agent = RandomAgent(
            num_batch=self._num_batch,
            env=self._env
        )
        self._init_config()
        # log("Col types:\n{}".format(self._col_types))

    def update_num_boost_round(self):
        globals.put(
            ["cls_kwargs", "num_boost_round"],
            self.num_boost_round
        )

    @property
    def num_boost_round(self):
        res = int(self._num_boost_round)
        self._num_boost_round += 0.5
        return res

    def _init_config(self):
        globals.put("cls_class", LgbClassifier)
        globals.put("cls_args", [self.ctx.num_class])
        globals.put(
            "cls_kwargs",
            {
                "num_threads": 1,
                "num_boost_round": self.num_boost_round,
                "early_stopping_rounds": 5,
                "verbose_eval": False
            }
        )
        _, y = self.ctx.train_data
        auc = self._env.evaluate(self._todo_x, y)
        globals.put("origin_result", auc)
        log("original auc {}".format(auc))

    @timeit
    def generate_data(self):
        self.update_num_boost_round()
        for _ in range(self._epochs):
            concat_action = []
            action_probs = self._agent.decide(None)
            log("action_probs {}:\n{}".format(action_probs.shape, action_probs))
            for _ in range(self._num_batch):
                batch_action = []
                for i in range(action_probs.shape[0]):
                    sample_action = np.random.choice(len(action_probs[i]), p=action_probs[i])
                    batch_action.append(sample_action)
                concat_action.append(batch_action)
            _, concat_rewards, done = self._env.step(concat_action)
            self._agent.learn(concat_action, concat_rewards)
        x, y = self.ctx.train_data
        return np.concatenate([x, self._env.translate(self._todo_x)], axis=1), y

    def transition(self):
        self.ctx.stage = ModelStage(**self._kwargs)

    @timeit
    def train(self, dataset, remain_time_budget=None):
        super().train(dataset, remain_time_budget)
        x, y = self.generate_data()
        log("x {}".format(x.shape))
        self._model = LgbClassifier(
            self.ctx.num_class,
            num_boost_round=200,
            early_stopping_rounds=20
        )
        self._model.fit((x, y))

    def test(self, dataset, remain_time_budget=None):
        super(FeatureStage, self).test(dataset, remain_time_budget)
        x_test = self.ctx.test_data
        x_test_extend = self._env.translate(x_test[:, self._selected_features])
        x_test = np.concatenate(
            [x_test, x_test_extend],
            axis=1)
        log("x_test {}".format(x_test.shape))
        y_test_hat = self._model.predict(x_test)
        return y_test_hat


class ModelStage(Stage):

    @timeit
    def __init__(self, **kwargs):
        super(ModelStage, self).__init__(**kwargs)
        
    def generate_data(self):
        return self.ctx.train_data

    def transition(self):
        pass

    @timeit
    def train(self, dataset, remain_time_budget=None):
        super().train(dataset, remain_time_budget)
        x_all, y_all = self.generate_data()
        x_search, _, y_search, _ = train_test_split(
            x_all, y_all,
            test_size=0.4
        )
        hyper_params = LgbClassifier.hyper_params_search(
            (x_search, y_search),
            params={
                "num_class": self.ctx.num_class,
                "num_boost_round": 150,
                "early_stopping_rounds": 20
            }
        )
        self._model = LgbClassifier(
            self.ctx.num_class,
            **hyper_params
        )
        self._model.fit((x_all, y_all))

    def test(self, dataset, remain_time_budget=None):
        super().test(dataset, remain_time_budget)
        x_test = self.ctx.test_data
        y_test_hat = self._model.predict(x_test)
        return y_test_hat
