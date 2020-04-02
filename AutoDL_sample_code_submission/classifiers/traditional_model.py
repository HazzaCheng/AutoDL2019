import os
import lightgbm as lgb
import numpy as np
try:
    import xgboost as xgb
except ImportError:
    os.system("pip3 install xgboost")
from sklearn.linear_model import ElasticNet
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from hyperopt import hp

from classifiers import RANDOM_STATE
from classifiers.base import Classifier
from hyper_search.bayes import bayes_opt
from metrics import auc_metric, acc_metric
from tools import timeit


class ElasticNetClassifier(Classifier):

    def __init__(self, *args, **kwargs):
        super(ElasticNetClassifier, self).__init__()
        self._model = ElasticNet(max_iter=10000, random_state=RANDOM_STATE)

    def fit(self, dataset, *args, **kwargs):
        x, y = dataset
        self._model.fit(x, y)

    def predict(self, data, *args, **kwargs):
        y_test_hat = self._model.predict(data)
        if len(y_test_hat.shape) == 1:
            res = 1 - y_test_hat
            y_test_hat = np.concatenate([res.reshape(-1, 1), y_test_hat.reshape(-1, 1)], axis=1)
        return y_test_hat

    @staticmethod
    def hyper_params_search(dataset, *args, **kwargs):
        x, y = dataset
        hyper_space = {
            'l1_ratio': hp.uniform('l1_ratio', .01, .9),
            'alpha': hp.uniform('alpha', .01, 1)
        }
        hyper_params = bayes_opt(x, y, {}, hyper_space, ElasticNetClassifier, roc_auc_score)
        return hyper_params


class LgbClassifier(Classifier):

    def __init__(self, num_class, **kwargs):
        super(LgbClassifier, self).__init__()
        self._model_params = {
            "objective": "multiclass",
            "num_class": num_class,
            "verbosity": -1,
            "seed": RANDOM_STATE,
        }
        self._model_hyper_params = {}
        default_args = lgb.train.__code__.co_varnames
        for k, v in kwargs.items():
            if k in default_args:
                self._model_hyper_params[k] = v
            else:
                self._model_params[k] = v
        self._model = None

    def fit(self, dataset, *args, **kwargs):
        x, y = dataset
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y,
            test_size=0.2, random_state=RANDOM_STATE
        )
        train_data = lgb.Dataset(x_train, label=y_train)
        valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data)
        self._model = lgb.train(
            self._model_params,
            train_set=train_data,
            valid_sets=valid_data,
            **self._model_hyper_params
        )
        self._model.feature_importance()

    def predict(self, data, *args, **kwargs):
        return self._model.predict(data)

    @staticmethod
    def hyper_params_search(dataset, *args, params=None, **kwargs):
        x, y = dataset
        params = {} if params is None else params
        hyper_space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
            "max_depth": hp.choice("max_depth", [-1, 5, 6, 7]),
            "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 40, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.6, 1.0, 0.1),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.6, 1.0, 0.1),
            "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
            # "num_boost_round": hp.choice("num_boost_round", np.linspace(100, 300, 50, dtype=int)),
            # "early_stopping_rounds": hp.choice("early_stopping_rounds", np.linspace(10, 30, 5, dtype=int))
        }
        hyper_params = bayes_opt(
            x, y,
            params,
            hyper_space, LgbClassifier, auc_metric,
            max_evals=30
        )
        return hyper_params

    @property
    def feature_importance(self):
        return self._model.feature_importance()


class MultiClassifier(Classifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = OneVsRestClassifier(lgb.LGBMClassifier(*args, **kwargs))

    def fit(self, dataset, *args, **kwargs):
        X, y = dataset
        self._model.fit(X, y)

    def predict(self, data, *args, **kwargs):
        return self._model.predict_proba(data)


class XgbClassifier(Classifier):
    @timeit
    def __init__(self, num_class, num_boost_round, **kwargs):
        super(XgbClassifier, self).__init__()
        self._model_params = {
            "objective": "multi:softmax",
            "num_class": num_class,
            "n_estimators": num_boost_round,
            "tree_method": "gpu_hist",
            'predictor': 'gpu_predictor',
            # "n_gpus": -1
        }
        self._model = None
        self._model = xgb.XGBClassifier(**self._model_params)

    @timeit
    def fit(self, dataset, *args, **kwargs):
        x, y = dataset
        self._model.fit(x, y)

    def predict(self, data, *args, **kwargs):
        return self._model.predict_proba(data)

    @property
    def feature_importance(self):
        return self._model.feature_importances_
