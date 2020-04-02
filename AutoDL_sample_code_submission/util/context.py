import numpy as np
import tensorflow as tf
from multiprocessing import Pool

from util.tools import batch_split
from classifiers.traditional_model import XgbClassifier


class Context:
    def __init__(self, sess, metadata, config=None, lgb_params=None):
        self.sess = sess
        self.metadata = metadata
        self.is_first_train = True

        self._train_x, self._train_y = None, None
        self._test_x = None
        self._stage = None

        self.num_class = self.metadata.get_output_size()

        self._config = config if config is not None\
            else {
                "lgb_params": lgb_params if lgb_params is not None\
                else {
                    "objective": "multiclass",
                    "num_class": self.metadata.get_output_size()
                }
            }

        self.is_ensemble = False

        self._mp = None

        self._init_model = XgbClassifier(
            self.num_class,
            num_boost_round=1,
            early_stopping_rounds=20
        )
        self._init_model.fit((np.random.randn(2, 10), np.random.randint(2, size=2)))

    def update_train_data(self, x, y):
        if x is not None and y is not None:
            if self._train_x is None and self._train_y is None:
                self._train_x = np.asarray(x, dtype=np.float)
                self._train_y = np.asarray(y)
            else:
                self._train_x = np.append(self._train_x, x, axis=0)
                self._train_y = np.append(self._train_y, y, axis=0)

    @property
    def train_data(self):
        return self._train_x, self._train_y
    
    @property
    def test_data(self):
        return self._test_x

    @test_data.setter
    def test_data(self, dataset):
        self._test_x = []
        next_batch = dataset.batch(1024).make_one_shot_iterator().get_next()
        while True:
            try:
                instances = self.sess.run(next_batch)[0]
                instances = batch_split(instances)
                instances = [np.squeeze(instance).reshape(-1) for instance in instances]
                self._test_x.extend(instances)
            except tf.errors.OutOfRangeError:
                break
        self._test_x = np.asarray(self._test_x)

    @property
    def config(self):
        return self._config

    @property
    def stage(self):
        return self._stage

    @stage.setter
    def stage(self, value):
        self._stage = value

    @property
    def mp(self):
        if self._mp is None:
            self._mp = Pool()
        return self._mp

    @property
    def num_feature(self):
        return self._train_x.shape[1]
