# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by: Zhengying Liu, Isabelle Guyon

"""An example of code submission for the AutoDL challenge.

It implements 3 compulsory methods ('__init__', 'train' and 'test') and
an attribute 'done_training' for indicating if the model will not proceed more
training due to convergence or limited time budget.

To create a valid submission, zip model.py together with other necessary files
such as Python modules/packages, pre-trained weights, etc. The final zip file
should not exceed 300MB.
"""
import os
import time

os.system("pip install cython")
import pyximport

pyximport.install()

try:
    import hyperopt
except ImportError:
    os.system("pip3 install hyperopt")

import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from domain_tools import infer_domain, get_domain_metadata, is_chinese
from features.nlp.nlp_data_manager import NlpDataManager
from features.nlp.nlp_features import load_embedding_dict, load_stopwords, NLP_FEATURE_DIR
from features.speech.speech_data_manager import SpeechDataManager
from models.nlp.nlp_model_manager import NlpModelManager
from models.speech.speech_model_manager import SpeechModelManager
from tools import log, timeit, set_random_seed_all


from Configurations import (CLASS_NUM, IS_LOAD_EMBEDDING, EN, ZH, EN_EMBEDDING_PATH, ZH_EMBEDDING_PATH,
                            NLP_PER_CLASS_READ_NUM, NLP_FAST_MODEL_RUN_LOOP, NLP_READ_CUT_LEN)
from tools import log, timeit
from tabular.model import TabularModel

from AutoCV.model_manager import ModelManager as AutoCVModel

os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
config.log_device_placement = False
# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
# set this TensorFlow session as the default session for Keras
set_session(sess)


# enable_eager_execution


class Model(object):
    """Trivial example of valid model. Returns all-zero predictions."""

    def __init__(self, metadata):
        """
        Args:
          metadata: an AutoDLMetadata object. Its definition can be found in
              AutoDL_ingestion_program/dataset.py
        """
        # set_random_seed_all(666)
        self.done_training = False
        self.train_loop_num = 0
        self._domain = infer_domain(metadata)
        log("The inferred domain of current dataset is: {}.".format(self._domain))
        self._domain_metadata = get_domain_metadata(metadata, self._domain)
        self._origin_metadata = metadata
        log("Metadata: {}".format(self._domain_metadata))

        self._data_manager = None
        self._model_manager = None

        if self._domain == "image" or self._domain == "video":
            self.CVModel = AutoCVModel(self._domain_metadata)
            return

        if self._domain == "tabular":
            self._tabular = TabularModel(metadata, sess)
            return

        if self._domain == 'text':
            if self._domain_metadata['language'] == EN:
                self._embedding_dict = load_embedding_dict(EN_EMBEDDING_PATH) if IS_LOAD_EMBEDDING else None
                self._stopwords = load_stopwords(NLP_FEATURE_DIR + '/english_stopwords.txt')
            elif self._domain_metadata['language'] == ZH:
                self._embedding_dict = load_embedding_dict(ZH_EMBEDDING_PATH) if IS_LOAD_EMBEDDING else None
                self._stopwords = load_stopwords(NLP_FEATURE_DIR + '/chinese_stopwords.txt')
            else:
                raise Exception('Unsupport language {}'.format(self._domain_metadata["language"]))

        self.session = None
        self._is_train_data_read = False
        self._is_test_data_read = False
        self._train_iterator = None
        self._next_element = None
        self._train_dataset = None
        self._test_dataset = None
        self._is_multilabel = False
        self._class_num = self._domain_metadata[CLASS_NUM]
        self._class_set = set()

        self._text_cut_len = NLP_READ_CUT_LEN
        self._need_reread = None

        if self._domain == "text":
            self.vocabulary = None
            self._is_use_fast_model = True #(self._domain_metadata['language'] == EN) and self._class_num == 2   # only use in EN text
            self._is_use_simple_model = self._class_num != 2

    def train(self, dataset, remaining_time_budget=None):
        """Train this algorithm on the tensorflow |dataset|.

        This method will be called REPEATEDLY during the whole training/predicting
        process. So your `train` method should be able to handle repeated calls and
        hopefully improve your model performance after each call.

        ****************************************************************************
        ****************************************************************************
        IMPORTANT: the loop of calling `train` and `test` will only run if
            self.done_training = False
          (the corresponding code can be found in ingestion.py, search
          'M.done_training')
          Otherwise, the loop will go on until the time budget is used up. Please
          pay attention to set self.done_training = True when you think the model is
          converged or when there is not enough time for next round of training.
        ****************************************************************************
        ****************************************************************************

        Args:
          dataset: a `tf.data.Dataset` object. Each of its examples is of the form
                (example, labels)
              where `example` is a dense 4-D Tensor of shape
                (sequence_size, row_count, col_count, num_channels)
              and `labels` is a 1-D Tensor of shape
                (output_dim,).
              Here `output_dim` represents number of classes of this
              multilabel classification task.

              IMPORTANT: some of the dimensions of `example` might be `None`,
              which means the shape on this dimension might be variable. In this
              case, some preprocessing technique should be applied in order to
              feed the training of a neural network. For example, if an image
              dataset has `example` of shape
                (1, None, None, 3)
              then the images in this datasets may have different sizes. On could
              apply resizing, cropping or padding in order to have a fixed size
              input tensor.

          remaining_time_budget: a float, time remaining to execute train(). The method
              should keep track of its execution time to avoid exceeding its time
              budget. If remaining_time_budget is None, no time budget is imposed.
        """
        try:
            # K.set_learning_phase(0)
            if self._domain == "tabular":
                self._tabular.train(dataset, remaining_time_budget)
                return

            if self.done_training:
                return
            self.train_loop_num += 1

            if self._domain in ['image', 'video']:
                self.CVModel.train(dataset, remaining_time_budget=remaining_time_budget)
                self.done_training = self.CVModel.done_training
                return

            if self._domain == 'speech':
                if self.train_loop_num == 1:
                    # dataset = dataset.map(lambda x, y: (x[:5*16000], y), num_parallel_calls=os.cpu_count())
                    read_num = 100
                    dataset = self._read_domain_dataset(dataset, is_training=True, read_num=read_num)
                elif self.train_loop_num == 2:
                    read_num = max(500, 5 * self._class_num)
                    dataset = self._read_domain_dataset(dataset, is_training=True, read_num=read_num)
                    if dataset is not None:
                        self._data_manager.add_data(dataset)
                elif self.train_loop_num == 3:
                    if self._class_num < 50:
                        read_num = int(self._domain_metadata['train_num'] * 0.5)
                    else:
                        read_num = 12 * self._class_num
                    dataset = self._read_domain_dataset(dataset, is_training=True, read_num=read_num)
                    if dataset is not None:
                        self._data_manager.add_data(dataset)
                elif self.train_loop_num == 4:
                    dataset = self._read_domain_dataset(dataset, is_training=True, read_num=None)
                    if dataset is not None:
                        self._data_manager.add_data(dataset)
            elif self._domain == 'text':
                if self.train_loop_num == 1 and self._text_cut_len:
                    dataset = dataset.map(lambda x, y: (x[:self._text_cut_len], y), num_parallel_calls=os.cpu_count())

                train_dataset = None
                if self.train_loop_num <= NLP_FAST_MODEL_RUN_LOOP and self._is_use_fast_model:
                    if self.train_loop_num == 1:
                        read_num = min(int(self._domain_metadata["train_num"]*0.15), 2000)
                    else:
                        upper_bound = 2000 if self._data_manager._is_balance else 3500
                        read_num = min(int(self._domain_metadata["train_num"] * 0.15), upper_bound)
                    train_dataset = self._read_domain_dataset(dataset, is_training=True, read_num=read_num)
                elif ((self.train_loop_num == NLP_FAST_MODEL_RUN_LOOP + 1 and self._is_use_fast_model) or
                        (self.train_loop_num == 1 and not self._is_use_fast_model)) and self._is_use_simple_model:
                    train_num = self._domain_metadata["train_num"]
                    read_num = min(max(int(0.7*train_num) - self._data_manager.get_sample_num(), 10000), 50000)
                    train_dataset = self._read_domain_dataset(dataset, is_training=True, read_num=read_num)
                elif self._model_manager.is_read_rest_1:
                    train_dataset = self._read_domain_dataset(dataset, is_training=True, read_num=100000)
                elif self._model_manager.is_read_rest_2:
                    self._need_reread = self.get_examples_mean_len(dataset) > self._text_cut_len
                    train_dataset = self._read_domain_dataset(dataset, is_training=True, read_num=None,
                                                              need_reread=self._need_reread)
                if train_dataset and self.train_loop_num > 1:
                    if self._model_manager.is_read_rest_2 and self._need_reread:
                        self._data_manager.reset_dataset(train_dataset)
                        log("Reread train dataset")
                    else:
                        self._data_manager.add_data(train_dataset)
                        self._model_manager.is_read_rest_1 = False
                        self._model_manager.is_read_rest_2 = False

            if self.train_loop_num == 1:
                if self._domain == 'text':
                    self._is_multilabel = self.is_multilabel(train_dataset[1])
                    self._data_manager = NlpDataManager(self._domain_metadata, train_dataset, self._is_multilabel,
                                                        vocabulary=self.vocabulary)
                    self._model_manager = NlpModelManager(self._domain_metadata, self._data_manager,
                                                          is_use_fast_model=self._is_use_fast_model,
                                                          is_use_simple_model=self._is_use_simple_model,
                                                          pretrained_embedding_matrix=self._embedding_dict,
                                                          is_multilabel=self._is_multilabel)
                elif self._domain == 'speech':
                    self._is_multilabel = self.is_multilabel(dataset[1])
                    self._data_manager = SpeechDataManager(self._domain_metadata, dataset, self._is_multilabel)
                    self._model_manager = SpeechModelManager(self._domain_metadata, self._data_manager,
                                                             is_multilabel=self._is_multilabel)
                    if self._is_multilabel:
                        train_dataset = self._read_domain_dataset(dataset, is_training=True, read_num=None)
                        self._data_manager.add_data(train_dataset)
                log("is_multilabel {}".format(self._is_multilabel))

            is_done = self._model_manager.fit(train_loop_num=self.train_loop_num,
                                              remaining_time_budget=remaining_time_budget)

            if is_done:
                self.done_training = True

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                self.done_training = True
                log("we met cuda out of memory")
            else:
                raise exception

    @timeit
    def test(self, dataset, remaining_time_budget=None):
        """Make predictions on the test set `dataset` (which is different from that
        of the method `train`).

        Args:
          Same as that of `train` method, except that the labels will be empty
              (all zeros) since this time `dataset` is a test set.
        Returns:
          predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
              here `sample_count` is the number of examples in this dataset as test
              set and `output_dim` is the number of labels to be predicted. The
              values should be binary or in the interval [0,1].
        """
        """Test method of domain-specific model."""
        try:
            if self._domain == "tabular":
                return self._tabular.test(dataset, remaining_time_budget)
            # Convert test dataset to necessary format and
            # store as self.domain_dataset_test
            # K.set_learning_phase(1)

            if self._domain in ['image', 'video']:
                pred_y = self.CVModel.test(dataset,
                                           remaining_time_budget=remaining_time_budget)

                self.done_training = self.CVModel.done_training
                return pred_y

            # As the original metadata doesn't contain number of test examples, we
            # need to add this information
            if self.train_loop_num == 1:
                if self._domain == 'text' and self._text_cut_len:
                    dataset = dataset.map(lambda x, y: (x[:self._text_cut_len], y), num_parallel_calls=os.cpu_count())
                self._test_dataset = self._read_domain_dataset(dataset, is_training=False)
            elif self._domain == 'text' and self._model_manager.is_read_rest_2 and self._need_reread:
                self._test_dataset = self._read_domain_dataset(dataset, is_training=False, need_reread=True)
                self._model_manager.is_read_rest_2 = False
                log("Reread test dataset")

                if self._domain in ['text', 'speech'] and \
                        (not self._domain_metadata['test_num'] >= 0):
                    self._domain_metadata['test_num'] = len(self._test_dataset)
                    print("test_num {}".format(self._domain_metadata['test_num']))

            # Make predictions
            pred_y = self._model_manager.predict(self._test_dataset)

            return pred_y

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                self.done_training = True
                log("we met cuda out of memory")
            else:
                raise exception

    @timeit
    def _read_train_data_from_tfrecord(self, dataset, read_num=None, need_reread=False):
        if self._is_train_data_read and not need_reread:
            return None, None
        # Only iterate the TF dataset when it's not done yet
        if self._next_element is None or need_reread:
            iterator = dataset.make_one_shot_iterator()
            self._next_element = iterator.get_next()
        train_sess = sess
        if need_reread:
            train_sess = tf.Session(config=config)
        X, Y = [], []
        cnt = 0
        next_element = self._next_element
        while read_num is None or cnt < read_num \
                or (self._domain == 'text' and (self._class_set.__len__() < self._class_num <= 5 or
                                                self._class_num > 5 and self._class_set.__len__() < int(self._class_num * 0.9))) \
                or (self._domain == 'speech' and (self._class_set.__len__() < self._class_num <= 5 or
                                                  self._class_num > 5 and self._class_set.__len__() < int(self._class_num * 0.9))):
            try:
                example, label = train_sess.run(next_element)
                X.append(example)
                Y.append(label)
                cnt += 1

                # should keep every class have one sample at least
                self._class_set.add(np.argmax(label))
            except tf.errors.OutOfRangeError:
                self._is_train_data_read = True
                break

        print("read {} train samples".format(cnt))

        return X, Y

    @timeit
    def _read_test_data_from_tfrecord(self, dataset, need_reread=False):
        if self._is_test_data_read and not need_reread:
            return None, None
        # Only iterate the TF dataset when it's not done yet
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        X, Y = [], []

        with tf.Session(config=config) as local_sess:
            while True:
                try:
                    example, label = local_sess.run(next_element)
                    X.append(example)
                    Y.append(label)
                except tf.errors.OutOfRangeError:
                    self._is_test_data_read = True
                    break

        return X, Y

    @timeit
    def _read_domain_dataset(self, dataset, is_training=True, read_num=None, need_reread=False):
        """Recover the dataset in corresponding competition format (esp. AutoNLP
        and AutoSpeech) and set corresponding attributes:
          self.domain_dataset_train
          self.domain_dataset_test
        according to `is_training`.
        """
        if self._domain == 'text':
            # Get X, Y as lists of NumPy array
            if is_training:
                X, Y = self._read_train_data_from_tfrecord(dataset, read_num=read_num, need_reread=need_reread)
                if X is None or X == []:
                    return None
            else:
                X, Y = self._read_test_data_from_tfrecord(dataset, need_reread=need_reread)

            # Retrieve vocabulary (token to index map) from metadata and construct
            # the inverse map
            # vocabulary = self._domain_metadata.get_channel_to_index_map()
            s = time.time()
            if self.vocabulary is None:
                self.vocabulary = self._origin_metadata.get_channel_to_index_map()
            vocabulary = self.vocabulary
            index_to_token = [None] * len(vocabulary)
            for token in vocabulary:
                index = vocabulary[token]
                index_to_token[index] = token

            # Get separator depending on whether the dataset is in Chinese
            if self._domain_metadata['language'] == ZH:
                log("is chinese")
                sep = ''
            elif self._domain_metadata['language'] == EN:
                log("is english")
                sep = ' '
            else:
                raise Exception("Unsupported language")

            # Construct the corpus
            corpus = []
            for x in X:  # each x in X is a list of indices (but as float)
                tokens = [index_to_token[int(i)] for i in x]
                document = sep.join(tokens)
                corpus.append(document)

            # Construct the dataset for training or test
            if is_training:
                labels = np.array(Y)
                domain_dataset = corpus, labels
            else:
                domain_dataset = corpus
            e = time.time()
            log("read vocabulary time {}".format((e - s)))
            # Set the attribute
            return domain_dataset
        elif self._domain == 'speech':
            if is_training:
                X, Y = self._read_train_data_from_tfrecord(dataset, read_num=read_num)
                if X is None or X == []:
                    return None
            else:
                X, Y = self._read_test_data_from_tfrecord(dataset)

            # Convert each array to 1-D array
            X = [np.squeeze(x) for x in X]

            # Construct the dataset for training or test
            if is_training:
                labels = np.array(Y)
                domain_dataset = X, labels
            else:
                domain_dataset = X

            return domain_dataset
        elif self._domain in ['image', 'video', 'tabular']:
            return dataset
        else:
            raise ValueError("The domain {} doesn't exist.".format(self._domain))

    def _text_cut_func(self, example, label):
        # if example.shape[0] > self._text_cut_len:
        #     begin_idx = np.random.randint(0, example.shape[0] - self._text_cut_len)
        #     example = example[begin_idx: begin_idx + self._text_cut_len]
        example = example[:self._text_cut_len]
        return example, label

    def get_examples_mean_len(self, dataset):
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        cnt = 0
        len_sum = 0.0

        with tf.Session(config=config) as local_sess:
            while cnt < 100:
                try:
                    example, labels = local_sess.run(next_element)
                    len_sum += example.shape[0]
                    cnt += 1
                except tf.errors.OutOfRangeError:
                    break
            return len_sum / cnt

    def is_multilabel(self, labels):
        is_multilabel = False
        for label in labels:
            if sum(label) > 1:
                is_multilabel = True
                break
        return is_multilabel


if __name__ == '__main__':
    from AutoDL_ingestion_program.dataset import AutoDLDataset
    import os

    data_dir = '/Users/chengfeng/Work/dataset/'
    # data_dir = r'C:\Users\90584\Desktop\Github\AutoClassifier\AutoDL_sample_data\O1'
    data_name = 'data02'  # speech
    # data_name = 'O1' # nlp
    D_train = AutoDLDataset(os.path.join(data_dir, data_name + '/' + data_name + '.data', "train"))
    D_test = AutoDLDataset(os.path.join(data_dir, data_name + '/' + data_name + '.data', "test"))

    m = Model(D_train.get_metadata())
    m.train(D_train.get_dataset(), remaining_time_budget=10000000)
    m.test(D_test.get_dataset(), remaining_time_budget=100000000)
    # m.train(D_train.get_dataset(), remaining_time_budget=10000000)
    # m.test(D_test.get_dataset(), remaining_time_budget=100000000)
    # m.train(D_train.get_dataset(), remaining_time_budget=10000000)
    # m.test(D_test.get_dataset(), remaining_time_budget=100000000)
    # m.train(D_train.get_dataset(), remaining_time_budget=10000000)
    # m.test(D_test.get_dataset(), remaining_time_budget=100000000)
