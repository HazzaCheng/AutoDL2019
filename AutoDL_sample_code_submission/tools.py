#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-22
import os
import random
import time
from typing import Any
import numpy as np
import functools
from keras.preprocessing import sequence
import tensorflow as tf

nesting_level = 0


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print("{}{}".format(space, entry))


def timeit(method):
    @functools.wraps(method)
    def timed(*args, **kw):
        global nesting_level

        class_name = ""
        if len(args) > 0 and hasattr(args[0], '__class__'):
            class_name = "{}.".format(args[0].__class__.__name__)
        log("Start [{}{}]:".format(class_name, method.__name__))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log("End [{}{}]. Time elapsed: {:0.2f} sec.".format(
            class_name,
            method.__name__,
            end_time - start_time
        ))
        return result

    return timed


def ohe2cat(label):
    return np.argmax(label, axis=1)


@timeit
def get_max_length(x, ratio=0.95):
    """
    Get the max length cover 95% data.
    """
    lens = [len(_) for _ in x]
    max_len = max(lens)
    min_len = min(lens)
    lens.sort()
    specified_len = lens[int(len(lens) * ratio)]
    log("Max length: {}; Min length {}; 95 length {}".format(max_len, min_len, specified_len))
    return specified_len


# @timeit
# def get_mean_length(x):
#     """
#     Get the mean length.
#     """
#     lens = [len(_) for _ in x]
#     max_len = max(lens)
#     min_len = min(lens)
#
#     mean_len = int(np.mean(lens))
#     log("Max length: {}; Min length {}; mean length {}".format(max_len, min_len, mean_len))
#     return mean_len


@timeit
def get_mean_length(x, sample_num=10000):
    """
    Get the mean length by sample num.
    """

    len_sum = 0
    shape = min(len(x), sample_num)
    for i in x[:shape]:
        len_sum += len(i)
    mean_len = len_sum // shape

    log("mean length {}".format(mean_len))
    return mean_len


def pad_seq(data, pad_len):
    return sequence.pad_sequences(data, maxlen=pad_len, dtype='float32', padding='post', truncating='post')


def set_random_seed_all(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)

# import logging
#
# formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
#
# def get_logger(name, level=logging.DEBUG):
#     logger = logging.getLogger(name)
#     logger.handlers.clear()
#     logger.setLevel(level)
#     ch = logging.StreamHandler()
#     ch.setLevel(level)
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)
#     return logger
#
#
# def add_filehandler(logger, filepath):
#     fh = logging.FileHandler(filepath)
#     fh.setLevel(logging.DEBUG)
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)
