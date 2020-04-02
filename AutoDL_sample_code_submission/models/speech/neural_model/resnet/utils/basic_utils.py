# -*- encoding: utf-8 -*-

import numpy as np


def ohe2cat(label):
    return np.argmax(label, axis=1)
