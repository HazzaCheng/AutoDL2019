import numpy as np


def batch_split(batch):
    batch_size = batch.shape[0]
    array = [item[0] for item in np.split(batch, batch_size, axis=0)]
    return array


def flatten(arrays):
    if len(arrays) == 1:
        return arrays[0]
    arrays[0].extend(flatten(arrays[1:]))
    return arrays[0]


def round_pow2(val):
    res = 1
    while res * 2 < val:
        res <<= 1
    if res * 2 - val < val - res:
        res <<= 1
    return res
