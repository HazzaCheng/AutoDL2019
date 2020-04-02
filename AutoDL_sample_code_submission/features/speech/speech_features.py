#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-22
import os
from functools import partial
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool

import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

from tools import timeit, pad_seq

NCPU = os.cpu_count() - 1
pool = ThreadPool(NCPU)

# ---------------- constants -----------------------
MFCC = 'mfcc'
ZERO_CROSSING_RATE = 'zero crossing rate'
SPECTRAL_CENTROID = 'spectral centroid'
MEL_SPECTROGRAM = 'mel spectrogram'
SPECTRAL_ROLLOFF = 'spectral rolloff'
CHROMA_STFT = 'chroma stft'
BANDWIDTH = 'bandwidth'
SPECTRAL_CONTRAST = 'spectral_contrast'
SPECTRAL_FLATNESS = 'spectral flatness'
TONNETZ = 'tonnetz'
CHROMA_CENS = 'chroma cens'
RMS = 'rms'
POLY_FEATURES = 'poly features'

NUM_MFCC = 96  # num of mfcc features, default value is 24
MAX_AUDIO_DURATION = 5  # limited length of audio, like 20s
AUDIO_SAMPLE_RATE = 16000
MAX_FRAME_NUM = 700
FFT_DURATION = 0.1
HOP_DURATION = 0.04

SPEECH_FEATURES = [MFCC,
                   ZERO_CROSSING_RATE,
                   SPECTRAL_CENTROID,
                   MEL_SPECTROGRAM,
                   SPECTRAL_ROLLOFF,
                   CHROMA_STFT,
                   BANDWIDTH,
                   SPECTRAL_CONTRAST,
                   SPECTRAL_FLATNESS,
                   TONNETZ,
                   CHROMA_CENS,
                   RMS,
                   POLY_FEATURES]

# ---------------- parallel extract features -----------------------


def extract_parallel(data, extract):
    data_with_index = list(zip(data, range(len(data))))
    results_with_index = list(pool.map(extract, data_with_index))

    results_with_index.sort(key=lambda x: x[1])

    results = []
    for res, idx in results_with_index:
        results.append(res)

    return np.asarray(results)


# mfcc
@timeit
def extract_mfcc(data, sr=16000, n_mfcc=NUM_MFCC):
    results = []
    for d in data:
        r = librosa.feature.mfcc(d, sr=sr, n_mfcc=n_mfcc)
        r = r.transpose()
        results.append(r)

    return results


def extract_for_one_sample(tuple, extract, use_power_db=False, **kwargs):
    data, idx = tuple
    r = extract(data, **kwargs)
    # for melspectrogram
    if use_power_db:
        r = librosa.power_to_db(r)

    r = r.transpose()
    return r, idx


@timeit
def extract_mfcc_parallel(data, sr=16000, n_fft=None, hop_length=None, n_mfcc=NUM_MFCC):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.mfcc, sr=sr,
                      n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    results = extract_parallel(data, extract)

    return results


# zero crossings

@timeit
def extract_zero_crossing_rate_parallel(data):
    extract = partial(extract_for_one_sample, extract=librosa.feature.zero_crossing_rate, pad=False)
    results = extract_parallel(data, extract)

    return results


# spectral centroid

@timeit
def extract_spectral_centroid_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_centroid, sr=sr,
                      n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_melspectrogram_parallel(data, sr=16000, n_fft=None, hop_length=None, n_mels=40, use_power_db=False):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.melspectrogram,
                      sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, use_power_db=use_power_db)
    print("===== data {}".format(type(data)))
    results = extract_parallel(data, extract)

    return results


# spectral rolloff
@timeit
def extract_spectral_rolloff_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_rolloff,
                      sr=sr, n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)  # data+0.01?
    # sklearn.preprocessing.scale()
    return results


# chroma stft
@timeit
def extract_chroma_stft_parallel(data, sr=16000, n_fft=None, hop_length=None, n_chroma=12):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.chroma_stft, sr=sr,
                      n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_bandwidth_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_bandwidth,
                      sr=sr, n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_spectral_contrast_parallel(data, sr=16000, n_fft=None, hop_length=None, n_bands=6):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_contrast,
                      sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_spectral_flatness_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_flatness,
                      n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_tonnetz_parallel(data, sr=16000):
    extract = partial(extract_for_one_sample, extract=librosa.feature.tonnetz, sr=sr)
    results = extract_parallel(data, extract)
    return results


@timeit
def extract_chroma_cens_parallel(data, sr=16000, hop_length=None, n_chroma=12):
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.chroma_cens, sr=sr,
                      hop_length=hop_length, n_chroma=n_chroma)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_rms_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.rms,
                      frame_length=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_poly_features_parallel(data, sr=16000, n_fft=None, hop_length=None, order=1):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.poly_features,
                      sr=sr, n_fft=n_fft, hop_length=hop_length, order=order)
    results = extract_parallel(data, extract)

    return results


# stft
def get_specified_feature_func(feature_name,
                               sr=16000,
                               n_fft=None,
                               hop_length=None,
                               n_mfcc=96,
                               n_mels=40,
                               use_power_db=True,
                               n_chroma=12,
                               n_bands=6,
                               order=1):
    if feature_name == MFCC:
        func = partial(extract_mfcc_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    elif feature_name == ZERO_CROSSING_RATE:
        func = partial(extract_zero_crossing_rate_parallel)
    elif feature_name == SPECTRAL_CENTROID:
        func = partial(extract_spectral_centroid_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length)
    elif feature_name == MEL_SPECTROGRAM:
        func = partial(extract_melspectrogram_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, use_power_db=use_power_db)
    elif feature_name == SPECTRAL_ROLLOFF:
        func = partial(extract_spectral_rolloff_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length)
    elif feature_name == CHROMA_STFT:
        func = partial(extract_chroma_stft_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma)
    elif feature_name == BANDWIDTH:
        func = partial(extract_bandwidth_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length)
    elif feature_name == SPECTRAL_CONTRAST:
        func = partial(extract_spectral_contrast_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands)
    elif feature_name == SPECTRAL_FLATNESS:
        func = partial(extract_spectral_flatness_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length)
    elif feature_name == TONNETZ:
        func = partial(extract_tonnetz_parallel, sr=sr)
    elif feature_name == CHROMA_CENS:
        func = partial(extract_chroma_cens_parallel, sr=sr, hop_length=hop_length, n_chroma=n_chroma)
    elif feature_name == RMS:
        func = partial(extract_rms_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length)
    elif feature_name == POLY_FEATURES:
        func = partial(extract_poly_features_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length, order=order)
    else:
        raise Exception("No such feature {}".format(feature_name))

    return func


def get_features_data(x, feature_func, model_kind,
                      split_length=None, feature_length=None):
    if model_kind == 0:
        x = [sample[:split_length] for sample in x]
        x = feature_func(x)
        x_feas = []
        for i in range(len(x)):
            fea = np.mean(x[i], axis=0).reshape(-1)
            fea_std = np.std(x[i], axis=0).reshape(-1)
            x_feas.append(np.concatenate([fea, fea_std], axis=-1))
        x_feas = np.asarray(x_feas, dtype=np.float32)
        scaler = StandardScaler()
        x = scaler.fit_transform(x_feas[:, :])
    elif model_kind == 1:
        x = [sample[:split_length] for sample in x]
        x = feature_func(x)
        x = pad_seq(x, pad_len=feature_length)

    return x