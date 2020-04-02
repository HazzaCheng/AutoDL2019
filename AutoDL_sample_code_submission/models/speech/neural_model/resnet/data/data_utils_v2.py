import librosa
import numpy as np
import multiprocessing
from multiprocessing import Pool
from itertools import repeat

NCPU = multiprocessing.cpu_count()


def extend_wav(wav, train_wav_len=40000, test_wav_len=40000, mode='train'):
    if mode == 'train':
        div, mod = divmod(train_wav_len, wav.shape[0])
        extended_wav = np.concatenate([wav]*div+[wav[:mod]])
        # 逆转
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        div, mod = divmod(test_wav_len, wav.shape[0])
        extended_wav = np.concatenate([wav]*div+[wav[:mod]])
        return extended_wav


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


def load_data(mag, train_spec_len=250, test_spec_len=250, mode='train'):
    freq, time = mag.shape
    if mode == 'train':
        if time-train_spec_len > 0:
            # 随机截取一段音频
            randtime = np.random.randint(0, time-train_spec_len)
            spec_mag = mag[:, randtime:randtime+train_spec_len]
        else:
            spec_mag = mag[:, :train_spec_len]
    else:
        # test数据不需要截取
        spec_mag = mag[:, :test_spec_len]
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)


def wav_to_mag(wav, params, win_length=400, hop_length=160, n_fft=512):
    mode = params["mode"]
    wav = extend_wav(wav, params["train_wav_len"], params["test_wav_len"], mode=mode)
    wav = np.asfortranarray(wav)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)
    mag_T = mag.T
    if mode == 'test':
        mag_T = load_data(mag_T, params["train_spec_len"], params["test_spec_len"], mode)
    return mag_T


def pre_trans_wav_update(wav_list, params):
    if len(wav_list) == 0:
        return []
    elif len(wav_list) > NCPU * 10:
        with Pool(NCPU) as pool:
            mag_arr = pool.starmap(wav_to_mag, zip(wav_list, repeat(params)))
            pool.close()
            pool.join()
            return mag_arr
    else:
        mag_arr = [wav_to_mag(wav, params) for wav in wav_list]
        return mag_arr

