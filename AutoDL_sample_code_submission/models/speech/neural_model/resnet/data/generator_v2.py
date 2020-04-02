import keras
import numpy as np

from models.speech.neural_model.resnet.data.data_utils_v2 import load_data


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, labels, dim, mp_pooler, augmentation=True, batch_size=32, nfft=512, spec_len=250,
                 win_length=400, hop_length=160, n_classes=5994, shuffle=True, normalize=True):
        'Initialization'
        self.dim = dim
        self.nfft = nfft
        self.spec_len = spec_len
        self.normalize = normalize
        self.mp_pooler = mp_pooler
        self.win_length = win_length
        self.hop_length = hop_length

        self.labels = labels
        self.shuffle = shuffle
        self.X = X
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X_temp = [self.X[k] for k in indexes]
        X, y = self.__data_generation_mp(X_temp, indexes)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation_mp(self, X_temp, indexes):
        # load data 随机截取一段音频
        X = [self.mp_pooler.apply_async(load_data, args=(mag, self.spec_len)) for mag in X_temp]
        X = np.expand_dims(np.array([p.get() for p in X]), -1)
        y = self.labels[indexes]
        return X, y
