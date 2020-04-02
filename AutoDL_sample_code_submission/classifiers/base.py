import abc


class Classifier(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def fit(self, dataset, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, data, *args, **kwargs):
        pass

    @staticmethod
    def hyper_params_search(dataset, *args, **kwargs):
        pass
