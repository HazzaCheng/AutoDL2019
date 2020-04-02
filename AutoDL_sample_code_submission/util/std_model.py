import abc


class StandardModel(metaclass=abc.ABCMeta):
    def __init__(self):
        self._model = None

    @abc.abstractmethod
    def train(self, dataset, remain_time_budget=None):
        pass

    @abc.abstractmethod
    def test(self, dataset, remain_time_budget=None):
        pass
