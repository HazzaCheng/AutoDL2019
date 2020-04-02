import abc
from util.std_model import StandardModel


class Stage(StandardModel, metaclass=abc.ABCMeta):
    def __init__(self, context, **kwargs):
        super(Stage, self).__init__()
        self._context = context
        self._need_transition = False

    @property
    def ctx(self):
        return self._context

    @ctx.setter
    def ctx(self, value):
        self._context = value

    @abc.abstractmethod
    def generate_data(self):
        pass

    @abc.abstractmethod
    def transition(self):
        pass

    def train(self, dataset, remain_time_budget=None):
        if self.ctx.is_first_train:
            self.ctx.raw_dataset = dataset
        self.ctx.time_budget = remain_time_budget

    def test(self, dataset, remain_time_budget=None):
        if self.ctx.test_data is None:
            self.ctx.test_data = dataset
        if self._need_transition:
            self.transition()
