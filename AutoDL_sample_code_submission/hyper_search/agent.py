import abc

from tools import log, timeit


class Agent(metaclass=abc.ABCMeta):

    @timeit
    def __init__(self, **kwargs):
        log("Agent kwargs:")
        log("{}".format(kwargs))

    @abc.abstractmethod
    def decide(self, obs):
        pass

    @abc.abstractmethod
    def learn(self, *args):
        pass
