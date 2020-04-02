import abc

from tools import log, timeit


class Environment(metaclass=abc.ABCMeta):
    @timeit
    def __init__(self, **kwargs):
        self._action_space = None
        self._observation_space = None
        log("Env kwargs: {}".format(kwargs))

    @abc.abstractmethod
    def step(self, action_idx):
        """

        Take an action when environment in current state (no need to pass state as a parameter,
        the class instance will remember the state), update the state, get immediate reward

        Parameters
        ----------
        action_idx

        Returns
        -------
        next_state
        immediate reward
        done: True means current episode ends.

        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        When one episode finished, reset the environment to initial state.
        Returns
        -------

        """
        pass

    @property
    def action_space(self):
        return self._action_space

    @property
    def obs_space(self):
        return self._observation_space
