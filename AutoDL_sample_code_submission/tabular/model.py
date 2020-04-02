import globals
import numpy as np
from tools import log, timeit
from util.context import Context
from util.std_model import StandardModel
from tabular.stage import RetrieveStage


class TabularModel(StandardModel):

    def __init__(self, metadata, sess):
        super().__init__()
        # global global_ctx
        self._metadata = metadata
        log("{}\n{}".format(metadata.metadata_, metadata.get_dataset_name()))
        self._context = Context(sess, metadata)
        globals.put("ctx", self._context)
        self._context.stage = RetrieveStage(
            batch_size=5120,
            context=self._context,
        )
        self._y_hats = []

    @timeit
    def train(self, dataset, remain_time_budget=None):
        self._context.stage.train(dataset, remain_time_budget)

    @timeit
    def test(self, dataset, remain_time_budget=None):
        result = self._context.stage.test(dataset, remain_time_budget)
        if self._context.is_ensemble:
            log("Start ensemble")
            self._y_hats.append(result)
            return np.asarray(self._y_hats).mean(axis=0)
        else:
            return result
