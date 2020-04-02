import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, LSTM


class LSTMController:
    def __init__(self,
                 num_state,
                 num_action,
                 num_unit,
                 name):
        self.num_state = num_state
        self.num_action = num_action
        self.num_unit = num_unit
        self.name = name
        self._build_graph()

    def _build_graph(self):
        self._create_placeholder()
        self._create_inference()

    def _create_placeholder(self):
        self._state = tf.placeholder(
            tf.int32,
            shape=(1, self.num_state),
            name="%s_state" % self.name
        )

    def _create_inference(self):
        embedding = Embedding(
            (-1, self.num_state),
            (-1, self.num_unit),
            mask_zero=True,
            name='%s_embedding' % self.name
        )(self._state)
        # TODO: maybe softmax?
        self._logits = LSTM(
            self.num_unit,
            name="%s_lstm_cell" % self.name
        )(embedding)

    @property
    def input(self):
        return self._state

    @property
    def output(self):
        return self._logits
