import numpy as np
import tensorflow as tf
from collections import namedtuple

from hyper_search.agent import Agent
from hyper_search.controller import LSTMController
from tools import timeit, log

Checkpoint = namedtuple("Checkpoint", ["data", "score"])


class FeatureAgent(Agent):
    """

    基础版，只针对num类型，所以action一样长

    TODO: 根据feature_type构建
    """

    def __init__(self, num_feature, batch_count, meta_feature, **kwargs):
        super().__init__(**kwargs)
        self.num_feature = num_feature
        self.meta_feature = meta_feature
        self.max_order = 10
        # FIXME: 不定长
        self.num_op = 10
        self.num_unit = 256
        self.batch_count = batch_count
        self.num_action = num_feature * self.max_order
        self._best = Checkpoint(None, None)

    def build_graph(self):
        self._create_rnn()
        self._create_placeholder()
        self._create_loss()

    def _create_rnn(self):
        self._rnns = {}
        for i in range(self.num_feature):
            rnn_name = "rnn_%d" % i
            self._rnns[rnn_name] = LSTMController(
                num_state=self.meta_feature["max_order"],
                num_action=self.num_op,
                num_unit=self.num_unit,
                name=rnn_name
            )

    def _create_placeholder(self):
        self._actions = tf.placeholder(
            tf.int32,
            shape=(self.batch_count, self.num_action),
            name="batch_actions"
        )
        self._rewards = tf.placeholder(
            tf.int32,
            shape=(self.batch_count, self.num_action),
            name="batch_rewards"
        )

    def _create_loss(self):
        # loss =
        #   \avg_{0}^{batch_count}
        #       \sum_t -log(a_t) * reward_t
        #       + \sum_{i, t}{ a_{i, t} *  log(a_{i, t}) }
        #       + reg
        pass

    def decide(self, obs):
        pass

    def learn(self, *args):
        pass

    @property
    def best(self):
        return self._best.data


class NFSAgent(Agent):
    def __init__(self, num_batch, env):
        super(NFSAgent, self).__init__()
        self.num_feature = env.num_feature
        self.env = env
        self.num_op = env.num_op
        self.max_order = env.max_order
        self.num_batch = num_batch
        # discount factor
        self.alpha = 0.99
        # \lambda - return
        self.lambd = 0.5
        # regularization term
        self.reg = 1e-5

        self.num_action = self.num_feature * self.max_order
        self._env = env
        self._ctx = env.ctx

        log("Num feature({}), num op({})".format(self.num_feature, self.num_op))
        self.build_graph()

    def decide(self, obs):
        action_probs = self._ctx.sess.run(tf.nn.softmax(self.concat_output))
        return action_probs

    def learn(self, actions, rewards, *args):
        for i in range(self.num_action):
            base = rewards[:, i:]
            rewards_order = np.zeros_like(rewards[:, i], dtype=np.float)
            for j in range(base.shape[1]):
                order = j + 1
                base_order = base[:, 0: order]
                alphas = []
                for o in range(order):
                    alphas.append(pow(self.alpha, o))
                base_order = np.sum(base_order * alphas, axis=1)
                base_order *= np.power(self.lambd, j)
                rewards_order += base_order.astype(np.float)
            rewards[:, i] = (1 - self.lambd) * rewards_order
        self.update_policy({
            self.concat_action: np.reshape(actions, (self.num_batch, -1)),
            self.rewards: np.reshape(rewards, (self.num_batch, -1))
        }, self._ctx.sess)

    @timeit
    def _create_rnn(self):
        self.rnns = {}
        for i in range(self.num_feature):
            self.rnns['rnn%d' % i] = tf.contrib.rnn.BasicLSTMCell(
                num_units=self.num_op, name='rnn%d' % i)

    @timeit
    def _create_placeholder(self):
        self.concat_action = tf.placeholder(
            tf.int32,
            shape=[self.num_batch, self.num_action], name='concat_action'
        )

        self.rewards = tf.placeholder(
            tf.float32,
            shape=[self.num_batch, self.num_action], name='rewards'
        )

    @timeit
    def _create_variable(self):
        self.input0 = np.ones(shape=[self.num_feature, self.num_op], dtype=np.float32)
        self.input0 = self.input0 / self.num_op

    @timeit
    def _create_inference(self):
        self.outputs = {}

        for i in range(self.num_feature):
            tmp_h = self.rnns['rnn%d' % i].zero_state(1, tf.float32)
            tmp_input = tf.reshape(tf.nn.embedding_lookup(self.input0, i),
                                   [1, -1])
            for order in range(self.max_order):
                tmp_input, tmp_h = self.rnns['rnn%d' % i].__call__(tmp_input, tmp_h)
                if order == 0:
                    self.outputs['output%d' % i] = tmp_input
                else:
                    self.outputs['output%d' % i] = tf.concat(
                        [self.outputs['output%d' % i], tmp_input], axis=0)
        self.concat_output = tf.concat(list(self.outputs.values()), axis=0, name='concat_output')

    @timeit
    def _create_loss(self):
        self.loss = 0.0
        for batch_count in range(self.num_batch):
            action = tf.nn.embedding_lookup(self.concat_action, batch_count)
            reward = tf.nn.embedding_lookup(self.rewards, batch_count)
            action_index = tf.stack([list(range(self.num_action)), action], axis=1)
            action_probs = tf.squeeze(tf.nn.softmax(self.concat_output))
            pick_action_prob = tf.gather_nd(action_probs, action_index)
            loss_batch = tf.reduce_sum(-tf.log(pick_action_prob) * reward)
            loss_entropy = tf.reduce_sum(-action_probs * tf.log(action_probs)) * self.reg
            loss_reg = 0.0
            for i in range(self.num_feature):
                weights = self.rnns['rnn%d' % i].weights
                for w in weights:
                    loss_reg += self.reg * tf.reduce_sum(tf.square(w))
            self.loss += loss_batch + loss_entropy + loss_reg

        self.loss /= self.num_batch

    @timeit
    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    @timeit
    def build_graph(self):
        self._create_rnn()
        self._create_variable()
        self._create_placeholder()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        self._ctx.sess.run(init_op)

    def update_policy(self, feed_dict, sess=None):
        _, loss = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
        return loss

    @property
    def best(self):
        return None


class RandomAgent(Agent):

    def __init__(self, num_batch, env):
        super(RandomAgent, self).__init__()
        self.num_feature = env.num_feature
        self.env = env
        self.num_op = env.num_op
        self.max_order = env.max_order
        self.num_batch = num_batch
        self._env = env
        self._ctx = env.ctx

        log("Num feature({}), num op({})".format(self.num_feature, self.num_op))

    def decide(self, obs):
        # shape: (num_feature * max_order, num_op)
        action_prob = np.ones(
            shape=(self.num_feature * self.max_order, self.num_op),
            dtype=np.float
        )
        action_prob /= self.num_op
        return action_prob

    def learn(self, *args):
        return
