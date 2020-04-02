import copy
import multiprocessing as mp
import numpy as np
from sklearn.model_selection import train_test_split

import globals
from hyper_search.env import Environment
from tabular.feature_engineer.op import Op
from metrics import auc_metric
from tools import log, timeit


def feature_transform(col, transformers):
    pass


class TabularEnv(Environment):
    """Environment of tabular data

    Attributes:
        x: num_instance x num_feature
        col_types: {col_index -> col_types}
        trajectory: [state, action, reward]
        cur_state: t x num_feature x 1,
        state: 1 x num_feature, 每个特征的上一次转换，通过LSTM存储每个特征的历史信息
        action: 1 x num_feature, 每个特征进行一次transform
        reward: V_{t} - V_{t-1}
    """

    def __init__(
            self,
            dataset,
            col_types,
            transformers,
            max_order,
            ctx,
            cls_class,
            cls_args,
            cls_kwargs,
            **kwargs):
        super().__init__(**kwargs)
        self._x, self._y = dataset
        self._col_num = self._x.shape[1]
        self._origin_cols = [self._x[:, i] for i in range(self._col_num)]
        self._col_types = col_types
        self._trajectory = []
        # translate binary op to n unary op
        self._action_space = transformers
        self._mp_pool = mp.Pool()
        self._max_order = max_order
        self._cur_state = None
        self._cur_cols = None

    def step(self, actions):
        self._cur_state.append(actions)
        next_state, reward, done = self._cur_state, 0, self._cur_order == self._max_order
        # TODO: 根据action构造新特征，训练并进行评价
        new_cols = self._mp_pool.map(
            feature_transform,
            [(self._origin_cols[i], actions[i]) for i in range(self._col_num)]
        )
        return next_state, reward, done

    def reset(self):
        self._trajectory.clear()
        self._cur_state = [np.zeros((1, self._col_num), dtype=np.int)]
        self._cur_cols = copy.deepcopy(self._origin_cols)
        return self._cur_state

    @property
    def _cur_order(self):
        return len(self._cur_state[0])


def get_rewards(actions, x, y, kwargs):
    num_feature = x.shape[1]
    copies = execute_actions(actions, x, **kwargs)
    action_per_feature = int(len(actions) / num_feature)
    rewards = evaluate_actions(copies, globals.get("origin_result"), action_per_feature, x, y, **kwargs)
    return rewards


def execute_actions(actions, x, **kwargs):
    from sklearn.preprocessing import MinMaxScaler
    num_feature = x.shape[1]
    action_per_feature = int(len(actions) / num_feature)
    copies = {}

    for feature_count in range(num_feature):
        # feature_name = X.columns[feature_count]
        feature_actions = actions[
                          feature_count * action_per_feature:
                          (feature_count + 1) * action_per_feature
                          ]
        copies[feature_count] = []
        if feature_actions[0] == 0:
            continue
        else:
            copy = np.asarray(x[:, feature_count], dtype=np.float)

        for action in feature_actions:
            if action == 0:
                break
            elif action > 0 and action <= 4:
                action_unary = action - 1
                if action_unary == 0:
                    copy = np.squeeze(np.sqrt(abs(copy)))
                elif action_unary == 1:
                    scaler = MinMaxScaler()
                    copy = np.squeeze(scaler.fit_transform(np.reshape(copy, [-1, 1])))
                elif action_unary == 2:
                    while (np.any(copy == 0)):
                        copy = copy + 1e-5
                    copy = np.squeeze(np.log(abs(np.array(copy))))
                elif action_unary == 3:
                    while (np.any(copy == 0)):
                        copy = copy + 1e-5
                    copy = np.squeeze(1 / (np.array(copy)))

            else:
                num_op_unary = 4
                action_binary = (action - num_op_unary - 1) // (num_feature - 1)
                rank = np.mod(action - num_op_unary - 1, num_feature - 1)

                if rank >= feature_count:
                    rank += 1

                target = np.array(x[:, rank])

                if action_binary == 0:
                    copy = np.squeeze(copy + target)
                elif action_binary == 1:
                    copy = np.squeeze(copy - target)
                elif action_binary == 2:
                    copy = np.squeeze(copy * target)
                elif action_binary == 3:
                    while (np.any(target == 0)):
                        target = target + 1e-5
                    copy = np.squeeze(copy / target)
                # elif action_binary == 4:
                #     copy = np.squeeze(mod_column(copy, X[target_feature_name].values))

            copies[feature_count].append(copy)
    return copies


def evaluate_actions(copies, origin_result, action_per_feature, x, y, **kwargs):
    rewards = []
    former_result = origin_result
    former_copys = [None]
    for key in sorted(copies.keys()):
        reward, former_result, return_copy = get_reward_per_feature(
            copies[key], action_per_feature, former_result, x, y, former_copys, **kwargs)
        former_copys.append(return_copy)
        rewards += reward
    return rewards


def get_reward_per_feature(copies, count, former_result, x, y, former_copys=None, **kwargs):
    former_copys = [None] if former_copys is None else former_copys
    reward = []
    previous_result = former_result
    for i, former_copy in enumerate(former_copys):
        if former_copy is not None:
            x = np.concatenate([x, former_copy.reshape(-1, 1)], axis=1)

    for copy in copies:
        x_extend = np.concatenate([x, copy.reshape(-1, 1)], axis=1)
        current_result = evaluate(x_extend, y, **kwargs)
        reward.append(current_result - previous_result)
        previous_result = current_result

    reward_till_now = len(reward)
    for _ in range(count - reward_till_now):
        reward.append(0)
    if len(copies) == 0:
        return_copy = None
    else:
        return_copy = copies[-1]

    return reward, previous_result, return_copy


def evaluate(x, y, cls_kwargs=None, **kwargs):
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y,
        test_size=0.2
    )
    if cls_kwargs is None:
        cls_kwargs = globals.get("cls_kwargs")
    model = globals.get("cls_class")(
        *globals.get("cls_args"),
        **cls_kwargs
    )
    model.fit((x_train, y_train))
    y_valid_hat = model.predict(x_valid)
    auc = auc_metric(y_valid, y_valid_hat)
    return auc


class NFSEnv(Environment):

    def __init__(
            self,
            dataset,
            ctx,
            max_order=5,
            num_op_unary=4, num_op_binary=4,
            **kwargs):
        super().__init__(**kwargs)
        self.ctx = ctx
        # fixed in get_reward
        self.dataset = dataset
        self.num_feature = dataset[0].shape[1]
        self.num_op_unary = num_op_unary
        self.num_op_binary = num_op_binary
        self.num_op = self.num_op_unary + (self.num_feature - 1) * self.num_op_binary + 1
        self.max_order = max_order

        # tuple(actions, auc)
        self.best = None

    def evaluate(self, x, y):
        return evaluate(x, y, cls_kwargs=globals.get("cls_kwargs"))

    def translate(self, x):
        actions, _ = self.best
        copies = execute_actions(actions, x)
        new_cols = []
        for _, col in copies.items():
            if len(col) > 0:
                new_cols.append(col[-1].reshape(-1, 1))
        x_final = np.concatenate(new_cols, axis=1)
        return x_final

    # @timeit
    # def step(self, actions):
    #     length = len(actions)
    #     x, y = self.dataset
    #     log("{}".format(globals.get("cls_kwargs")))
    #     map_args = zip(
    #         actions,
    #         [x] * length,
    #         [y] * length,
    #         [{
    #             "cls_kwargs": globals.get("cls_kwargs")
    #           }] * length
    #     )
    #     rewards = np.asarray(self.ctx.mp.starmap(get_rewards, map_args), dtype=np.float)
    #     increased_auc = np.sum(rewards, axis=1)
    #     best_i = np.argmax(increased_auc)
    #
    #     best = (actions[best_i], increased_auc[best_i])
    #     self.best = best
    #     # if self.best is None or best[1] > self.best[1]:
    #     #     self.best = best
    #     log("rewards {}, best auc {}".format(rewards.shape, self.best[1]))
    #     return None, rewards, True

    def step(self, actions):
        self.best = (actions[-1], 0.0)
        return None, None, None

    def reset(self):
        pass
