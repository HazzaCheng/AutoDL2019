import os
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.callbacks import Callback
import keras
import random
from collections import OrderedDict

from keras import backend as K

from models.speech.neural_model.resnet.data.data_utils_v2 import pre_trans_wav_update
from models.speech.neural_model.resnet.data.generator_v2 import DataGenerator
from .nn.module import vggvox_resnet2d_icassp
from .utils.autospeech_eda import AutoSpeechEDA
from .utils.autospeech_sampler import AutoSpSamplerNew


class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor="acc", baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print("Epoch %d: Reached baseline, terminating training" % (epoch))
                self.model.stop_training = True


class ThinRes34Config(object):
    ENABLE_CB_LRS = True
    ENABLE_CB_ES = True

    Epoch = 1
    VERBOSE = 2
    MAX_SEQ_NUM_CUT = 140000

    MAX_ROUND_NUM = 3000
    MAX_LEFT_TIME_BUD = 10
    TR34_INIT_WD = 1e-3
    INIT_BRZ_L_NUM = 124
    INIT_BRZ_L_NUM_WILD = 100
    PRED_SIZE = 8
    CLASS_NUM_THS = 37

    ENABLE_PRE_ENSE = True
    ENS_TOP_VLOSS_NUM = 8
    ENS_TOP_VACC_NUM = 0

    MAX_BATCHSIZE = 32
    FIRST_ROUND_EPOCH = 14
    LEFT_ROUND_EPOCH = 1

    SAMPL_PA_F_PERC_NUM = 10
    SAMPL_PA_F_MAX_NUM = 200
    SAMPL_PA_F_MIN_NUM = 200

    TR34_INIT_LR = 0.00175
    STEP_DE_LR = 0.002
    MAX_LR = 1e-3 * 1.5
    MIN_LR = 1e-4 * 5

    FULL_VAL_R_START = 2500
    G_VAL_CL_NUM = 3
    G_VAL_T_MAX_MUM = 0
    G_VAL_T_MIN_NUM = 0

    HIS_METRIC_SHOW_NUM = 10

    FE_RS_SPEC_LEN_CONFIG = {
        1: (125, 125, 0.002),
        5: (250, 250, 0.003),
        20: (1500, 1500, 0.004),
        50: (2250, 2250, 0.004),
    }

    FE_RS_SPEC_LEN_CONFIG_AGGR = {
        1: (100, 100, 0.002),
        10: (500, 500, 0.003),
        21: (1500, 1500, 0.004),
        50: (2250, 2250, 0.004),
    }

    FE_RS_SPEC_LEN_CONFIG_MILD = {
        1: (250, 250, 0.002),
        2: (500, 500, 0.004),
        3: (1000, 1000, 0.002),
        4: (1500, 1500, 0.004),
    }


def set_random_seed_all(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)


def set_mp(processes=4):
    import multiprocessing as mp

    def init_worker():
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    global pool
    try:
        pool.terminate()
    except BaseException:
        pass

    if processes:
        pool = mp.Pool(processes=processes, initializer=init_worker)
    else:
        pool = None
    return pool


def get_top_players(data, sort_keys, reverse=True, n=2, order=True):
    top = sorted(data.items(), key=lambda x: x[1][sort_keys], reverse=reverse)[:n]
    if order:
        return OrderedDict(top)
    return dict(top)


class Model(object):
    def __init__(self, metadata, is_multilabel, train_output_path="./", test_input_path="./"):
        """ Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 7,
             "train_num": 428,
             "test_num": 107,
             "time_budget": 1800}
        """
        # set_random_seed_all(0xC0FFEE)
        self.done_training = False
        self.metadata = metadata
        self._is_multilabel = is_multilabel
        self.train_num = self.metadata.get("train_num")
        self.class_num = self.metadata.get("class_num")
        self.tr34_mconfig = ThinRes34Config()
        self.train_output_path = train_output_path
        self.test_input_path = test_input_path
        self.batch_size = self.set_batch_size()

        self.params = {
            "dim": (257, 250, 1),
            "mp_pooler": set_mp(processes=4),
            "nfft": 512,
            "spec_len": 250,
            "win_length": 400,
            "hop_length": 160,
            "n_classes": self.class_num,
            "batch_size": self.batch_size,
            "shuffle": True,
            "normalize": True,
        }

        self.imp_feat_args = dict()
        self.spec_len_status = 0
        self.config = {
            "gpu": 1,
            "resume_pretrained": "pretrained_models/thin_resnet34.h5",
            "multiprocess": 4,
            "net": "resnet34s",
            "ghost_cluster": 2,
            "vlad_cluster": 8,
            "bottleneck_dim": 512,
            "aggregation_mode": "gvlad",
            "epochs": self.tr34_mconfig.Epoch,
            "lr": self.tr34_mconfig.TR34_INIT_LR,
            "warmup_ratio": 0.1,
            "loss": "softmax",
            "optimizer": "adam",
            "ohem_level": 0,
        }
        if self._is_multilabel:
            self.config["loss"] = "sigmoid"
        self.round_idx = 1
        self.test_idx = 1

        self.first_r_train_x = None
        self.first_r_train_y = None
        self.first_r_data_generator = None
        self.trn_gen = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.val_x = None
        self.g_cached_train_x = dict()
        self.g_train_y = None
        self.g_cached_x_sampldid_list = list()
        self.g_ori_train_x = None
        self.g_ori_train_y = None

        self.accpet_loss_list = [100]
        self.accpet_acc_list = [-1]
        self.g_train_loss_list = list()
        self.g_train_acc_list = list()
        self.g_val_loss_list = list()
        self.g_val_acc_list = list()
        self.g_his_eval_dict = dict()

        self.sampling_ratio_list = [1.0 / 8] * 1000
        self.best_test_predict_prob_list = dict()

        self.autospeech_eda = AutoSpeechEDA()

        self.autospeech_sampler = None

        self.model, self.callbacks = self.tr34_model_init()
        self.unfreeze = False
        self.fullvalid_stage = False
        self.loss_list_clear_switch = False

        self.round_spec_len = self.tr34_mconfig.FE_RS_SPEC_LEN_CONFIG

        self.last_y_pred = None
        self.last_y_pred_round = 0
        self.g_accept_cur_list = list()
        self.train_x_data_report = dict()
        self.test_x_data_report = dict()
        return

    def set_batch_size(self):
        bs = 32
        bs = min(bs, self.tr34_mconfig.MAX_BATCHSIZE)
        return bs

    def step_decay(self, epoch):
        """
        The learning rate begins at 10^initial_power,
        and decreases by a factor of 10 every step epochs.
        :param epoch:
        :return:
        """
        epoch = self.round_idx - 1
        stage1, stage2, stage3 = 10, 20, 40

        if epoch < self.tr34_mconfig.FULL_VAL_R_START:
            lr = self.tr34_mconfig.TR34_INIT_LR
        if epoch == self.tr34_mconfig.FULL_VAL_R_START:
            self.cur_lr = self.tr34_mconfig.STEP_DE_LR
            lr = self.cur_lr
        if epoch > self.tr34_mconfig.FULL_VAL_R_START:
            if self.g_accept_cur_list[-10:].count(False) == 10:
                self.cur_lr = self.tr34_mconfig.MAX_LR
            if self.g_accept_cur_list[-10:].count(True) >= 2:
                self.cur_lr = self.cur_lr * 1.05
            self.cur_lr = max(1e-4 * 3, self.cur_lr)
            self.cur_lr = min(1e-3 * 1.5, self.cur_lr)
            lr = self.cur_lr
        return np.float(lr)

    def tr34_model_init(self):
        def build_model(net):
            return vggvox_resnet2d_icassp(
                input_dim=self.params["dim"],
                num_class=self.params["n_classes"],
                mode="pretrain",
                config=self.config,
                net=net,
            )
        model_34 = build_model('resnet34s')
        model = model_34
        pretrain_path = os.path.join(os.path.dirname(__file__), self.config["resume_pretrained"])
        if self.config["resume_pretrained"]:
            if os.path.isfile(pretrain_path):
                model.load_weights(pretrain_path, by_name=True, skip_mismatch=True)
                if self.class_num >= self.tr34_mconfig.CLASS_NUM_THS:
                    frz_layer_num = self.tr34_mconfig.INIT_BRZ_L_NUM
                else:
                    frz_layer_num = self.tr34_mconfig.INIT_BRZ_L_NUM_WILD
                for layer in model.layers[: frz_layer_num]:
                    layer.trainable = False

            pretrain_output = model.output
            weight_decay = self.tr34_mconfig.TR34_INIT_WD
            if self._is_multilabel:
                output_activation = "sigmoid"
                loss_func = "binary_crossentropy"
            else:
                output_activation = "softmax"
                loss_func = "categorical_crossentropy"
            y = keras.layers.Dense(
                self.params["n_classes"],
                activation=output_activation,
                kernel_initializer="orthogonal",
                use_bias=False,
                trainable=True,
                kernel_regularizer=keras.regularizers.l2(weight_decay),
                bias_regularizer=keras.regularizers.l2(weight_decay),
                name="prediction",
            )(pretrain_output)
            model = keras.models.Model(model.input, y, name="vggvox_resnet2D_{}_{}_new".format(output_activation, "gvlad"))
            opt = keras.optimizers.Adam(lr=1e-3)
            model.compile(optimizer=opt, loss=loss_func, metrics=["acc"])

        model.summary()

        callbacks = list()
        if self.tr34_mconfig.ENABLE_CB_ES:
            early_stopping = EarlyStopping(monitor="val_loss", patience=15)
            callbacks.append(early_stopping)
        if self.tr34_mconfig.ENABLE_CB_LRS:
            normal_lr = LearningRateScheduler(self.step_decay)
            callbacks.append(normal_lr)
        return model, callbacks

    def set_g_valid_ratio(self):
        g_valid_num = self.class_num * self.tr34_mconfig.G_VAL_CL_NUM
        g_valid_num = min(g_valid_num, self.tr34_mconfig.G_VAL_T_MAX_MUM)
        g_valid_num = max(g_valid_num, self.tr34_mconfig.G_VAL_T_MIN_NUM)
        g_valid_ratio = g_valid_num / self.train_num
        return g_valid_ratio

    def train_first_eda_cut(self, train_x, train_y):
        self.train_x_data_report = self.autospeech_eda.get_x_data_report(x_data=train_x)
        return train_x, train_y

    def train_fit_first_by_generator(self):
        print("==== sample train {}".format(len(self.train_x)))
        print("train_x {}".format(np.asarray(self.train_x).shape))
        self.trn_gen = DataGenerator(self.train_x, self.train_y, **self.params)
        self.first_r_train_x = self.train_x
        self.first_r_data_generator = self.trn_gen
        # 决定跑多少个epoch
        cur_epoch = self.decide_epoch_curround()
        early_stopping = TerminateOnBaseline(monitor="acc", baseline=0.999)
        self.model.fit_generator(
            self.first_r_data_generator,
            steps_per_epoch=int(len(self.first_r_train_x) // self.params["batch_size"] // 2),
            epochs=cur_epoch,
            max_queue_size=10,
            callbacks=self.callbacks + [early_stopping],
            use_multiprocessing=False,
            workers=1,
            verbose=ThinRes34Config.VERBOSE)
        return

    def get_x_data_features_by_sampleidlist(self, cur_sampid_list):
        """
        Do features extraction and cached, get x_data array by cur_sampid_list.
        :param cur_sampid_list:
        :return: list of arrays.
        """
        if self.spec_len_status == 1:
            # 清空缓存
            self.g_cached_train_x.clear()
            self.g_cached_x_sampldid_list.clear()
            self.spec_len_status = 2
        cur_need_id_list = [sampleid for sampleid in cur_sampid_list if sampleid not in self.g_cached_x_sampldid_list]

        if len(cur_need_id_list) > 0:
            cur_need_origin_xdata = [self.g_ori_train_x[cur_sampid] for cur_sampid in cur_need_id_list]
            self.imp_feat_args["mode"] = "train"
            cur_need_x_data_list = pre_trans_wav_update(cur_need_origin_xdata, self.imp_feat_args)
            for i, cur_sampid in enumerate(cur_need_id_list):
                self.g_cached_x_sampldid_list.extend(cur_sampid_list)
                self.g_cached_x_sampldid_list = list(set(self.g_cached_x_sampldid_list))
                self.g_cached_train_x[cur_sampid] = cur_need_x_data_list[i]
        return [self.g_cached_train_x[cur_sampid] for cur_sampid in cur_sampid_list]

    def preprocess_and_train_first_round(self, train_dataset):
        train_x, train_y = train_dataset
        self.train_num = len(train_x)
        g_train_x, g_train_y = shuffle(train_x, train_y)
        g_valid_ratio = self.set_g_valid_ratio()
        n_valid = int(self.train_num * g_valid_ratio)
        valid_sample_id_list = random.sample(population=range(self.train_num), k=n_valid)
        train_sample_id_list = [sid for sid in range(self.train_num) if sid not in valid_sample_id_list]
        train_x = [g_train_x[s_id] for s_id in train_sample_id_list]
        train_y = g_train_y[train_sample_id_list]
        self.g_valid_x = [g_train_x[s_id] for s_id in valid_sample_id_list]
        self.g_valid_y = g_train_y[valid_sample_id_list]
        # 拿到数据集报告，不改变数据集
        self.g_ori_train_x, self.g_ori_train_y = self.train_first_eda_cut(train_x=train_x, train_y=train_y)
        # 根据类数改变SPEC_LEN
        self.choose_round_spec_len()
        # 根据轮数更新SPEC_LEN
        self.try_to_update_spec_len()
        # 采样
        self.autospeech_sampler = AutoSpSamplerNew(y_train_labels=self.g_ori_train_y)
        self.autospeech_sampler.set_up()
        self.g_train_y = train_y

        cur_round_sample_id_list = self.autospeech_sampler.get_downsample_index_list_by_class(
            per_class_num=self.tr34_mconfig.SAMPL_PA_F_PERC_NUM,
            max_sample_num=self.tr34_mconfig.SAMPL_PA_F_MAX_NUM,
            min_sample_num=self.tr34_mconfig.SAMPL_PA_F_MIN_NUM,
        )
        self.cur_sample_num = len(cur_round_sample_id_list)
        # 数据预处理，提特征
        self.train_x = self.get_x_data_features_by_sampleidlist(cur_sampid_list=cur_round_sample_id_list)
        self.train_y = self.g_train_y[cur_round_sample_id_list]
        self.train_fit_first_by_generator()
        self.round_idx += 1
        return True

    def try_to_update_spec_len(self):
        if self.round_idx in self.round_spec_len:
            train_spec_len, test_spec_len, suggest_lr = self.round_spec_len[self.round_idx]
            self.update_spec_len(train_spec_len, test_spec_len)

    def update_spec_len(self, train_spec_len, test_spec_len):
        self.imp_feat_args = {
            "train_spec_len": train_spec_len,
            "test_spec_len": test_spec_len,
            "train_wav_len": train_spec_len * 160,
            "test_wav_len": test_spec_len * 160,
            "mode": "train",
        }
        self.spec_len_status = 1
        print("===== update spec len {}".format(self.imp_feat_args))
        return True

    def decision_if_pre_ensemble(self):
        pre_ens_round_start = 35
        if self.round_idx > pre_ens_round_start:
            return True
        else:
            return False

    def train_bestmodel_decision(self):
        """
        Two stage: only train stage, valid stage.
        :param cur_round_loss:
        :return:

        """
        if self.round_idx < 10:
            return True
        cur_train_loss = self.g_train_loss_list[-1]
        pre_train_loss = min(self.g_train_loss_list[:-1])
        cur_train_acc = self.g_train_acc_list[-1]
        pre_train_acc = max(self.g_train_acc_list[:-1])
        if self.round_idx > 1:
            best_loss_rate_thres = 1.1
            best_acc_rate_thres = 0.9
        train_loss_ths = pre_train_loss * best_loss_rate_thres
        train_acc_ths = pre_train_acc * best_acc_rate_thres
        accept_train_loss = (cur_train_loss <= (train_loss_ths))
        accept_train_acc = (cur_train_acc >= (train_acc_ths))
        if not self.fullvalid_stage:
            accept_cur = accept_train_loss or accept_train_acc
            return accept_cur
        else:
            accept_valid_loss = self.g_val_loss_list[-1] <= min(self.g_val_loss_list) * 1.02
            accept_valid_acc = self.g_val_acc_list[-1] >= max(self.g_val_acc_list) * 0.99
            accept_cur = [accept_train_loss, accept_valid_acc, accept_valid_loss].count(True) >= 1
        return accept_cur

    def decide_epoch_curround(self):
        if self.round_idx == 1:
            cur_epoch_num = self.tr34_mconfig.FIRST_ROUND_EPOCH
        else:
            cur_epoch_num = self.tr34_mconfig.LEFT_ROUND_EPOCH
        return cur_epoch_num

    def train_left_fit_by_generator(self):
        print("==== sample train {}".format(len(self.train_x)))
        self.trn_gen = DataGenerator(self.train_x, self.train_y, **self.params)
        model = self.model
        accept_cur = False
        cur_epoch = self.decide_epoch_curround()
        if self.fullvalid_stage:
            m_history = model.fit_generator(
                self.trn_gen,
                steps_per_epoch=int(len(self.train_x) // self.params["batch_size"]),
                validation_data=self.g_valid_gen,
                epochs=cur_epoch,
                max_queue_size=10,
                callbacks=self.callbacks,
                use_multiprocessing=False,
                workers=1,
                verbose=ThinRes34Config.VERBOSE,
            )
            cur_valid_loss = round(m_history.history.get("val_loss")[-1], 6)
            cur_valid_acc = round(m_history.history.get("val_acc")[-1], 6)
            self.g_val_loss_list.append(cur_valid_loss)
            self.g_val_acc_list.append(cur_valid_acc)
        else:
            print("train_x {}".format(np.asarray(self.train_x).shape))
            self.trn_gen = DataGenerator(self.train_x, self.train_y, **self.params)
            m_history = model.fit_generator(
                self.trn_gen,
                steps_per_epoch=int(len(self.train_x) // self.params["batch_size"]),
                epochs=cur_epoch,
                max_queue_size=10,
                callbacks=self.callbacks,
                use_multiprocessing=False,
                workers=1,
                verbose=ThinRes34Config.VERBOSE,
            )
            cur_valid_loss = 100
            cur_valid_acc = -1
        cur_train_loss = round(m_history.history.get("loss")[-1], 6)
        cur_train_acc = round(m_history.history.get("acc")[-1], 6)
        cur_lr = m_history.history.get("lr")[-1]
        self.g_train_loss_list.append(cur_train_loss)
        self.g_train_acc_list.append(cur_train_acc)
        self.g_his_eval_dict[self.round_idx] = {
            "t_loss": cur_train_loss,
            "t_acc": cur_train_acc,
            "v_loss": cur_valid_loss,
            "v_acc": cur_valid_acc
        }
        accept_cur = self.train_bestmodel_decision()
        if self.fullvalid_stage:
            self.g_accept_cur_list.append(accept_cur)
        return model, accept_cur

    def decide_if_full_valid(self):
        if self.round_idx == self.tr34_mconfig.FULL_VAL_R_START:
            self.accpet_loss_list = [100]
            self.imp_feat_args["mode"] = "train"
            self.g_valid_x_features = pre_trans_wav_update(self.g_valid_x, self.imp_feat_args)
            self.g_valid_gen = DataGenerator(self.g_valid_x_features, self.g_valid_y, **self.params)
        if self.round_idx > self.tr34_mconfig.FULL_VAL_R_START:
            return True
        else:
            return False

    def train_left_rounds(self, remaining_time_budget):
        self.try_to_update_spec_len()
        model = self.model
        callbacks = self.callbacks

        self.fullvalid_stage = self.decide_if_full_valid()
        print("===== self.fullvalid_stage {}".format(self.fullvalid_stage))
        cur_round_sample_id_list = self.autospeech_sampler.get_downsample_index_list_by_class(
            per_class_num=10, max_sample_num=200, min_sample_num=300
        )

        self.train_x = self.get_x_data_features_by_sampleidlist(cur_sampid_list=cur_round_sample_id_list)
        self.train_y = self.g_train_y[cur_round_sample_id_list]
        model, accept_cur = self.train_left_fit_by_generator()
        self.round_idx += 1
        if accept_cur:
            self.model = model
            return True
        else:
            return False

    def test_pred_ensemble(self):
        """
        get best val loss round id list, best val acc round id list, and mean to prevent overfitting.
        :return:
        """
        key_t_loss = "t_loss"
        key_t_acc = "t_acc"
        key_loss = key_t_loss
        key_acc = key_t_acc

        topn_vloss = self.tr34_mconfig.ENS_TOP_VLOSS_NUM
        topn_vacc = self.tr34_mconfig.ENS_TOP_VACC_NUM
        pre_en_eval_rounds = list(self.best_test_predict_prob_list.keys())
        cur_eval_dict = {k: self.g_his_eval_dict.get(k) for k in pre_en_eval_rounds}

        top_n_val_loss_evals = get_top_players(data=cur_eval_dict, sort_keys=key_loss, n=topn_vloss,
                                               reverse=False)
        top_n_val_acc_evals = get_top_players(data=cur_eval_dict, sort_keys=key_acc, n=topn_vacc, reverse=True)
        top_n_val_loss_evals = list(top_n_val_loss_evals.items())
        top_n_val_acc_evals = list(top_n_val_acc_evals.items())
        topn_valloss_roundidx = [a[0] for a in top_n_val_loss_evals]
        topn_valacc_roundidx = [a[0] for a in top_n_val_acc_evals]

        merge_roundids = list()
        merge_roundids.extend(topn_valloss_roundidx)
        merge_roundids.extend(topn_valacc_roundidx)
        merge_preds_res = [self.best_test_predict_prob_list[roundid] for roundid in merge_roundids]
        return np.mean(merge_preds_res, axis=0)

    def choose_round_spec_len(self):
        if self.class_num >= 37:
            self.round_spec_len = self.tr34_mconfig.FE_RS_SPEC_LEN_CONFIG_AGGR
        else:
            self.round_spec_len = self.tr34_mconfig.FE_RS_SPEC_LEN_CONFIG_MILD
        return

    def update_round_spec_len_by_eda(self):
        std = self.train_x_data_report["x_seq_len_std"]
        mean = self.train_x_data_report["x_seq_len_mean"]
        rate = std / mean
        if rate < 0.1:
            target_len = mean
        else:
            target_len = self.train_x_data_report["x_seq_len_max"]

        precise_target_spec_len = (target_len / 160)
        target_spec_len = round(precise_target_spec_len / 250) * 250
        target_k = None
        rsl = self.round_spec_len
        rsl_list = [(k, rsl[k]) for k in sorted(rsl, key=rsl.get, reverse=False)]
        for k, paras in rsl_list:
            train_spec_len, test_spec_len, lr = paras
            if train_spec_len >= target_spec_len:
                target_k = k
                break
        if target_k:
            new_rsl = {k: v for k, v in rsl.items() if k <= target_k}
            new_rsl[target_k] = (target_spec_len, target_spec_len, 0.002)
            self.round_spec_len = new_rsl
        return

    def train(self, train_dataset, remaining_time_budget=None):
        """model training on train_dataset.

        :param train_dataset: tuple, (x_train, y_train)
            train_x: list of vectors, input train speech raw data.
            train_y: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
        """
        if self.done_training:
            return True
        K.set_learning_phase(1)
        if (
            remaining_time_budget < self.tr34_mconfig.MAX_LEFT_TIME_BUD
            or self.round_idx >= self.tr34_mconfig.MAX_ROUND_NUM
        ):
            self.done_training = True
            return True

        if self.round_idx == 1:
            self.preprocess_and_train_first_round(train_dataset)
        else:
            while not self.train_left_rounds(remaining_time_budget):
                continue

    def good_to_predict(self):
        flag = (
            (self.round_idx in self.round_spec_len)
            or (self.round_idx < 6)
            or (self.round_idx < 21 and self.round_idx % 2 == 1)
            or (self.round_idx - self.last_y_pred_round > 3)
        )
        return flag

    def test(self, test_x, is_val=False):
        """
        :param test_x: list of vectors, input test speech raw data.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        """
        K.set_learning_phase(0)
        model = self.model
        if is_val:
            if self.spec_len_status == 2:
                args = self.imp_feat_args
                args["mode"] = "test"
                test_x = pre_trans_wav_update(test_x, args)
                test_x = np.array(test_x)
                test_x = test_x[:, :, :, np.newaxis]
                self.val_x = test_x
            y_pred = model.predict(self.val_x, batch_size=self.tr34_mconfig.PRED_SIZE)
            return y_pred

        if self.test_idx == 1 or self.spec_len_status == 2:
            self.spec_len_status = 0
            self.imp_feat_args["mode"] = "test"
            test_x = pre_trans_wav_update(test_x, self.imp_feat_args)
            test_x = np.array(test_x)
            self.test_x = test_x[:, :, :, np.newaxis]

        if self.good_to_predict():
            y_pred = model.predict(self.test_x, batch_size=self.tr34_mconfig.PRED_SIZE)
            if self.tr34_mconfig.ENABLE_PRE_ENSE and self.decision_if_pre_ensemble():
                self.best_test_predict_prob_list[self.round_idx - 1] = y_pred
                y_pred = self.test_pred_ensemble()
            self.test_idx += 1
            self.last_y_pred = y_pred
            self.last_y_pred_round = self.round_idx
            return y_pred
        else:
            return self.last_y_pred
