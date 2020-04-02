# coding:utf-8
import os
import numpy as np
import sys


def ohe2cat(label):
    return np.argmax(label, axis=1)


class AutoSpeechEDA(object):
    def __init__(self, data_expe_flag=False, x_train=None, y_train=None):
        self.train_data_sinffer_num_per_class = 50
        self.expe_x_data_num = 3
        self.expe_y_labels_num = 3
        self.data_expe_flag = data_expe_flag

    def get_y_label_distribution_by_bincount(self, y_onehot_labels):
        y_sample_num, y_label_num = y_onehot_labels.shape
        y_as_label = ohe2cat(y_onehot_labels)
        y_as_label_bincount = np.bincount(y_as_label)
        y_label_distribution_array = y_as_label_bincount / y_sample_num
        y_label_distribution_array = list(y_label_distribution_array)
        return y_label_distribution_array

    def get_y_label_eda_report(self, y_onehot_labels):
        y_train_len_num = len(y_onehot_labels)
        if self.data_expe_flag:
            expe_x_data_list = [a_x_data.tolist() for a_x_data in y_onehot_labels[:self.expe_x_data_num]]
        y_sample_num, y_label_num = y_onehot_labels.shape
        y_label_distribution_array = self.get_y_label_distribution_by_bincount(y_onehot_labels=y_onehot_labels)
        eda_y_report = dict()
        eda_y_report["y_sample_num"] = int(y_sample_num)
        eda_y_report["y_class_num"] = int(y_label_num)
        eda_y_report["y_label_distribution_array"] = y_label_distribution_array
        return eda_y_report

    def get_x_data_report(self, x_data):
        x_sample_num = len(x_data)
        if self.data_expe_flag:
            expe_x_data_list = [a_x_data.tolist() for a_x_data in x_data[:self.expe_x_data_num]]
        x_train_word_len_list = list()
        for x_train_sample in x_data:
            len_a_x_sample = x_train_sample.shape[0]
            x_train_word_len_list.append(len_a_x_sample)
        x_train_word_len_array = np.array(x_train_word_len_list)
        x_train_sample_mean = x_train_word_len_array.mean()
        x_train_sample_std = x_train_word_len_array.std()

        eda_x_data_report = dict()
        eda_x_data_report["x_total_seq_num"] = int(x_train_word_len_array.sum())
        eda_x_data_report["x_seq_len_mean"] = int(x_train_sample_mean)
        eda_x_data_report["x_seq_len_std"] = x_train_sample_std
        eda_x_data_report["x_seq_len_max"] = int(x_train_word_len_array.max())
        eda_x_data_report["x_seq_len_min"] = int(x_train_word_len_array.min())
        eda_x_data_report["x_seq_len_median"] = int(np.median(x_train_word_len_array))
        eda_x_data_report["x_sample_num"] = int(x_sample_num)
        return eda_x_data_report


def main():
    pass

if __name__ == '__main__':
    main()
