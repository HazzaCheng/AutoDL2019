# coding:utf-8
import random
import numpy as np
from .autospeech_eda import AutoSpeechEDA, ohe2cat


class AutoSpSamplerNew(object):
    def __init__(self, y_train_labels):
        self.autosp_eda = AutoSpeechEDA()
        self.y_train_labels = y_train_labels
        self.y_train_cat_list = None
        self.y_class_num = None
        self.g_label_sample_id_list = list()

    def set_up(self):
        self.y_train_cat_list = ohe2cat(self.y_train_labels)
        y_labels_eda_report = self.autosp_eda.get_y_label_eda_report(y_onehot_labels=self.y_train_labels)
        self.y_class_num = y_labels_eda_report.get("y_class_num")
        for y_label_id in range(self.y_class_num):
            label_sample_id_list = list(np.where(self.y_train_cat_list == y_label_id)[0])
            self.g_label_sample_id_list.append(label_sample_id_list)

    def get_downsample_index_list_by_class(self, per_class_num, max_sample_num, min_sample_num):
        # 没有做类别平衡
        train_data_sample_id_list = list()
        min_sample_perclass = int(min_sample_num / self.y_class_num)
        for y_label_id in range(self.y_class_num):
            random_sample_k = per_class_num
            random_sample_k = max(min_sample_perclass, random_sample_k)
            label_sample_id_list = self.g_label_sample_id_list[y_label_id]
            if len(label_sample_id_list) > random_sample_k:
                downsampling_label_sample_id_list = random.sample(population=label_sample_id_list, k=random_sample_k)
                train_data_sample_id_list.extend(downsampling_label_sample_id_list)
            else:
                train_data_sample_id_list.extend(label_sample_id_list)

        if len(train_data_sample_id_list) > max_sample_num:
            train_data_sample_id_list = random.sample(population=train_data_sample_id_list, k=max_sample_num)
        return train_data_sample_id_list


if __name__ == '__main__':
    pass
