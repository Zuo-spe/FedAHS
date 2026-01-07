
import os
from collections import Counter
from SDUS_SPN import SDUS_SPN
import numpy as np

class SDUS1:
    def __init__(self,sampling_rate=1.0):
        self.X = None
        self.y = None
        self.Sinfor = None  # SPN information
        self.S_maj = []  # Indicates an SPN that needs to be independently selected
        self.S_rest = []  # Indicates an SPN that needs to be aggregated for selection
        self.maj_num = None
        self.min_num = None
        self.ID = []  # Sample index of majority to be selected
        self.sampling_rate = sampling_rate  # 采样率参数

    def distance(self, ma, mb):
        l = len(ma)
        abt = np.matrix(ma) * np.matrix(mb.T)
        sqa = np.tile(np.sum(ma ** 2, axis=1), l).reshape((l, l)).T
        sqb = np.tile(np.sum(mb ** 2, axis=1), l).reshape((l, l))
        dis_matrix = np.sqrt(sqa + sqb - 2 * abt + 0.0001)
        return np.array(dis_matrix)

    def weight_norm(self, matrix):
        sum_ = np.sum(matrix)
        matrix = list([x / sum_ for x in matrix])
        return matrix

    def weighted_under_sampling(self, id, weight, num):
        select_index = np.random.choice(id, size=num, replace=True, p=weight)
        return select_index

    def fit(self, X, y):
        self.X = X
        self.y = y
        dic = Counter(self.y)
        spn = SDUS_SPN()
        self.Sinfor = spn.fit(self.X, self.y)
        self.maj_num, self.min_num = dic[0], dic[1]
        #计算需要调整多数类数量
        adjusted_maj_num = int(self.sampling_rate * self.maj_num)

        # Determine how many of each SPN should be sampled, if greater than or equal to 1,
        # then sample independently, otherwise aggregate and sample
        for S in self.Sinfor:
            if (len(S['id']) / self.maj_num) * self.min_num >= 1:
                self.S_maj.append(S)
            else:
                self.S_rest.append(S)

        # SPN内是否存在要采样的样本数大于或等于1的SPN
        if len(self.S_maj) != 0:
            print("A")
            select_num = 0
            for S in self.S_maj:
                id_num = len(S['id'])
                sel_num = round(id_num / adjusted_maj_num * self.min_num)
                #sel_num = int(self.maj_num - adjusted_maj_num)
                select_num = select_num + sel_num
                #select_num = int(self.maj_num - adjusted_maj_num)

                sample_set = self.X[S['id']]
                dis_matrix = self.distance(sample_set, sample_set)
                sum_matrix = np.sum(dis_matrix, axis=1)
                norm_matrix = self.weight_norm(sum_matrix)
                weights = norm_matrix
                sampled = self.weighted_under_sampling(S['id'], weights, sel_num)
                for temp_sampled_index in sampled:
                    self.ID.append(temp_sampled_index)

            # [处理有SPNs需要采样超过1个样本的情况的代码]
            if select_num < self.min_num:
                if len(self.S_rest) != 0:
                    sample_set_rest_index = []
                    for S in self.S_rest:
                        for index in S['id']:
                            sample_set_rest_index.append(index)

                    sample_set_rest = self.X[sample_set_rest_index]
                    select_num_rest = round(len(sample_set_rest) / self.maj_num * self.min_num)
                    dis_matrix_rest = self.distance(sample_set_rest, sample_set_rest)
                    sum_matrix_rest = np.sum(dis_matrix_rest, axis=1)
                    sum_matrix_rest = 1 / sum_matrix_rest
                    sum_matrix_rest = self.weight_norm(sum_matrix_rest)

                    weights = sum_matrix_rest
                    sampled = self.weighted_under_sampling(sample_set_rest_index, weights, select_num_rest)
                    for temp_sampled_index in sampled:
                        self.ID.append(temp_sampled_index)

            re_maj = np.c_[self.X[np.array(self.ID)], np.zeros(len(self.ID))]  # 多数类特征和标签 (0)
            re_min = np.c_[self.X[np.where(self.y == 1)[0]], np.ones(self.min_num)]  # 少数类特征和标签 (1)
            resample = np.r_[re_maj, re_min]
            resampled_features = resample[:, :-1]  # 所有行，除了最后一列
            resampled_labels = resample[:, -1]  # 所有行，只有最后一列
            return resampled_features, resampled_labels

        else:  # [处理没有SPNs需要采样超过1个样本的情况的代码]
            sample_set_index = []
            for S in self.S_rest:
                for index in S['id']:
                    sample_set_index.append(index)
            sample_set = self.X[sample_set_index]
            if self.sampling_rate == 1:
                select_num = self.min_num
            else:
                select_num = int(self.maj_num - adjusted_maj_num)
            dis_matrix = self.distance(sample_set, sample_set)
            sum_matrix = np.sum(dis_matrix, axis=1)
            sum_matrix = 1 / sum_matrix
            sum_matrix = self.weight_norm(sum_matrix)
            weights = sum_matrix
            sampled = self.weighted_under_sampling(sample_set_index, weights, select_num)
            for temp_sampled_index in sampled:
                self.ID.append(temp_sampled_index)

            re_maj = np.c_[self.X[np.array(self.ID)], np.zeros(len(self.ID))]  # 多数类特征和标签 (0)
            re_min = np.c_[self.X[np.where(self.y == 1)[0]], np.ones(self.min_num)]  # 少数类特征和标签 (1)
            resample = np.r_[re_maj, re_min]
            resampled_features = resample[:, :-1]  # 所有行，除了最后一列
            resampled_labels = resample[:, -1]  # 所有行，只有最后一列
            return resampled_features, resampled_labels


