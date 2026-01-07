'''
调试完成
'''


import csv
import os
import random
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)
target_defect_ratio = 0.5
# fold = 5
neighbor = 5
tuned_parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100, 1000]}
classifier_for_selection = {"svm": svm.SVC(), "knn": neighbors.KNeighborsClassifier(), "rf": RandomForestClassifier(),
                            "tree": tree.DecisionTreeClassifier()}

classifier = "svm"


class stable_SMOTE:
    def __init__(self, z_nearest=5):
        self.z_nearest = z_nearest

    def fit_sample(self, x, y):

        # x_dataset = pd.DataFrame(x_dataset)
        x_dataset = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1)  # 组合数据和标签
        total_pair = []
        # print(k_nearest)
        feature = x_dataset.shape[1] - 1
        x_dataset.columns = list(range(x_dataset.shape[1]))  # 组合数据和标签时的标签列的索引需要更新
        defective_instance = x_dataset[x_dataset[feature] > 0]
        clean_instance = x_dataset[x_dataset[feature] == 0]
        defective_number = len(defective_instance)
        clean_number = len(clean_instance)
        need_number = int((target_defect_ratio * len(x_dataset) - defective_number) / (
                1 - target_defect_ratio))  # clean_number - defective_number
        # print(need_number)
        # print(clean_number - defective_number)
        # exit()
        if need_number <= 0:
            return False
        generated_dataset = []
        synthetic_dataset = pd.DataFrame()
        number_on_each_instance = need_number / defective_number  # 每个实例分摊到了生成几个的任务
        total_pair = []

        rround = number_on_each_instance / self.z_nearest
        while rround >= 1:
            for index, row in defective_instance.iterrows():
                temp_defective_instance = defective_instance.copy(deep=True)
                subtraction = row - temp_defective_instance
                square = subtraction ** 2
                row_sum = square.apply(lambda s: s.sum(), axis=1)
                distance = row_sum ** 0.5
                temp_defective_instance["distance"] = distance
                temp_defective_instance = temp_defective_instance.sort_values(by="distance", ascending=True)
                neighbors = temp_defective_instance[1:self.z_nearest + 1]
                for a, r in neighbors.iterrows():
                    selected_pair = [index, a]
                    selected_pair.sort()
                    total_pair.append(selected_pair)
            rround = rround - 1
        need_number1 = need_number - len(total_pair)
        number_on_each_instance = need_number1 / defective_number

        for index, row in defective_instance.iterrows():
            temp_defective_instance = defective_instance.copy(deep=True)
            subtraction = row - temp_defective_instance
            square = subtraction ** 2
            row_sum = square.apply(lambda s: s.sum(), axis=1)
            distance = row_sum ** 0.5

            temp_defective_instance["distance"] = distance
            temp_defective_instance = temp_defective_instance.sort_values(by="distance", ascending=True)
            neighbors = temp_defective_instance[1:self.z_nearest + 1]
            neighbors = neighbors.sort_values(by="distance", ascending=False)  # 这里取nearest neighbor里最远的
            target_sample_instance = neighbors[0: int(number_on_each_instance)]
            target_sample_instance = target_sample_instance.drop(columns="distance")
            for a, r in target_sample_instance.iterrows():
                selected_pair = [index, a]
                selected_pair.sort()
                total_pair.append(selected_pair)
        temp_defective_instance = defective_instance.copy(deep=True)
        residue_number = need_number - len(total_pair)
        residue_defective_instance = temp_defective_instance.sample(n=residue_number)
        for index, row in residue_defective_instance.iterrows():
            temp_defective_instance = defective_instance.copy(deep=True)
            subtraction = row - temp_defective_instance
            square = subtraction ** 2
            row_sum = square.apply(lambda s: s.sum(), axis=1)
            distance = row_sum ** 0.5

            temp_defective_instance["distance"] = distance
            temp_defective_instance = temp_defective_instance.sort_values(by="distance", ascending=True)
            neighbors = temp_defective_instance[1:self.z_nearest + 1]
            target_sample_instance = neighbors[-1:]
            for a in target_sample_instance.index:
                selected_pair = [index, a]
                selected_pair.sort()
                total_pair.append(selected_pair)
        total_pair_tuple = [tuple(l) for l in total_pair]
        result = Counter(total_pair_tuple)
        result_number = len(result)
        result_keys = result.keys()
        result_values = result.values()
        for f in range(result_number):
            current_pair = list(result_keys)[f]
            row1_index = current_pair[0]
            row2_index = current_pair[1]
            row1 = defective_instance.loc[row1_index]
            row2 = defective_instance.loc[row2_index]
            generated_num = list(result_values)[f]
            generated_instances = np.linspace(row1, row2, generated_num + 2)
            generated_instances = generated_instances[1:-1]
            generated_instances = generated_instances.tolist()
            for w in generated_instances:
                generated_dataset.append(w)
        final_generated_dataset = pd.DataFrame(generated_dataset)
        # final_generated_dataset = final_generated_dataset.rename(
        #     columns={0: "wmc", 1: "dit", 2: "noc", 3: "cbo", 4: "rfc", 5: "lcom", 6: "ca", 7: "ce", 8: "npm",
        #              9: "lcom3", 10: "loc", 11: "dam", 12: "moa", 13: "mfa", 14: "cam", 15: "ic", 16: "cbm", 17: "amc",
        #              18: "max_cc", 19: "avg_cc", 20: "bug"}
        # )
        result = pd.concat([clean_instance, defective_instance, final_generated_dataset])
        return result


def separate_data(original_data):
    '''

    用out-of-sample bootstrap方法产生训练集和测试集,参考论文An Empirical Comparison of Model Validation Techniques for DefectPrediction Models
    A bootstrap sample of size N is randomly drawn with replacement from an original dataset that is also of size N .
    The model is instead tested using the rows that do not appear in the bootstrap sample.
    On average, approximately 36.8 percent of the rows will not appear in the bootstrap sample, since the bootstrap sample is drawn with replacement.
    OriginalData:整个数据集

    return: 划分好的 训练集和测试集

    '''
    original_data = np.array(original_data).tolist()
    size = len(original_data)
    train_dataset = []
    train_index = []
    for i in range(size):
        index = random.randint(0, size - 1)
        train_instance = original_data[index]
        train_dataset.append(train_instance)
        train_index.append(index)

    original_index = [z for z in range(size)]
    train_index = list(set(train_index))
    test_index = list(set(original_index).difference(set(train_index)))
    original_data = np.array(original_data)
    train_dataset = original_data[train_index]
    # original_data = pd.DataFrame(original_data)
    # original_data = original_data.rename(
    #     columns={0: "wmc", 1: "dit", 2: "noc", 3: "cbo", 4: "rfc", 5: "lcom", 6: "ca", 7: "ce", 8: "npm",
    #              9: "lcom3", 10: "loc", 11: "dam", 12: "moa", 13: "mfa", 14: "cam", 15: "ic", 16: "cbm", 17: "amc",
    #              18: "max_cc", 19: "avg_cc", 20: "bug"})
    # train_dataset = pd.DataFrame(train_dataset)
    # train_dataset = train_dataset.rename(
    #     columns={0: "wmc", 1: "dit", 2: "noc", 3: "cbo", 4: "rfc", 5: "lcom", 6: "ca", 7: "ce", 8: "npm",
    #              9: "lcom3", 10: "loc", 11: "dam", 12: "moa", 13: "mfa", 14: "cam", 15: "ic", 16: "cbm", 17: "amc",
    #              18: "max_cc", 19: "avg_cc", 20: "bug"}
    # )
    test_dataset = original_data[test_index]
    return train_dataset, test_dataset


def sampling_start(path, sampled_data_save_path):
    file_target = []
    file_data = []
    file_path = os.listdir(path)
    os.mkdir(sampled_data_save_path + "S-SMOTE" + '/')
    for i in file_path:
        if 'label' in i:
            file_target.append(path + i)
        else:
            file_data.append(path + i)
    for i in range(0, len(file_data)):
        data = np.loadtxt(file_data[i], dtype=float, delimiter=',')
        target = np.loadtxt(file_target[i], dtype=float, delimiter=',')
        print(file_data[i].split('.')[0].split('/')[-1].split('_')[0] + ' in S-SMOTE is running……')
        result = stable_SMOTE().fit_sample(data, target)
        label_save = pd.DataFrame(result).iloc[:, -1]
        data_save = pd.DataFrame(result).drop([result.shape[1]-1], axis=1)
        data_save.to_csv(
            sampled_data_save_path + "S-SMOTE" + '/' + file_data[i].split('.')[0].split('/')[-1] + '_S-SMOTE.csv',
            header=False, index=False)
        label_save.to_csv(
            sampled_data_save_path + "S-SMOTE" + '/' + file_target[i].split('.')[0].split('/')[-1] + '_S-SMOTE.csv',
            header=False, index=False)
        print(file_data[i].split('.')[0].split('/')[-1].split('_')[0] + ' in S-SMOTE is Done')


def ten_fold_sampling_start(path, samplied_data_save_path):
    file_path = os.listdir(path)
    save_path = os.listdir(samplied_data_save_path)
    for i in range(len(file_path)):
        sampling_start(path + file_path[i] + '/', samplied_data_save_path + save_path[i] + '/')


path = 'D:/python_project/projectfile/dataset_prepare/10-fold/all-train/'
save = 'D:/python_project/projectfile/dataset_sampling/FIO-V2/all-train/'
ten_fold_sampling_start(path, save)

# path = 'D:/python_project/projectfile/dataset_prepare/10-fold/software-bug-train/'
# save = 'D:/python_project/projectfile/dataset_sampling/diffuision/'
# ten_fold_sampling_start(path, save)

# data_path = 'D:/python_project/projectfile/KC1_train.csv'
# label_path = 'D:/python_project/projectfile/KC1_label.csv'
# data = np.loadtxt(data_path, dtype=float, delimiter=',')
# label = np.loadtxt(label_path, dtype=float, delimiter=',')
# samling_data = stable_SMOTE().fit_sample(data, label)
# samling_data.to_csv('D:/python_project/projectfile/KC1_S-SMOTE.csv', header=False, index=False)
