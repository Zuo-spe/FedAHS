import os

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


# 分层十折交叉验证分割数据集的实例
def ten_fold(datapath, targetfile, save_path):
    data = np.loadtxt(datapath, dtype=float, delimiter=',')
    target = np.loadtxt(targetfile, dtype=int)
    # skf = StratifiedKFold(n_splits=10, shuffle=False)
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    count = 0
    for train_index, test_index in skf.split(data, target):
        # print('test_index', test_index)
        # print('train_index', train_index, 'test_index', test_index)
        name = datapath.split('/')[-1][:-9]
        train_X, train_y = data[train_index], target[train_index]
        test_X, test_y = data[test_index], target[test_index]
        os.makedirs(save_path + '10_KF/' + str(name) + '/', exist_ok=True)
        fw_train = open(save_path +'10_KF/' + str(name) + '/train_dataset_' + str(count) + '.csv', "a")

        #fw_target = open(save_path + 'train' + str(count) + '-fold/' + targetfile.split('/')[-1], "a")
        for i in range(0, len(train_X)):
            data_temp = ""
            for j in range(0, len(train_X[0])):
                data_temp = data_temp + str(train_X[i][j]) + ','
            data_temp = data_temp + str(train_y[i]) + ','
            data_temp = data_temp[:len(data_temp) - 1] + '\n'
            fw_train.write(data_temp)
            #fw_target.write(str(train_y[i]) + '\n')
        fw_train.close()
        #fw_target.close()
        os.makedirs(save_path + '10_KF/' + str(name) + '/', exist_ok=True)
        fw_test_X = open(save_path + '10_KF/' + str(name) + '/test_dataset_'  + str(count) + '.csv', "a")
        #fw_test_Y = open(save_test_path + 'test' + str(count) + '-fold/' + targetfile.split('/')[-1], "a")
        for i in range(0, len(test_X)):
            data_temp = ""
            for j in range(0, len(test_X[0])):
                data_temp = data_temp + str(test_X[i][j]) + ','
            data_temp = data_temp + str(test_y[i]) + ','
            data_temp = data_temp[:len(data_temp) - 1] + '\n'
            fw_test_X.write(data_temp)
            #fw_test_Y.write(str(test_y[i]) + '\n')
        fw_test_X.close()
        #fw_test_Y.close()
        count = count + 1


def start(data_path, save_path):
    file_target = []
    file_data = []
    file_path = os.listdir(data_path)

    for i in file_path:
        if "label" in i:
            file_target.append(data_path + '/' + i)
        else:
            file_data.append(data_path + '/' + i)
    for i in range(0, len(file_data)):  # 遍历文件夹内的每个数据集
        ten_fold(file_data[i], file_target[i], save_path)
        print(file_target[i].split('/')[-1].split('.')[0].split('_')[0] + ' is Done')


# data_path = "D:/projectfile/dataset_prepare/Normalized/"
# save_train_path = "D:/projectfile/dataset_prepare/10-fold/train/"
# save_test_path = "D:/projectfile/dataset_prepare/10-fold/test/"
# start(data_path, save_train_path, save_test_path)
data_path = r"I:\desktop\1"
save_path = r"I:\desktop\2"
start(data_path, save_path)