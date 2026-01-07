import csv

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

def OREM(data, label, sampling_rate, q):

    minclass = 1
    min_ind = np.where(label == minclass)[0]
    maj_ind = np.where(label != minclass)[0]
    data_p = data[min_ind]  # data_positive
    data_n = data[maj_ind]  # data_negative
    np1 = len(min_ind)  # number of minority
    nn = len(maj_ind)  # number of majority

    # Calculate the number of samples to generate
    n_g = int(sampling_rate * nn) - np1
    if n_g <= 0:
        raise ValueError("The calculated number of samples to generate is not positive.")

    # Compute nearest neighbors
    NNI_p, NND_p = computeDis(data_p, data_p, len(min_ind) - 1, 'euclidean', include_self=True)
    NNI_n, NND_n = computeDis(data_p, data_n, len(maj_ind), 'euclidean')

    # Discover critical areas for sampling
    CAS = discovCMR(NND_p, NND_n, NNI_p, NNI_n, q)
    AS = idenCleanReg(data_p, data_n, CAS, 'euclidean')

    os_ind = np.repeat(np.arange(len(min_ind)), np.floor(n_g / len(min_ind)))
    os_ind = np.append(os_ind, np.random.choice(np.arange(len(min_ind)), n_g - len(os_ind), replace=False))
    SynData = np.zeros((n_g, data_p.shape[1]))

    # Generate synthetic samples
    for i in range(n_g):
        SynData[i, :] = Generate(data_p[os_ind[i]], data_p, data_n, AS[os_ind[i]])

    data_oversampled = np.vstack((data, SynData))
    labels_syn = np.full(n_g, minclass)
    label_oversampled = np.hstack((label, labels_syn))
    # Combine original and synthetic data
    #data_oversampled = np.vstack((data, SynData))
    #label_oversampled = np.vstack((label, np.full((n_g, 1),1)))

    return data_oversampled, label_oversampled

def Generate(sample, data_p, data_n, AS):

    data = np.vstack([data_p, data_n])
    if len(AS) == 0:
        return sample
    ind = np.random.choice(len(AS), 1, p=np.ones(len(AS)) / len(AS))
    gap = np.random.rand(data.shape[1])

    if AS[ind[0]] > len(data_p):
        gap /= 2
    syn = sample + gap * (data[AS[ind[0]]] - sample)
    return syn

def idenCleanReg(data_p, data_n, CAS, distance_metric):

    num_points = data_p.shape[0]
    data = np.vstack([data_p, data_n])
    AS = [None] * num_points
    for i in range(num_points):
        AS[i] = []
        CAS[i] = list(map(int, CAS[i]))
        for j in CAS[i]:
            mean_i = np.mean([data_p[i, :], data[j, :]], axis=0)
            thre_dis_ij = distance.cdist([data_p[i, :]], [mean_i], metric=distance_metric)[0][0]
            dis_i = distance.cdist(data[CAS[i][:j], :], [mean_i], metric=distance_metric).flatten()
            smaller_dis_ij_ind = np.where(dis_i[:j-1] - thre_dis_ij < 1e-5)[0]
            maj_count = np.sum(np.array(CAS[i])[smaller_dis_ij_ind] > len(data_p))
            if maj_count == 0:
                AS[i].append(j)
    return AS

def discovCMR(NND_p, NND_n, NNI_p, NNI_n, q):
    num_points = NND_p.shape[0]
    CAS = [None] * num_points
    Mark = [None] * num_points

    for i in range(num_points):  # 依次求每个少数类样本的CMR
        dis_i = np.hstack([NND_p[i, :], NND_n[i, :]])
        ind_i = np.hstack([NNI_p[i, :], NNI_n[i, :] + num_points])
        sorted_ind = np.argsort(dis_i)
        count_break = 0
        CAS[i] = ind_i[sorted_ind].tolist()
        for j in range(len(sorted_ind)):
            temp = len(NND_p[i, :])
            if sorted_ind[j] >= len(NND_p[i, :]):
                count_break += 1
            else:
                count_break = 0
            if count_break >= q:
                CAS[i] = ind_i[sorted_ind[:max(j-q+1, 1)]].tolist()  # 是不是应该是j-q+1？因为下标是从0开始，不加1的话，少了一个？？
                Mark[i] = [ind_i[j - q], dis_i[j - q]]
                break
    return CAS

def computeDis(data_u, data_s, K, distance_metric, include_self=False):
    # !!计算 距离可以试一下除了欧式距离以外的
    # data_u: the considered data
    # data_s: the search data
    # include_self: whether the samples in data_u are included in data_s
    nbrs = NearestNeighbors(n_neighbors=K+1 if include_self else K, algorithm='auto', metric=distance_metric).fit(data_s)
    distances, indices = nbrs.kneighbors(data_u)

    if include_self:
        indices1 = np.ones((indices.shape[0], indices.shape[1] - 1))
        distances1 = np.ones((distances.shape[0], distances.shape[1] - 1))
        for i in range(data_u.shape[0]):
            self_index = np.where(indices[i] == i)[0][0]
            indices1[i] = np.delete(indices[i], self_index)
            distances1[i] = np.delete(distances[i], self_index)
        return indices1, distances1
    else:
        return indices, distances


# 读取已经归一化与编码后的数据
with open("data/magic.csv", 'rt') as raw_data:
    readers = csv.reader(raw_data, delimiter=',')
    dr_data = list(readers)
    dr_data = np.array(dr_data)
dr_data = dr_data.astype(float)
fearture_data = dr_data[:, -1]  # 标签数据
real_data = dr_data[:, :-1]  # 特征数据

# 交叉验证分割数据集，因为仅用于测试，这里只取了最后一组
skf = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
for train_index, test_index in skf.split(real_data, fearture_data):
            # print('test_index', test_index)
            # print('train_index', train_index, 'test_index', test_index)
    X_train, y_train = real_data[train_index], fearture_data[train_index]
    X_test, y_test = real_data[test_index], fearture_data[test_index]

sampling_rate = 1.05
q = 2  # 当少数类某样本周围连续出现q个多数类样本（按距离先排列某样本周围的所有样本）

class_0_samples = X_test[y_test == 0][:20]  # 提取类别为0的样本
class_1_samples = X_test[y_test == 1][:5]  # 提取类别为1的样本
X = np.concatenate((class_0_samples, class_1_samples), axis=0)  # 合并样本
y = np.concatenate((np.zeros(20), np.ones(5)), axis=0)

data_oversampled, label_oversampled = OREM(X, y, sampling_rate, q)
# data_oversampled, label_oversampled = OREM(X_test, y_test, sampling_rate, q)


print('end')
