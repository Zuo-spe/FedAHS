#The dataset is divided into multiple clients, which is mainly used for data distribution in federated learning.
#The code mainly implements the data partition strategy based on Dirichlet distribution, which is suitable for non IID data partition.
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame


class Partitioner(object):
    def __init__(self, dataset: DataFrame, n_clients: int, alpha: float, seed_value: int, show_img: bool = False):
        '''
        Class for initializing data partitioning
        :param dataset: DataFrame
        :param n_clients: client number
        :param split_policy: split method, Incoming string non-iid or iid
        :param alpha: Parameters of Dirichlet distribution function
        '''
        self.dataset = dataset
        self.n_clients = n_clients
        self.alpha = alpha
        self.show_img = show_img
        np.random.seed(seed_value)

    def partition(self):
        labels = self.dataset.iloc[:, -1]
        target_name = labels.unique()
        target = [np.argwhere(target_name == name).flatten()
                  for name in labels]
        target = np.concatenate(np.array(target))
        num_cls = len(target_name)

        client_idcs = self.split_index(target, alpha=self.alpha, n_clients=self.n_clients)
        if self.show_img:
            self.show_split_img(target, client_idcs, num_cls, target_name)

        return client_idcs

    def split_index(self, target, alpha, n_clients):
        n_classes = target.max() + 1
        class_idcs = [np.argwhere(target == y).flatten() for y in range(n_classes)]
        label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)  # dirichlet分布的数据集划分
        client_idcs = [[] for _ in range(n_clients)]

        for c, fracs in zip(class_idcs, label_distribution):
            split_indexs = (np.cumsum(fracs)[:-1] * len(c)).astype(int)

            for i, idcs in enumerate(np.split(c, split_indexs)):
                client_idcs[i] += [idcs]

        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

        return client_idcs

