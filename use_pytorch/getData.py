
import pandas as pd

class GetDataSet_KFold(object):
    def __init__(self, dataSetName,i_KF):
        self.name = dataSetName
        self.train_datasets = None
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None
        self._index_in_train_epoch = 0
        print("dataSetName:", self.name)
        print("i_KF:", i_KF)
        data_dir_train = 'data/{}/train_dataset_{}.csv'.format(dataSetName, i_KF)
        data_dir_test = 'data/{}/test_dataset_{}.csv'.format(dataSetName, i_KF)
        train_dataset = pd.read_csv(data_dir_train, header=None)
        test_dataset = pd.read_csv(data_dir_test, header=None)
        self.train_datasets = train_dataset
        self.train_data = train_dataset.iloc[:, 0:train_dataset.shape[1] - 1]
        self.train_label = train_dataset.iloc[:, -1]
        self.test_data = test_dataset.iloc[:, 0:test_dataset.shape[1] - 1]
        self.test_label = test_dataset.iloc[:, -1]






