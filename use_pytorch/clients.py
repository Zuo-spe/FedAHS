import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet_KFold
from sample_dirichlet import Partitioner
import pandas as pd
import torch.nn.functional as F
from Parallel_ReSampling import *
from torchmetrics.classification import BinaryAUROC
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

class client(object):
    def __init__(self, trainDataSet, dev, i_KF, Res, dataSetName, client_data):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        self.Size_dataset = len(trainDataSet)
        self.Res = Res
        self.dataSetName = dataSetName
        self.i_KF = i_KF
        self.a = None
        self.client_data = client_data
        self.b = None
        # =========  client_data 划分 =========
        data_dir_test = r'data\{}\test_dataset_{}.csv'.format(
            dataSetName, i_KF)
        test_dataset = pd.read_csv(data_dir_test, header=None)

        self.test_data = torch.tensor(
            test_dataset.iloc[:, 0:test_dataset.shape[1] - 1].values,
            dtype=torch.float32
        ).to(self.dev)

        self.test_labels = torch.tensor(
            test_dataset.iloc[:, -1].values,
            dtype=torch.long
        ).to(self.dev)


    def Local_val(self, Net):
        Net.eval()
        with torch.no_grad():
            logits = Net(self.test_data)  # [N, C]
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            labels = self.test_labels.cpu().numpy()

        try:
            auc_score = roc_auc_score(labels, probs)
        except ValueError:
            auc_score = 0.5

        preds = (probs >= 0.5).astype(int)

        try:
            f1 = f1_score(labels, preds)
        except ValueError:
            f1 = 0.0

        try:
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            tpr = tp / (tp + fn + 1e-8)  # Recall / Sensitivity
            tnr = tn / (tn + fp + 1e-8)  # Specificity
            gmean = np.sqrt(tpr * tnr)
        except ValueError:
            gmean = 0.0
        auc_score = round(auc_score, 4)
        f1  = round(f1, 4)
        gmean = round(gmean, 4)

        return auc_score

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, a, b):
        self.a = a
        self.b = b

        Net.load_state_dict(global_parameters, strict=True)
        full_data = self.train_ds.tensors[0].to(self.dev).float()
        full_labels = self.train_ds.tensors[1].to(self.dev).long()
        biglast = (full_labels == 0).sum().item()
        smalllast = (full_labels == 1).sum().item()

        if self.Res:
            #print("Begin sampling")
            resampled_data, resampled_labels = resampling(full_data.cpu().numpy(), full_labels.cpu().numpy(), biglast,
                                                          smalllast, a, b, self.client_data)
            biglast = (resampled_labels == 0).sum().item()
            smalllast = (resampled_labels == 1).sum().item()
            full_data = torch.tensor(resampled_data).to(self.dev).float()
            full_labels = torch.tensor(resampled_labels).to(self.dev).long()

        else:
            print("No sampling operation")


        epoch_loss = []
        all_preds = []
        all_labels = []


        for epoch in range(localEpoch):
            opti.zero_grad()
            preds = Net(full_data)
            loss = lossFun(preds, full_labels)
            loss.backward()
            opti.step()
            epoch_loss.append(loss.item())
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(full_labels.cpu().numpy())
        test_auc = self.Local_val(Net)
        return Net.state_dict(), sum(epoch_loss) / len(epoch_loss), test_auc


class ClientsGroup(object):
    def __init__(self, dataSetName, n_kf, i_kf, isIID, isRes, numOfClients, dirichlet_alpha, seed_v, dev):
        self.data_set_name = dataSetName
        self.n_kf = n_kf
        self.i_KF = i_kf
        self.is_iid = isIID
        self.is_Res = isRes
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.dirichlet = dirichlet_alpha
        self.seed_value = seed_v
        self.test_data_loader = None
        self.testset_label = None
        self.dataSet_DirichletAllocation()

    def dataSet_DirichletAllocation(self):
        DataSet = GetDataSet_KFold(self.data_set_name, self.i_KF)
        test_data = torch.tensor(DataSet.test_data.values)
        test_label = torch.tensor(DataSet.test_label.values)
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=512, shuffle=False)
        self.testset_label = test_label
        train_dataset = DataSet.train_datasets
        non_iid_partitioner = Partitioner(train_dataset, self.num_of_clients, self.dirichlet, self.seed_value)
        user_groups = non_iid_partitioner.partition()

        for i in range(self.num_of_clients):
            client_index = list(user_groups[i])
            client_data = train_dataset.loc[client_index]
            local_data = client_data.iloc[:, 0:client_data.shape[1] - 1].values
            local_label = client_data.iloc[:, -1].values.astype(int)

            # num_class_0 = (local_label == 0).sum().item()
            # num_class_1 = (local_label == 1).sum().item()
            # print(f"client{i}majority class: {num_class_0}")
            # print(f"客户端{i}Minority class: {num_class_1}")

            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev, self.i_KF,
                             self.is_Res, self.data_set_name, client_data)

            self.clients_set['client{}'.format(i)] = someone  # self.clients_set（dict），存储内容为 class object




