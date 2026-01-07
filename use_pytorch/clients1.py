import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import  GetDataSet_KFold
from sample_dirichlet import Partitioner
import pandas as pd
import torch.nn.functional as F
from Parallel_ReSampling import *
from torchmetrics.classification import BinaryAUROC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

class client(object):
    def __init__(self, trainDataSet, dev, i_KF, Res, dataSetName,client_data):
        self.train_ds = trainDataSet # 训练数据集
        #self.test_data_loader = test_loader# 测试数据集
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        self.Size_dataset = len(trainDataSet)
        self.Res = Res
        self.dataSetName = dataSetName
        self.i_KF = i_KF
        self.a = None
        self.client_data = client_data  # DataFrame，当前客户端的原始数据，用于OREM等模型
        self.b = None
        # ========= 本地验证集：由 client_data 划分 =========
        X = client_data.iloc[:, 0:client_data.shape[1] - 1].values.astype(np.float32)
        y = client_data.iloc[:, -1].values.astype(int)
        X_train_loc, X_val, y_train_loc, y_val = train_test_split(
            X, y,
            test_size=0.1,
            stratify=y,
            random_state=42
        )
        self.val_data = torch.tensor(X_val, dtype=torch.float32).to(self.dev)
        self.val_labels = torch.tensor(y_val, dtype=torch.long).to(self.dev)



    #本地模型评估
    def Local_val(self, Net):
        Net.eval()
        with torch.no_grad():
            logits = Net(self.val_data)  # [N, C]
            # 若输出为 logits，这里做 softmax 得到正类概率
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            labels = self.val_labels.cpu().numpy()

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

        return gmean


    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, a, b):
        self.a = a  # 保存 a 的值
        self.b = b  # 保存 b 的值

        # 加载当前通信中最新全局参数
        Net.load_state_dict(global_parameters, strict=True)

        # 提取完整数据和标签
        full_data = self.train_ds.tensors[0].to(self.dev).float()
        full_labels = self.train_ds.tensors[1].to(self.dev).long()

        #这里需要统计多数类与少数类
        # 统计full_labels中0和1的数量
        biglast = (full_labels == 0).sum().item()  # 计算标签为0的样本数
        smalllast = (full_labels == 1).sum().item()  # 计算标签为1的样本数


        # 如果需要采样，进行采样操作
        if self.Res:
            print("Begin Sampling")
            resampled_data, resampled_labels = resampling(full_data.cpu().numpy(), full_labels.cpu().numpy(),biglast, smalllast,a,b,self.client_data)

            biglast = (resampled_labels == 0).sum().item()  # 计算标签为0的样本数
            smalllast = (resampled_labels == 1).sum().item()  # 计算标签为1的样本数
            print("采样后多数类与少数类数量：", biglast, smalllast)
            full_data = torch.tensor(resampled_data).to(self.dev).float()
            full_labels = torch.tensor(resampled_labels).to(self.dev).long()

        else:
            print("未进行采样操作")

        # 初始化训练记录
        epoch_loss = []
        all_preds = []
        all_labels = []

        # Local epoch
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
        #self.data_temporary_save()  # 直接读取处理好的客户端数据集，原因：CNN处理的时间很长，1h左右，所需时间与训练1000轮的时间差不多


    def dataSet_DirichletAllocation(self):
        DataSet = GetDataSet_KFold(self.data_set_name, self.i_KF)
        test_data = torch.tensor(DataSet.test_data.values)
        test_label = torch.tensor(DataSet.test_label.values)
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=512, shuffle=False)
        self.testset_label = test_label
        train_dataset = DataSet.train_datasets
        non_iid_partitioner = Partitioner(train_dataset, self.num_of_clients, self.dirichlet, self.seed_value)
        user_groups = non_iid_partitioner.partition()

        # According to the obtained data index, divide the sample into clients
        for i in range(self.num_of_clients):
            client_index = list(user_groups[i])
            client_data = train_dataset.loc[client_index]
            local_data = client_data.iloc[:, 0:client_data.shape[1] - 1].values
            local_label = client_data.iloc[:, -1].values.astype(int)
            # 统计local_label中0和1的数量
            num_class_0 = (local_label == 0).sum().item()  # 计算标签为0的样本数
            num_class_1 = (local_label == 1).sum().item()  # 计算标签为1的样本数

            print(f"客户端{i}多数类数量: {num_class_0}")
            print(f"客户端{i}少数类数量: {num_class_1}")

            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev, self.i_KF,
                             self.is_Res, self.data_set_name,client_data)

            '''
            #这里进行并行混合采样,初始采样
            if self.is_Res:
                #这里欠采样率与过采样为初始化采样率只在第一次采样的过程中有效
                U_local_data, U_local_label = resampling(local_data, local_label,num_class_0,num_class_1, a = self.a, b = self.b)#这里local_data, local_label是numpy.ndarray类型
                someone = client(TensorDataset(torch.tensor(U_local_data), torch.tensor(U_local_label)), self.dev, self.i_KF , self.is_Res,self.data_set_name)#这里将上面的数据与标签转换为tensor类型
            else:
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)),  self.dev, self.i_KF, self.is_Res,self.data_set_name)
                print("未进行采样操作")
                # someone是 client类的实例化对象，client(self,trainDataset,dev)
            '''

            self.clients_set['client{}'.format(i)] = someone  # self.clients_set（dict），存储内容为 class object




