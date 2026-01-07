from cgitb import small
from lib2to3.btm_matcher import type_repr
import queue
from imblearn.under_sampling import CondensedNearestNeighbour, ClusterCentroids, OneSidedSelection,  NearMiss,\
     EditedNearestNeighbours
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
import numpy as np
import threading
from sklearn.cluster import KMeans
from pandas import concat
import multiprocessing as mp
import pandas as pd
from OREM import OREM
from model_Dit.train import *

'''
def OverSampling_methods(q_over_x, q_over_y, x_train, y_train, big, small, a):
    vs1 = small * (1 + a) / big  # vs为采样后少数类和多数类的比

    if vs1 <= 1:
        X_resampled_smote, y_resampled_smote = OREM(x_train, y_train,vs1)  # 过采样
        X_resampled_smote, y_resampled_smote = pd.DataFrame(X_resampled_smote), pd.DataFrame(y_resampled_smote)
    else:  # 过采样后少数类比多数类多

        X_resampled_smote1, y_resampled_smote1 = OREM(x_train, y_train,1)  # 过采样
        delta = int(small * (1 + a) - big)
        X_resampled_smote2, y_resampled_smote2 = OREM(x_train, y_train,1)
        X_resampled_smote1, y_resampled_smote1 = pd.DataFrame(X_resampled_smote1), pd.DataFrame(y_resampled_smote1)
        X_resampled_smote2, y_resampled_smote2 = pd.DataFrame(X_resampled_smote2), pd.DataFrame(y_resampled_smote2)
        X_resampled_smote = concat([X_resampled_smote1, X_resampled_smote2[0:delta]], axis=0)
        y_resampled_smote = concat([y_resampled_smote1, y_resampled_smote2[0:delta]], axis=0)

    resampled = concat([X_resampled_smote, y_resampled_smote], axis=1)
    resampled = resampled.loc[resampled.iloc[:, -1] == 1]
    x_smote_small = resampled.iloc[:, 0: -1]
    y_smote_small = resampled.iloc[:, -1]

    q_over_x.put(x_smote_small)
    q_over_y.put(y_smote_small)
'''
'''
def OverSampling_methods(q_over_x, q_over_y, x_train, y_train, big, small, a,client_data):
    vs1 = small * (1 + a) / big  
    client_dataset = client_data
    model_path = 'weight'  
    train(parent_dir=model_path, model_type='mlp', client_dataset=client_dataset, min_class=1,
          model_params={})  
    if vs1 <= 1:
        X_resampled_smote, y_resampled_smote = sample(model_type='mlp', client_dataset=client_dataset,
                              min_class=1, sampling_rate=vs1,big=big,  model_path=model_path, model_params={})
    else:  
        X_resampled_smote1, y_resampled_smote1 =  sample(model_type='mlp', client_dataset=client_dataset,
                              min_class=1, sampling_rate=vs1,big=big,  model_path=model_path, model_params={}) # 过采样
        delta = int(small * (1 + a) - big)
        X_resampled_smote2, y_resampled_smote2 = sample(model_type='mlp', client_dataset=client_dataset,
                                                        min_class=1, sampling_rate=delta, big=big, model_path=model_path,
                                                        model_params={})
        X_resampled_smote1, y_resampled_smote1 = pd.DataFrame(X_resampled_smote1), pd.DataFrame(y_resampled_smote1)
        X_resampled_smote2, y_resampled_smote2 = pd.DataFrame(X_resampled_smote2), pd.DataFrame(y_resampled_smote2)
        X_resampled_smote = concat([X_resampled_smote1, X_resampled_smote2[0:delta]], axis=0)
        y_resampled_smote = concat([y_resampled_smote1, y_resampled_smote2[0:delta]], axis=0)

    resampled = concat([X_resampled_smote, y_resampled_smote], axis=1)
    resampled = resampled.loc[resampled.iloc[:, -1] == 1]
    x_smote_small = resampled.iloc[:, 0: -1]
    y_smote_small = resampled.iloc[:, -1]

    q_over_x.put(x_smote_small)
    q_over_y.put(y_smote_small)
'''
def OverSampling_methods(q_over_x, q_over_y, x_train, y_train, big, small, a):
    x_train, y_train = pd.DataFrame(x_train), pd.DataFrame(y_train)
    r_cur = small / big
    vs1 = small * (1 + a) / big
    vs1 = float(np.clip(vs1, r_cur, 2.0))
    if small / big == 1:
        X_resampled_smote, y_resampled_smote = pd.DataFrame(x_train), pd.DataFrame(y_train)

    else:
        if vs1 <= 1 :
            sm = SMOTE(sampling_strategy = vs1 )
            X_resampled_smote, y_resampled_smote = sm.fit_resample(x_train, y_train.values.ravel())  # 过采样
            X_resampled_smote, y_resampled_smote = pd.DataFrame(X_resampled_smote), pd.DataFrame(y_resampled_smote)

        else :  # 过采样后少数类比多数类多
            sm = SMOTE(sampling_strategy = 1)
            X_resampled_smote, y_resampled_smote = sm.fit_resample(x_train, y_train.values.ravel())  # 过采样
            X_resampled_smote, y_resampled_smote = pd.DataFrame(X_resampled_smote), pd.DataFrame(y_resampled_smote)
            delta = int(small * (1 + a) - big)
            X_resampled_smote2, y_resampled_smote2 = sm.fit_resample(x_train, y_train.values.ravel())
            X_resampled_smote1, y_resampled_smote1 = pd.DataFrame(X_resampled_smote), pd.DataFrame(y_resampled_smote)
            X_resampled_smote2, y_resampled_smote2 = pd.DataFrame(X_resampled_smote2), pd.DataFrame(y_resampled_smote2)
            X_resampled_smote = concat([X_resampled_smote1, X_resampled_smote2[0:delta]], axis=0)
            y_resampled_smote = concat([y_resampled_smote1, y_resampled_smote2[0:delta]], axis=0)

    resampled = concat([X_resampled_smote, y_resampled_smote], axis=1)
    resampled = resampled.loc[resampled.iloc[:, -1] == 1]
    x_smote_small = resampled.iloc[:, 0: -1]
    y_smote_small = resampled.iloc[:, -1]

    q_over_x.put(x_smote_small)
    q_over_y.put(y_smote_small)

def UnderSampling_methods(q_under_x, q_under_y, x_train, y_train,big, small,b):
    b = b/100
    x_train, y_train = pd.DataFrame(x_train), pd.DataFrame(y_train)
    vs2 = small / (big * (1 - b))
    ratio = small / ((big - ((big - small) / 2)))
    if small / big == 1:
        X_resampled, y_resampled = pd.DataFrame(x_train), pd.DataFrame(y_train)
    else:
        if vs2 >= small/big and vs2 < 1 :
            nm1 = NearMiss(sampling_strategy = vs2 , version=2)
            X_resampled, y_resampled = nm1.fit_resample(x_train, y_train)
            X_resampled, y_resampled = pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)
        else:
            rus = NearMiss(sampling_strategy=1,version=2)
            # rus = ClusterCentroids(ratio=1)
            X_resampled, y_resampled = rus.fit_resample(x_train, y_train.values.ravel())  # 欠采样
            A = small - (big * (1 - b))
            delta = int(A)
            X_resampled, y_resampled = pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)
            X_resampled = X_resampled[delta:]
            y_resampled = y_resampled[delta:]

    resampled = concat([X_resampled, y_resampled], axis=1)
    resampled = resampled.loc[resampled.iloc[:, -1] == 0]
    x_res_big = resampled.iloc[:, 0: -1]
    y_res_big = resampled.iloc[:, -1]

    q_under_x.put(x_res_big)
    q_under_y.put(y_res_big)
    '''
def under_sampling(q_under_x, q_under_y, x_train, y_train, big, small, b):
    vs2 = small / (big * (1 - b))
    #x_train, y_train = pd.DataFrame(x_train), pd.DataFrame(y_train)
    if vs2 <= 1:
        SDUS1_test = SDUS1(sampling_rate=vs2)
        X_res, y_res = SDUS1_test.fit(x_train, y_train) # 欠采样
        X_res, y_res = pd.DataFrame(X_res), pd.DataFrame(y_res)
    else:  # 欠采样后少数类比多数类多
        SDUS1_test = SDUS1(sampling_rate=1)
        # rus = ClusterCentroids(ratio=1)
        X_res, y_res = SDUS1_test.fit(x_train, y_train)  # 欠采样
        A = small - (big * (1 - b))
        delta = int(A)
        X_res, y_res = pd.DataFrame(X_res), pd.DataFrame(y_res)
        X_res = X_res[delta:]
        y_res = y_res[delta:]

    resampled = concat([X_res, y_res], axis=1)
    resampled = resampled.loc[resampled.iloc[:, -1] == 0]
    x_res_big = resampled.iloc[:, 0: -1]
    y_res_big = resampled.iloc[:, -1]

    q_under_x.put(x_res_big)
    q_under_y.put(y_res_big)
        '''
#client_data not use
def resampling(x_train, y_train, big, small,a,b,client_data):
    q_under_x = queue.Queue()
    q_under_y = queue.Queue()
    q_over_x = queue.Queue()
    q_over_y = queue.Queue()
    a = round(a, 2)
    b = round(b, 2)
    under_p = threading.Thread(target=UnderSampling_methods, args=(q_under_x, q_under_y, x_train, y_train, big, small,b))  # 创建进程
    over_p = threading.Thread(target=OverSampling_methods, args=(q_over_x, q_over_y, x_train, y_train, big, small,a))
    under_p.start()
    over_p.start()
    x_train_under = q_under_x.get()
    y_train_under = q_under_y.get()
    x_train_over = q_over_x.get()
    y_train_over = q_over_y.get()
    x_train_resampling = concat([x_train_under, x_train_over], axis=0)
    y_train_resampling = concat([y_train_under, y_train_over], axis=0)
    over_p.join()
    under_p.join()

    x_train_resampling = x_train_resampling.to_numpy()
    y_train_resampling = y_train_resampling.to_numpy()
    return x_train_resampling, y_train_resampling













