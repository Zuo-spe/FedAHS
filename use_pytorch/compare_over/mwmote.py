# Owen Coyne
# Implementation of oversampling method MWMOTE
# As presented in "MWMOTE--Majority Weighted Minority Oversampling Technique for Imbalanced Data Set Learning" (https://ieeexplore.ieee.org/document/6361394)
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering

warnings.filterwarnings("ignore", category=Warning)


#########################---Helper Functions--################

def closeness_factor(Yi, Xj, Smin, Nmin, Sbmaj, Cf_th, CMAX):
    if Smin.index.get_loc(Xj.name) not in Nmin[Sbmaj.index.get_loc(Yi.name)]:
        closeness_factor = 0
    else:
        d = np.linalg.norm(Yi.values - Xj.values)
        if (1 / d) <= Cf_th:
            f = (1 / d)
        else:
            f = Cf_th
        closeness_factor = (f / Cf_th) * CMAX

    return closeness_factor


def Davg(SminF):
    Davg = 0
    for DXindex, Dx in SminF.iterrows():
        temp = []
        for DYindex, Dy in SminF.iterrows():
            if (DYindex != DXindex):
                temp.append(np.linalg.norm(Dy - Dx))

        Davg += min(temp)

    Davg /= len(SminF)
    return Davg


def mwmote_(data, target):
    # Import Data

    imbalanced_data = pd.concat([pd.DataFrame(data), pd.DataFrame(target)], axis=1)
    imbalanced_data.columns = list(range(imbalanced_data.shape[1]))
    # imbalanced_data = pd.read_csv("D:/python_project/projectfile/KC1_train.csv",header=0)
    # print(imbalanced_data)
    feature = imbalanced_data.shape[-1] - 1
    # Input variables
    # Minority Set
    Smin = imbalanced_data[imbalanced_data[feature] > 0]
    # Majority Set
    Smaj = imbalanced_data[imbalanced_data[feature] == 0]
    # Number of Samples we need to create
    N = len(Smaj) - len(Smin)
    # Clustering Parameters
    k1 = 5
    k2 = 3
    k3 = int(len(Smin) / 2)

    # Other Parameters
    Cp = 3
    CMAX = 2
    Cf_th = 5
    #########################---Main Algorithm--################
    # print("Setup Beginning")

    # For each minority example Xi 2 Smin, compute the nearest neighbor set, which consists of the
    # nearest k1 neighbors of Xi according to euclidean distance.

    construction = NearestNeighbors(n_neighbors=k1 + 1)
    construction.fit(imbalanced_data)
    NN = construction.kneighbors(X=Smin.values, n_neighbors=k1 + 1, return_distance=False)

    # Construct the filtered minority set, Sminf by removing those minority class samples which have no minority
    # example in their neighborhood
    SFIDX = []
    for mcs in NN:
        # Get index of each minority class sample in NN and remove it from its subset (so it can't match to itself)
        tempidx = mcs[0]
        mcs = np.delete(mcs, 0)
        # Check if there are minority class samples
        if (any(Smin.index.isin(mcs)) == True):
            SFIDX.append(tempidx)

    # Construct set from indexes
    SminF = imbalanced_data.loc[np.array(SFIDX)]

    # For each in SminF construct the nearest majority set (Nmaj) which consists of the nearest k2 majority
    # samples from xi according to euclidean distance.
    majority_set = NearestNeighbors(n_neighbors=k2 + 1)
    majority_set.fit(Smaj)
    Nmaj = majority_set.kneighbors(X=SminF.values, n_neighbors=k2 + 1, return_distance=False)

    # Find the borderline majority set (Sbmaj) as the union of all Nmaj sets
    Sbmaj = imbalanced_data.loc[np.unique(Nmaj.reshape(-1))]

    # For each majority example in Sbmaj, compute the nearest minority set Nmin(Yi) which consists of
    # the nearest k3 minority examples from Yi by euclidean distance
    minority_set = NearestNeighbors(n_neighbors=k3 + 1)
    minority_set.fit(Smin)
    Nmin = minority_set.kneighbors(X=Sbmaj.values, n_neighbors=k3 + 1, return_distance=False)

    # Find the informative minority set, Simin, as the union of all Nmin(Yi)s
    Simin = Smin.iloc[np.unique(Nmin.reshape(-1))]

    # For each Yi belonging to Sbmaj and for each Xi belonging 2 Simin, compute the
    # information weight

    # Use this to calculate selection weights and probabilities for each Xi in Simin
    selection_weights = []
    for j, Xj in Simin.iterrows():
        weights = []
        c_f = []
        d_f = []
        for i, Yi in Sbmaj.iterrows():
            c_f.append(closeness_factor(Yi, Xj, Smin, Nmin, Sbmaj, Cf_th, CMAX))
        for counter in range(0, len(Sbmaj)):
            density_factor = c_f[counter] / np.array(c_f).sum()
            d_f.append(density_factor)

        information_weight = np.array(c_f) * np.array(d_f)

        selection_weights.append(information_weight.sum())

    selection_probs = selection_weights / np.array(selection_weights).sum()

    # Initiliase Somin the oversampled minority set
    oversampled_Smin = Smin
    # Calculate Th the parameter we use as distance threshold in our Agglomerative Clustering
    D_avg = Davg(SminF)
    Th = D_avg * Cp

    # Seperate minority set into clusters using Agglomerative Clustering
    model = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='average', distance_threshold=Th)
    clusters = model.fit_predict(Smin)

    # print("Sampling Beginning")
    # print(str(N) + " samples to be completed...")

    # Keep track of the end of the dataframe
    loop_index = len(imbalanced_data)
    # Iterate through the number of samples we need to create
    for k in range(0, N):
        # Progress Log
        # if (k % 10000) == 0: print("Samples completed: " + str(k))

        # Choose a random element of Simin based on the previously calculated selection probabilities
        sample = np.random.choice(Simin.index.values, p=selection_probs)

        # Locate the sample and its cluster
        searchval = clusters[Smin.index.get_loc(sample)]
        cluster_population = np.array(np.where(clusters == searchval)).reshape(-1)

        # Select another Sample from the same cluster as our original sample
        x_sample = Smin.loc[sample].values
        y_sample = Smin.iloc[np.random.choice(cluster_population)].values

        # Calculate a random alpha between [0,1]
        alpha = np.random.uniform()
        # Create synthetic sample where s = x + alpha*(y-x)
        synthetic = x_sample + alpha * (y_sample - x_sample)

        # Append synthetic sample to the oversampled dataframe
        output = pd.Series(data=synthetic, index=imbalanced_data.columns, name=loop_index)
        oversampled_Smin = oversampled_Smin.append(output)
        loop_index += 1

    # Add to original majority set
    result = Smaj.append(oversampled_Smin)
    return result
    # Append to csv


def sampling_start(path, sampled_data_save_path):
    file_target = []
    file_data = []
    file_path = os.listdir(path)
    os.mkdir(sampled_data_save_path + "MWMOTE" + '/')
    for i in file_path:
        if 'label' in i:
            file_target.append(path + i)
        else:
            file_data.append(path + i)
    for i in range(0, len(file_data)):
        data = np.loadtxt(file_data[i], dtype=float, delimiter=',')
        target = np.loadtxt(file_target[i], dtype=int, delimiter=',')
        print(file_data[i].split('.')[0].split('/')[-1].split('_')[0] + ' in MWMOTE is running……')
        result = mwmote_(data, target)
        label_save = pd.DataFrame(result).iloc[:, -1]
        data_save = pd.DataFrame(result).drop([result.shape[1] - 1], axis=1)
        data_save.to_csv(
            sampled_data_save_path + "MWMOTE" + '/' + file_data[i].split('.')[0].split('/')[-1] + '_MWMOTE.csv',
            header=False, index=False)
        label_save.to_csv(
            sampled_data_save_path + "MWMOTE" + '/' + file_target[i].split('.')[0].split('/')[-1] + '_MWMOTE.csv',
            header=False, index=False)
        print(file_data[i].split('.')[0].split('/')[-1].split('_')[0] + ' in MWMOTE is Done')


def ten_fold_sampling_start(path, samplied_data_save_path):
    file_path = os.listdir(path)
    save_path = os.listdir(samplied_data_save_path)
    for i in range(len(file_path)):
        sampling_start(path + file_path[i] + '/', samplied_data_save_path + save_path[i] + '/')



path = 'D:/python_project/projectfile/dataset_prepare/10-fold/all-train/'
save = 'D:/python_project/projectfile/dataset_sampling/FIO-V2/all-train/'
ten_fold_sampling_start(path, save)
os.system('shutdown -s -t 1')