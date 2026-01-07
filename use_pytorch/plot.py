import csv
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

matplotlib.use('Agg')

f_1 = pd.read_csv('D:/Project-python/Results_all/PRE_experiment/MLP(30)/Global_Banlance_2_no_sampling_GlobalComm300_E5_B16_lr0.1_Dirichlet1.0(seeds 30)_num_clients50_cf0.2.csv')
f_2 = pd.read_csv('D:/Project-python/Results_all/PRE_experiment_imb/MLP(30)/Global_Imbanlance_2_no_sampling_GlobalComm300_E5_B16_lr0.1_Dirichlet1.0(seeds 30)_num_clients50_cf0.2.csv')
f_3 = pd.read_csv('D:/Project-python/Results_all/PRE_experiment_imb/MLP(30)/Global_Imbanlance_3_no_sampling_GlobalComm300_E5_B16_lr0.1_Dirichlet1.0(seeds 30)_num_clients50_cf0.2.csv')


dataset_name = 'PRE_experiment'
model_name = 'MLP(30)'
plot_file = 'plot_2'

labels = ['global_balance', 'global_imbalance_2', 'global_imbalance_3']#, 'W_cnn', 'W_nm1', 'W(Fed_Avg)_Hash_U', 'no_sampling', 'rus', 'cnn', 'nm1']
colors = ['#f9ed69', '#fcbad3', '#8fbaf3']#, '#e23e57', '#a3de83', '#fdb87d', '#71c9ce', '#e0f9b5', '#cabbe9', '#c86b85']
f = [f_1, f_2, f_3]#, f_4, f_5, f_6, f_7, f_8, f_9, f_10]

# labels = ['W_Hash_U', 'no_sampling', 'rus', 'nm1', 'cnn', 'tl', 'oss']
# colors = ['#f9ed69', '#fcbad3', '#8fbaf3', '#e23e57', '#a3de83', '#fdb87d', '#cca8e9']
# f = [f_1, f_2, f_3, f_4, f_5, f_6, f_7]


'''---------------accuracy-------------'''
plt.figure()
plt.title('Testing Accuracy vs Communication rounds')
r = []
for i in range(len(f)):
    r_i = f[i].loc[:, 'accuracy']
    r.append(r_i)

for i in range(len(r)):
    plt.plot(range(len(r[i])), r[i], linewidth=0.8, color=colors[i], label=labels[i])

# plt.ylim(0.5, 0.8)
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.xlabel('Communication Rounds')
plt.savefig('D:/Project-python/Results_all/{}/{}/{}/accuracy.svg'.format(dataset_name, model_name, plot_file), dpi=600, format='svg')


'''------------f1-score-------------'''
plt.figure()
plt.title('Testing F1-score vs Communication rounds')
r = []
for i in range(len(f)):
    r_i = f[i].loc[:, 'F1_score']
    r.append(r_i)

for i in range(len(r)):
    plt.plot(range(len(r[i])), r[i], linewidth=0.8, color=colors[i], label=labels[i])

plt.legend(loc='lower right')
plt.ylabel('F1_score')
plt.xlabel('Communication Rounds')
plt.savefig('D:/Project-python/Results_all/{}/{}/{}/F1_score.svg'.format(dataset_name, model_name, plot_file), dpi=600, format='svg')


'''--------------G-means---------------'''
plt.figure()
plt.title('Testing G-means vs Communication rounds')
r = []
for i in range(len(f)):
    r_i = f[i].loc[:, 'G_means']
    r.append(r_i)

for i in range(len(r)):
    plt.plot(range(len(r[i])), r[i], linewidth=0.8, color=colors[i], label=labels[i])

plt.legend(loc='lower right')
plt.ylabel('G_means')
plt.xlabel('Communication Rounds')
plt.savefig('D:/Project-python/Results_all/{}/{}/{}/G_means.svg'.format(dataset_name, model_name, plot_file), dpi=600, format='svg')


'''------------AUC--------------'''
plt.figure()
plt.title('Testing AUC vs Communication rounds')
r = []
for i in range(len(f)):
    r_i = f[i].loc[:, 'AUC']
    r.append(r_i)

for i in range(len(r)):
    plt.plot(range(len(r[i])), r[i], linewidth=0.8, color=colors[i], label=labels[i])

# plt.ylim(0.5, 0.7)
plt.legend(loc='lower right')
plt.ylabel('AUC')
plt.xlabel('Communication Rounds')
plt.savefig('D:/Project-python/Results_all/{}/{}/{}/AUC.svg'.format(dataset_name, model_name, plot_file), dpi=600, format='svg')


'''--------time takes-----------'''
plt.figure()
plt.title('Testing Time Taken vs Communication rounds')
r = []
for i in range(len(f)):
    r_i = f[i].loc[:, 'Time Taken(s)']
    r.append(r_i)

for i in range(len(r)):
    plt.plot(range(len(r[i])), r[i], linewidth=0.8, color=colors[i], label=labels[i])

plt.legend(loc='upper left')
plt.ylabel('Test_time')
plt.xlabel('Communication Rounds')
plt.savefig('D:/Project-python/Results_all/{}/{}/{}/time_taken.svg'.format(dataset_name, model_name, plot_file), dpi=600, format='svg')


# '''-------------train loss---------------'''
# plt.figure()
# plt.title('Train Loss vs Communication rounds')
# r = []
# for i in range(len(f)):
#     r_i = f[i].loc[:, 'train_loss']
#     r.append(r_i)
#
# for i in range(len(r)):
#     plt.plot(range(len(r[i])), r[i], linewidth=0.8, color=colors[i], label=labels[i])
#
# # plt.ylim(0.5, 0.8)
# plt.legend(loc='upper right')
# plt.ylabel('Train_Loss')
# plt.xlabel('Communication Rounds')
# plt.savefig('D:/Project-python/Results_all/{}/{}/{}/train_loss.svg'.format(dataset_name, model_name, plot_file), dpi=600, format='svg')
#
#
# '''-------------test loss-------------'''
# plt.figure()
# plt.title('Test Loss vs Communication rounds')
# r = []
# for i in range(len(f)):
#     r_i = f[i].loc[:, 'test_loss']
#     r.append(r_i)
#
# for i in range(len(r)):
#     plt.plot(range(len(r[i])), r[i], linewidth=0.8, color=colors[i], label=labels[i])
#
# # plt.ylim(0.5, 0.8)
# plt.legend(loc='upper right')
# plt.ylabel('Test_Loss')
# plt.xlabel('Communication Rounds')
# plt.savefig('D:/Project-python/Results_all/{}/{}/{}/test_loss.svg'.format(dataset_name, model_name, plot_file), dpi=600, format='svg')
