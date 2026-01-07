import csv
import time
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from collections import deque
from torch import optim
from torch import tensor
from Models import Mnist_2NN, Mnist_CNN, MLP, LR, tabular_2NN, SVM, MLPclassify
from compute_metrics import *
from aggregate_models import *
from aggregate_models import aggrated_FedAvg, aggrated_FedAvg_with_R

# Argument parsing
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAHS")
parser.add_argument('-g', '--gpu', type=str, default='1', help='gpu id to use(GPU is:0,CPU is：1)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=10, help='number of the clients')#客户端数量
parser.add_argument('-cf', '--cfraction', type=float, default=1,help='C fraction, 0 means 1 client, 1 means total clients')  # 每一轮参与客户端的比例
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')  # 客户端训练的次数
parser.add_argument('-B', '--batchsize', type=int, default=64, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='MLP(10)', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.1,help="learning rate, use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=300,help='number of communications')  # 这个对应结果communication epoch()
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-isReS', '--Resampling', type=int, default=1 ,help='whether use a Parallel reasmpling method in client data（1 use，0 not use）')
parser.add_argument('-n_KF_Split', '--n_kf', type=int, default=10, help='10-kf or 5-kf')
parser.add_argument('-dataset_name', '--data_name', type=str, default='abalone', help='dataset name')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def gaussian_mutation(xi, t, T, delta_R=None, Fmin=0.2, Fmax=5.0, gamma=5.0):
    """
    Adaptive Gaussian mutation.
    - Uses N(0, I) instead of Uniform.
    - Mutation strength decays with time and can be further adjusted by delta_R (optional).
    """
    # time-decay schedule: strong exploration early, weaker later
    Ft = Fmin + (Fmax - Fmin) * np.exp(-gamma * (t / max(T, 1)))
    if delta_R is not None:
        s = delta_R / (delta_R + 1e-6 + 0.1)  # 0~1
        Ft = Ft * (0.5 + 0.5 * s)
    z = np.random.normal(loc=0.0, scale=1.0, size=2)  # Gaussian noise
    new_xi = xi + Ft * z
    # keep in valid range
    new_xi = np.clip(new_xi, 1, 80)
    return new_xi


def crossover(x_best, x_i, CR, j_rand):
    # 初始化新解
    u_i = np.copy(x_i)
    for j in range(len(x_i)):
        if (np.random.rand() < CR) or (j == j_rand):
            u_i[j] = x_best[j]
    return u_i


def compute_delta_R_from_window(rate_deque):
    if rate_deque is None or len(rate_deque) < 2:
        return 0.0

    diffs = []
    for k in range(1, len(rate_deque)):
        diffs.append(np.linalg.norm(rate_deque[k] - rate_deque[k - 1], ord=2))
    return float(np.mean(diffs)) if diffs else 0.0


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    dataset_names = ['abalone','bean2','credit_card','htru','magic04','shuttle','statlog','wine','page_blocks0']

    K_window = 5
    client_rate_hist = {}
    for dataset_namei in range(len(dataset_names)):
        args['data_name'] = dataset_names[dataset_namei]
        print(f"Testing {dataset_names[dataset_namei]}")
        weighted_type = ['FedAHS']
        for w_i in range(len(weighted_type)):
            args['weighted'] = weighted_type[w_i]

            if args['data_name'] in dataset_names:
                args['cfraction'] = 0.2
                args['model_name'] = 'MLP(10)'
                args['alpha'] = 2.0
                args['seed'] = 10
            else:
                exit('Error: unrecognized dataset')

            for i_kf in range(args['n_kf']):
                model = None
                if args['data_name'] in ['abalone']:
                    model = MLP(input_size=8, net=10)
                elif args['data_name'] in ['bean2']:
                    model = MLP(input_size=16, net=10)
                elif args['data_name'] in ['CM1']:
                    model = MLP(input_size=21, net=10)
                elif args['data_name'] in ['credit_card']:
                    model = MLP(input_size=23, net=10)
                elif args['data_name'] in ['htru']:
                    model = MLP(input_size=8, net=10)
                elif args['data_name'] in ['magic04']:
                    model = MLP(input_size=10, net=10)
                elif args['data_name'] in ['shuttle']:
                    model = MLP(input_size=9, net=10)
                elif args['data_name'] in ['statlog']:
                    model = MLP(input_size=36, net=10)
                elif args['data_name'] in ['wine']:
                    model = MLP(input_size=11, net=10)
                elif args['data_name'] in ['page_blocks0']:
                    model = MLP(input_size=10, net=10)
                elif args['data_name'] in ['CSE-CIC-IDS2018']:
                    model = MLP(input_size=78, net=10)
                elif args['data_name'] in ['UNSW-NB15']:
                    model = MLP(input_size=43, net=10)
                else:
                    exit('Error: unrecognized model')

                if torch.cuda.device_count() > 1:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    model = torch.nn.DataParallel(model)
                model = model.to(dev)

                # Loss function: Cross entropy loss function
                loss_func = F.cross_entropy

                opti = optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=0.9)

                # Dataset processing - Divide into N clients, here call client. py
                myClients = ClientsGroup(args['data_name'], args['n_kf'], i_kf, args['IID'], args['Resampling'],
                                         args['num_of_clients'], args['alpha'], args['seed'], dev)
                testDataLoader = myClients.test_data_loader
                testset_labels = myClients.testset_label

                num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))
                print("num_in_comm:", num_in_comm)

                global_parameters = {}
                for key, var in model.state_dict().items():
                    global_parameters[key] = var.clone()
                #
                data_file_path = 'Result/Results_{}_S/{}/Test_W_(C{}_D{}_S{})'.format(
                    args['n_kf'],
                    args['data_name'],
                    args['num_of_clients'],
                    args['alpha'],
                    args['seed'])
                os.makedirs(data_file_path, exist_ok=True)

                results_file = 'KF_{}.csv'.format(i_kf)
                data_file = os.path.join(data_file_path, results_file)
                # 将数据写入文件中
                with open(data_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['communication_epoch', 'train_loss', 'test_loss', 'Training Time(s)', 'Gmean',
                                     'F1_score', 'AUC'])

                start_time = time.time()  # Calculate the time of each communication
                train_loss, train_accuracy = [], []
                test_loss = 0.0
                G_mean = 0
                F1_score = 0.0
                AUC = 0.0
                np.random.seed(50)

                # Initialize the 'client-side sampling rates' dictionary to store the sampling rate for each client
                client_sampling_rates = {}
                # Initialize the local_fitness dictionary to store the current optimal fitness for each client
                client_local_fitness = {}
                # Record the sampling rate of the previous round for calculating Δ R_i
                prev_sampling_rates = {}

                for i in tqdm(range(args['num_comm'])):
                    local_losses = []
                    print("communicate round {}".format(i + 1))

                    order = np.random.permutation(args['num_of_clients'])
                    clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
                    sum_parameters = None
                    list_nums_local_data = []
                    list_dicts_local_params = []
                    list_delta_R = []
                    all_client_aucs = []
                    num_pop = 0

                    for client in clients_in_comm:
                        #print("client is :", client)
                        list_nums_local_data.append(copy.deepcopy(myClients.clients_set[client]).Size_dataset)

                        if client not in client_sampling_rates:
                            Over_sr = np.random.uniform(10, 80)  # 初始过采样率
                            Under_sr = np.random.uniform(10, 80)  # 初始欠采样率
                            client_sampling_rates[client] = (Over_sr, Under_sr)  # 存储客户端的采样率

                        if client not in client_local_fitness:
                            client_local_fitness[client] = -np.inf

                        old_over, old_under = client_sampling_rates[client]
                        old_vec = np.array([old_over, old_under], dtype=float)

                        Over_sr, Under_sr = client_sampling_rates[client]

                        delta_R_est = None
                        if client in client_rate_hist and len(client_rate_hist[client]) >= 2:
                            delta_R_est = compute_delta_R_from_window(client_rate_hist[client])
                        mut_vec = gaussian_mutation(np.array([Over_sr, Under_sr], dtype=float), t=i, T=args['num_comm'],
                                                    delta_R=delta_R_est)

                        mut_vec = gaussian_mutation(np.array([Over_sr, Under_sr], dtype=float), t=i, T=args['num_comm'])
                        New_over_sr, New_under_sr = float(mut_vec[0]), float(mut_vec[1])

                        CR = 0.2
                        j_rand = np.random.randint(0, 2)
                        x_best = np.array(client_sampling_rates[client], dtype=float)
                        trial_vec = crossover(x_best, np.array([New_over_sr, New_under_sr], dtype=float), CR, j_rand)
                        New_over_sr, New_under_sr = float(trial_vec[0]), float(trial_vec[1])

                        #------training------
                        local_parameters, loss, trial_fitness = myClients.clients_set[client].localUpdate(args['epoch'],
                                                                                                 args['batchsize'],
                                                                                                 model,
                                                                                                 loss_func, opti,
                                                                                                 global_parameters, New_over_sr,
                                                                                                 New_under_sr)

                        if trial_fitness > client_local_fitness[client]:
                            client_sampling_rates[client] = (New_over_sr, New_under_sr)
                            client_local_fitness[client] = trial_fitness
                            new_vec = np.array([New_over_sr, New_under_sr], dtype=float)
                        else:
                            new_vec = old_vec

                        # ---------------- Sliding-window ΔR_i ----------------
                        if client not in client_rate_hist:
                            client_rate_hist[client] = deque(maxlen=K_window + 1)

                        client_rate_hist[client].append(new_vec.copy())
                        delta_R_i = compute_delta_R_from_window(client_rate_hist[client])
                        list_delta_R.append(delta_R_i)
                        prev_sampling_rates[client] = new_vec.copy()

                        # ---------------- Record local loss and parameters ----------------
                        local_losses.append(copy.deepcopy(loss))
                        list_dicts_local_params.append(copy.deepcopy(local_parameters))

                    half_comm = args['num_comm'] // 2
                    if args['weighted'] == 'FedAHS':
                        if i < half_comm:
                            global_parameters = aggrated_FedAvg(list_dicts_local_params,
                                                                list_nums_local_data,
                                                                )
                        else:
                            global_parameters = aggrated_FedAvg_with_R(
                                list_dicts_local_params,
                                list_nums_local_data,
                                list_delta_R,
                                eps=1e-3
                            )

                    else:
                        exit('Error: unrecognized weighted type')

                    loss_avg = sum(local_losses) / len(local_losses)
                    train_loss.append(loss_avg)

                    # --------after a communication, to testing on the model in server for showing the performance-------------
                    with torch.no_grad():
                        model.load_state_dict(global_parameters, strict=True)
                        batch_loss = []
                        num = 0
                        preds_sum = tensor([], device=dev)
                        labels_sum = tensor([], device=dev)
                        for data, label in testDataLoader:
                            data, label = data.to(dev), label.to(dev)
                            preds = model(data.float())
                            loss = loss_func(preds, label.long())
                            batch_loss.append(loss.item())
                            preds = torch.argmax(preds, dim=1)
                            preds_sum = torch.cat([preds_sum, preds], dim=0)
                            labels_sum = torch.cat([labels_sum, label], dim=0)

                        test_loss = sum(batch_loss) / len(batch_loss)
                        G_mean, F1_score, AUC = metrics_compute(preds_sum, labels_sum)

                    with open(data_file, 'a', encoding='utf-8', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [i + 1, train_loss[-1], test_loss, (time.time() - start_time), G_mean, F1_score, AUC])
