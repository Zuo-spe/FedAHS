import copy
import numpy as np
from clients import ClientsGroup, client


def aggrated_FedAvg(list_dicts_local_params, list_nums_local_data):

    fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
    for name_param in list_dicts_local_params[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        fedavg_global_params[name_param] = value_global_param
    return fedavg_global_params


def aggrated_FedAvg_with_R(list_dicts_local_params,
                            list_nums_local_data,
                            list_delta_R,
                            eps: float = 1e-3):

    assert len(list_dicts_local_params) == len(list_nums_local_data) == len(list_delta_R)

    raw_weights = []
    for Di, dRi in zip(list_nums_local_data, list_delta_R):
        raw = Di / float(eps + dRi)
        raw_weights.append(raw)

    raw_weights = np.array(raw_weights, dtype=float)
    weight_sum = raw_weights.sum()
    if weight_sum == 0:
        raw_weights = np.ones_like(raw_weights)
        weight_sum = raw_weights.sum()

    alphas = raw_weights / weight_sum

    fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
    for name_param in fedavg_global_params.keys():
        agg = 0.0
        for local_params, alpha in zip(list_dicts_local_params, alphas):
            agg = agg + local_params[name_param] * alpha
        fedavg_global_params[name_param] = agg

    return fedavg_global_params
