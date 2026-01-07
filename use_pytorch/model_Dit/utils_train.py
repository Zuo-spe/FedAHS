import numpy as np
import os
from use_pytorch import lib
from use_pytorch.tab_ddpm.modules import MLPDiffusion, ResNetDiffusion, FTTransfMLPDiffusion

def get_model(
    model_name,
    model_params,
    n_num_features,
    category_sizes
): 
    print(model_name)
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    elif model_name == 'FTTrmlp':
        model = FTTransfMLPDiffusion(**model_params)
    elif model_name == 'resnet':
        model = ResNetDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)

def make_dataset(
    X_num_t,
    y_t,
    T: lib.Transformations,
    num_classes: int,
    is_y_cond: bool,
    change_val: bool
):
    # classification
    X_cat, X_cat_t = None, None
    X_num = {}
    y = {}

    for split in ['train']:
        if X_num is not None:
            X_num[split] = X_num_t
        if not is_y_cond:
            X_cat_t = concat_y_to_X(X_cat_t, y_t)
        if X_cat is not None:
            X_cat[split] = X_cat_t
        y[split] = y_t

    # info = lib.load_json(os.path.join(data_path, 'info.json'))

    D = lib.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type='binclass',  # lib.TaskType(info['task_type']),
        n_classes=None  # info.get('n_classes')
    )

    if change_val:
        D = lib.change_val(D)
    
    return lib.transform_dataset(D, T, None)
