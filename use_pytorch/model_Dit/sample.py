import shutil
import torch
import numpy as np
import os
from use_pytorch.tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from use_pytorch.model_Dit.utils_train import get_model, make_dataset
from use_pytorch.lib import round_columns
from use_pytorch import lib


def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)


def sample(
        # parent_dir,
        # real_data_path='data/abalonel',
        # flag_unnorm,
        # client_dataset_unnorm,
        client_dataset,
        min_class,
        batch_size=32,
        sampling_rate,
        big,
        num_samples=0,
        model_type='FTTrmlp',
        model_params=None,
        model_path=None,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        T_dict=None,
        num_numerical_features=0,
        disbalance=None,  # None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=0,
        change_val=False
):
    num_samples = big*sampling_rate
    # parent_dir = os.path.normpath(parent_dir)
    model_path = os.path.join(model_path, 'model.pt')
    num_numerical_features = client_dataset.shape[-1] - 1

    feature_idx = client_dataset.shape[-1] - 1
    # 归一化的数据集
    client_data = client_dataset

    Smin = client_data[client_data[feature_idx] == min_class]
    train_data = Smin.iloc[:, 0:Smin.shape[1] - 1].values
    train_label = Smin.iloc[:, -1].values.astype(int)

    model_params['is_y_cond'] = True
    model_params['num_classes'] = 2
    rtdl_params = {
        'd_layers': [256, 256, ],
        'dropout': 0.0
    }
    model_params['rtdl_params'] = rtdl_params
    T_dict = {}
    T_dict['seed'] = 0
    T_dict['normalization'] = 'quantile'  # "quantile"
    T_dict['num_nan_policy'] = None
    T_dict['cat_nan_policy'] = None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = None
    T_dict['y_policy'] = "default"
    # zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)
    D = make_dataset(
        # real_data_path,
        train_data,  # train_data
        train_label,  # train_label
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(D.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes('train')
    )

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model,  # 模型！！
        num_timesteps=num_timesteps,
        gaussian_loss_type=gaussian_loss_type,
        scheduler=scheduler, device=device
    )

    diffusion.to(device)
    diffusion.eval()

    _, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)
    # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
    if disbalance == 'fix':
        empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)

    elif disbalance == 'fill':
        ix_major = empirical_class_dist.argmax().item()
        val_major = empirical_class_dist[ix_major].item()
        x_gen, y_gen = [], []
        for i in range(empirical_class_dist.shape[0]):
            if i == ix_major:
                continue
            distrib = torch.zeros_like(empirical_class_dist)
            distrib[i] = 1
            num_samples = val_major - empirical_class_dist[i].item()
            x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False)
            x_gen.append(x_temp)
            y_gen.append(y_temp)

        x_gen = torch.cat(x_gen, dim=0)
        y_gen = torch.cat(y_gen, dim=0)

    else:
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), min_class, ddim=False)

    # try:
    # except FoundNANsError as ex:
    #     print("Found NaNs during sampling!")
    #     loader = lib.prepare_fast_dataloader(D, 'train', 8)
    #     x_gen = next(loader)[0]
    #     y_gen = torch.multinomial(
    #         empirical_class_dist.float(),
    #         num_samples=8,
    #         replacement=True
    #     )
    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

    ###
    # X_num_unnorm = X_gen[:, :num_numerical_features]
    # lo = np.percentile(X_num_unnorm, 2.5, axis=0)
    # hi = np.percentile(X_num_unnorm, 97.5, axis=0)
    # idx = (lo < X_num_unnorm) & (hi > X_num_unnorm)
    # X_gen = X_gen[np.all(idx, axis=1)]
    # y_gen = y_gen[np.all(idx, axis=1)]
    ###

    num_numerical_features = num_numerical_features + int(D.is_regression and not model_params["is_y_cond"])

    X_num_ = X_gen
    if num_numerical_features < X_gen.shape[1]:
        # np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:])  # 保存离散特征（非数值特征）
        # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
        if T_dict['cat_encoding'] == 'one-hot':
            X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1],
                                                            X_num_[:, num_numerical_features:])
        X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

    if num_numerical_features_ != 0:
        # _, normalize = lib.normalize({'train' : X_num_real}, T_dict['normalization'], T_dict['seed'], True)
        # np.save(os.path.join(parent_dir, 'X_num_unnorm'), X_gen[:, :num_numerical_features])  # 未标准化数据
        X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
        X_num = X_num_[:, :num_numerical_features]

        # X_num_real = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True)
        X_num_real = train_data
        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print("Discrete cols:", disc_cols)
        if model_params['num_classes'] == 0:
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if len(disc_cols):
            X_num = round_columns(X_num_real, X_num, disc_cols)

    # if num_numerical_features != 0:
    #     print("Num shape: ", X_num.shape)
    #     np.save(os.path.join(parent_dir, 'X_num'.format(res_name)), X_num)
    # if num_numerical_features < X_gen.shape[1]:
    #     np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)
    # np.save(os.path.join(parent_dir, '{}_y_gen'.format(res_name)), y_gen)

    return X_num, y_gen


# def save_file(parent_dir, config_path):
#     try:
#         dst = os.path.join(parent_dir)
#         os.makedirs(os.path.dirname(dst), exist_ok=True)
#         shutil.copyfile(os.path.abspath(config_path), dst)
#     except shutil.SameFileError:
#         pass

