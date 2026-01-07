from copy import deepcopy
import torch
import os
import numpy as np
from use_pytorch.tab_ddpm import GaussianMultinomialDiffusion
from use_pytorch.model_Dit.utils_train import get_model, make_dataset, update_ema
from use_pytorch import lib
import pandas as pd


class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=torch.device('cuda:1')):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] = [step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1


def train(
        # flag_unnorm,
        parent_dir,
        # client_dataset_unnorm,
        client_dataset,
        min_class,
        steps=1000,
        lr=0.001,
        weight_decay=1e-5,
        batch_size=32,
        model_type='FTTrmlp',
        model_params=None,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        T_dict=None,
        num_numerical_features=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=0,
        change_val=False
):
    parent_dir = os.path.normpath(parent_dir)
    feature_idx = client_dataset.shape[-1] - 1
    # 归一化的数据集
    print(client_dataset.columns)
    client_data = client_dataset
    # 设置模型的训练数据为 少数类样本
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
    T_dict['normalization'] = 'quantile'  # "quantile" minmax
    T_dict['num_nan_policy'] = None
    T_dict['cat_nan_policy'] = None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = None
    T_dict['y_policy'] = "default"
    # zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)

    dataset = make_dataset(
        train_data,  # train_data
        train_label,  # train_label
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )  # dataset: X_num(dict)--'train'(ndarray:n*m);

    K = np.array(dataset.get_category_sizes('train'))
    # 离散型特征处理成数字表示，对于每一个离散特征表示的大小进行计算（比如第一个特征0-8表示）
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    print(K)

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0  # 数值特征的数量
    d_in = np.sum(K) + num_numerical_features  # 注：离散特征和连续特征之和
    model_params['d_in'] = d_in
    print(d_in)

    print(model_params)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    model.to(device)

    # train_loader = lib.prepare_beton_loader(dataset, split='train', batch_size=batch_size)
    train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device
    )
    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))
