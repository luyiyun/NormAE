import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.utils.data as data
from sklearn.decomposition import FastICA
from sklearn.feature_selection import f_classif

from datasets import get_demo_data, get_metabolic_data, ConcatData
from transfer import Normalization


def generate(
    models, data_loader, no_be_num=None, bs=64, nw=12,
    device=torch.device('cuda:0'), verbose=True, ica=True
):
    '''
    data_loaders: Dataset对象或Dataloader对象
    no_be_num: 如果None，则表示对所有codes进行ICA，如果不是None，则其表示的前
        几个维度的codes是非批次效应编码，只会对其之后的编码进行ICA;

    return：
        It's generator，the element is dict，the keys are "recons_no_batch、
        recons_all、original_x、ys”, the values aredataframe.
    '''
    for m in models.values():
        m.eval()
    if isinstance(data_loader, data.Dataset):
        data_loader = data.DataLoader(
            data_loader, batch_size=bs, num_workers=nw
        )
    x_recon, x_recon_ica, x_ori, x_recon_be, ys, codes = [], [], [], [], [], []
    # 先把encode部分完成
    if verbose:
        print('----- encoding -----')
    with torch.no_grad():
        if verbose:
            iterator = tqdm(data_loader, 'encoder: ')
        else:
            iterator = data_loader
        for batch_x, batch_y in iterator:
            x_ori.append(batch_x)
            ys.append(batch_y)
            batch_x = batch_x.to(device, torch.float)
            hidden = models['encoder'](batch_x)[-1]
            codes.append(hidden)
            x_recon_be.append(models['decoder'](hidden)[-1])
    res = {
        'original_x': torch.cat(x_ori), 'ys': torch.cat(ys),
        'codes':torch.cat(codes), 'recons_all': torch.cat(x_recon_be),
    }
    for k, v in res.items():
        if v is not None:
            if k == 'ys':
                res[k] = pd.DataFrame(
                    v.detach().cpu().numpy(),
                    index=data_loader.dataset.Y_df.index,
                    columns=data_loader.dataset.Y_df.columns
                )
            elif k != 'codes':
                res[k] = pd.DataFrame(
                    v.detach().cpu().numpy(),
                    index=data_loader.dataset.X_df.index,
                    columns=data_loader.dataset.X_df.columns
                )
            else:
                res[k] = pd.DataFrame(
                    v.detach().cpu().numpy(),
                    index=data_loader.dataset.X_df.index,
                )

    # 将codes的batch effect部分使用其均值来替代
    if verbose:
        print('----- decoding without ICA -----')
    use_data = res['codes'].values.copy()
    use_data[:, no_be_num:] = use_data[:, no_be_num].mean(axis=0)
    use_data = data.TensorDataset(torch.tensor(use_data))
    use_data = data.DataLoader(use_data, batch_size=bs, num_workers=nw)
    # decode部分
    if verbose:
        iterator = tqdm(use_data, 'decode without ICA: ')
    else:
        iterator = use_data
    with torch.no_grad():
        for hidden, in iterator:
            hidden = hidden.to(device, torch.float)
            batch_x_recon = models['decoder'](hidden)[-1]
            x_recon.append(batch_x_recon)
    res['recons_no_batch'] = pd.DataFrame(
        torch.cat(x_recon).detach().cpu().numpy(),
        index=data_loader.dataset.X_df.index,
        columns=data_loader.dataset.X_df.columns
    )

    # 进行ICA
    if ica:
        if verbose:
            print('----- decoding with ICA -----')
            print('FastICA beginning!!')
        if no_be_num is None:
            ica_use_data = res['codes'].values
        else:
            ica_use_data = res['codes'].values[:, no_be_num:]
        ica_estimator = FastICA()
        transfered_data = ica_estimator.fit_transform(ica_use_data)
        # 使用标签对ICA分解后的数据进行筛选，选择没有意义的成分
        _, pvals = f_classif(transfered_data, res['ys'].values[:, 1])
        mask_sig = pvals < 0.05
        if verbose:
            print('total %d codes, significated %d, their indices are:'
                % (len(mask_sig), mask_sig.sum()))
            print(np.argwhere(mask_sig).squeeze())
        transfered_data[:, mask_sig] = 0
        # ICA逆转换回来
        filtered_data = ica_estimator.inverse_transform(transfered_data)
        if no_be_num is not None:
            filtered_data = np.concatenate(
                [res['codes'].values[:, :no_be_num], filtered_data], axis=1
            )
        filtered_data = data.TensorDataset(torch.tensor(filtered_data))
        filtered_data = data.DataLoader(
            filtered_data, batch_size=bs, num_workers=nw)
        # decode部分
        if verbose:
            iterator = tqdm(filtered_data, 'decode with ICA: ')
        else:
            iterator = filtered_data
        with torch.no_grad():
            for hidden, in iterator:
                hidden = hidden.to(device, torch.float)
                batch_x_recon = models['decoder'](hidden)[-1]
                x_recon_ica.append(batch_x_recon)
        res['recons_no_batch_ica'] = pd.DataFrame(
            torch.cat(x_recon_ica).detach().cpu().numpy(),
            index=data_loader.dataset.X_df.index,
            columns=data_loader.dataset.X_df.columns
        )
    return res


def generate_forAE(
    models, data_loader, no_be_num=None, bs=64, nw=12,
    device=torch.device('cuda:0'),
):
    '''
    针对于AutoEncoder的generate

    data_loaders: Dataset对象或Dataloader对象

    return：
        It's generator，the element is dict，the keys are "recons_no_batch、
        recons_all、original_x、ys”, the values aredataframe.
    '''
    for m in models.values():
        m.eval()
    if isinstance(data_loader, data.Dataset):
        data_loader = data.DataLoader(
            data_loader, batch_size=bs, num_workers=nw
        )
    x_recon_aov, x_recon_aovall, x_ori, x_recon_be, ys, codes = \
        [], [], [], [], [], []
    # 先把encode部分和重建的部分完成并处理
    with torch.no_grad():
        for batch_x, batch_y in tqdm(data_loader, 'encoder: '):
            x_ori.append(batch_x)
            ys.append(batch_y)
            batch_x = batch_x.to(device, torch.float)
            hidden = models['encoder'](batch_x)
            codes.append(hidden)
            x_recon_be.append(models['decoder'](hidden))
    res = {
        'original_x': torch.cat(x_ori), 'ys': torch.cat(ys),
        'codes':torch.cat(codes), 'recons_all': torch.cat(x_recon_be),
    }
    # 把samples name和meta names写上
    for k, v in res.items():
        if v is not None:
            if k == 'ys':
                res[k] = pd.DataFrame(
                    v.detach().cpu().numpy(),
                    index=data_loader.dataset.Y_df.index,
                    columns=data_loader.dataset.Y_df.columns
                )
            elif k != 'codes':
                res[k] = pd.DataFrame(
                    v.detach().cpu().numpy(),
                    index=data_loader.dataset.X_df.index,
                    columns=data_loader.dataset.X_df.columns
                )
            else:
                res[k] = pd.DataFrame(
                    v.detach().cpu().numpy(),
                    index=data_loader.dataset.X_df.index,
                )

    # 使用卡方检验来对100个编码特征进行分析，选择出p值最小的no_be_num个，将其
    #   去除，然后重建
    print('AOV testing begin!!')
    be_num = res['codes'].shape[1] - no_be_num
    aov_use_data = res['codes'].values.copy()
    _, pvals = f_classif(aov_use_data, res['ys']['class'])
    idxs = np.argpartition(pvals, be_num)[:be_num]
    print('min %d codes: %s, the indices are %s' %
          (be_num, str(pvals[idxs]), str(idxs)))
    aov_use_data[:, idxs] = aov_use_data[:, idxs].mean(axis=0)
    aov_use_data = data.TensorDataset(torch.tensor(aov_use_data))
    aov_use_data = data.DataLoader(
        aov_use_data, batch_size=bs, num_workers=nw)
    # decoder部分
    with torch.no_grad():
        for hidden, in tqdm(aov_use_data, 'aov decode: '):
            hidden = hidden.to(device, torch.float)
            batch_x_recon = models['decoder'](hidden)
            x_recon_aov.append(batch_x_recon)
    res['recons_nobe_aov'] = pd.DataFrame(
        torch.cat(x_recon_aov).detach().cpu().numpy(),
        index=data_loader.dataset.X_df.index,
        columns=data_loader.dataset.X_df.columns
    )


    # 使用卡方检验来对100个编码特征进行分析，选择出所有p值小于0.05的codes，置
    #   为0
    aov_use_data2 = res['codes'].values.copy()
    bool_index = pvals < 0.05
    print('%d codes significate!' % sum(bool_index))
    aov_use_data2[:, bool_index] = aov_use_data2[:, bool_index].mean(axis=0)
    aov_use_data2 = data.TensorDataset(torch.tensor(aov_use_data2))
    aov_use_data2 = data.DataLoader(
        aov_use_data2, batch_size=bs, num_workers=nw)
    # decoder部分
    with torch.no_grad():
        for hidden, in tqdm(aov_use_data2, 'aov sig decode: '):
            hidden = hidden.to(device, torch.float)
            batch_x_recon = models['decoder'](hidden)
            x_recon_aovall.append(batch_x_recon)
    res['recons_nobe_aovall'] = pd.DataFrame(
        torch.cat(x_recon_aovall).detach().cpu().numpy(),
        index=data_loader.dataset.X_df.index,
        columns=data_loader.dataset.X_df.columns
    )

    return res


def main():
    from config import Config
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('save', help="保存的结果的文件夹")
    parser.add_argument('--autoencoder', action='store_true',
                        help="面对的结果是AE得到的，要使用generate_forAE")
    args = parser.parse_args()
    task_path = args.save

    print('开始使用训练好的模型去除批次效应')
    # config
    with open(os.path.join(task_path, 'config.json'), 'r') as f:
        save_json = json.load(f)
    if 'no_batch_num' not in save_json:
        save_json['no_batch_num'] = None
    print(save_json)

    # ----- 读取数据 -----
    pre_transfer = Normalization(save_json['data_normalization'])
    if save_json['task'] == 'demo':
        subject_dat, qc_dat = get_demo_data(
            Config.demo_sub_file, Config.demo_qc_file, pre_transfer
        )
    else:
        subject_dat, qc_dat = get_metabolic_data(
            Config.metabolic_x_files[save_json['task']],
            Config.metabolic_y_files[save_json['task']],
            pre_transfer=pre_transfer
        )

    # ----- 读取训练好的模型 -----
    models = os.path.join(task_path, 'models.pth')
    models = torch.load(models)

    if args.autoencoder:
        print('For AutoEncoder')
        all_res = generate_forAE(
            models, ConcatData(subject_dat, qc_dat),
            save_json['no_batch_num'], bs=save_json['batch_size'],
            nw=save_json['num_workers'], device=torch.device('cuda:0')
        )
        for k, v in all_res.items():
            if k not in ['ys', 'codes']:
                v, _ = pre_transfer.inverse_transform(v, None)
            v.to_csv(os.path.join(task_path, 'AE_%s.csv' % k))
        print('')
    else:
        # ----- 得到生成的数据 -----
        all_res = generate(
            models, ConcatData(subject_dat, qc_dat),
            save_json['no_batch_num'], bs=save_json['batch_size'],
            nw=save_json['num_workers'], device=torch.device('cuda:0')
        )
        # ----- 保存 -----
        for k, v in all_res.items():
            if k not in ['ys', 'codes']:
                v, _ = pre_transfer.inverse_transform(v, None)
            v.to_csv(os.path.join(task_path, 'all_res_%s.csv' % k))
        print('')


if __name__ == '__main__':
    main()
