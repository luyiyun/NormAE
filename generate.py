import os
import json

import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as data
from sklearn.decomposition import FastICA
from sklearn.feature_selection import f_classif

from datasets import get_demo_data, get_metabolic_data
from transfer import Normalization


def generate(
    models, data_loader, no_be_num, bs=64, nw=12,
    device=torch.device('cuda:0')
):
    '''
    data_loaders: Dataset对象或Dataloader对象，如果是Dataset则会利用实例化
        时的num_workers和batch_size来将其转换成一个Dataloader对象，可以输入
        多个；

    return：
        It's generator，the element is dict，the keys are "recons_no_batch、
        recons_all、original_x、ys”, the values are ndarrays
    '''
    for m in models.values():
        m.eval()
    if isinstance(data_loader, data.Dataset):
        data_loader = data.DataLoader(
            data_loader, batch_size=bs, num_workers=nw
        )
    x_recon, x_ori, x_recon_be, ys, codes = [], [], [], [], []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(data_loader, 'Transform batch: '):
            x_ori.append(batch_x)
            ys.append(batch_y)
            batch_x = batch_x.to(device, torch.float)
            hidden = models['encoder'](batch_x)
            codes.append(hidden.clone().detach())
            x_recon_be.append(models['decoder'](hidden))
            hidden[:, no_be_num:] = 0
            batch_x_recon = models['decoder'](hidden)
            x_recon.append(batch_x_recon)
    res = {
        'recons_no_batch': torch.cat(x_recon),
        'recons_all': torch.cat(x_recon_be),
        'original_x': torch.cat(x_ori), 'ys': torch.cat(ys),
        'codes': torch.cat(codes)
    }
    for k, v in res.items():
        if v is not None:
            res[k] = v.detach().cpu().numpy()
    return res



def generate_ica(
    models, data_loader, no_be_num=None, bs=64, nw=12,
    device=torch.device('cuda:0'),
):
    '''
    data_loaders: Dataset对象或Dataloader对象，如果是Dataset则会利用实例化
        时的num_workers和batch_size来将其转换成一个Dataloader对象，可以输入
        多个；
    no_be_num: 如果None，则表示对所有codes进行ICA，如果不是None，则其表示的前
        几个维度的codes是非批次效应编码，只会对其之后的编码进行ICA;

    return：
        It's generator，the element is dict，the keys are "recons_no_batch、
        recons_all、original_x、ys”, the values are ndarrays
    '''
    for m in models.values():
        m.eval()
    if isinstance(data_loader, data.Dataset):
        data_loader = data.DataLoader(
            data_loader, batch_size=bs, num_workers=nw
        )
    x_recon, x_ori, x_recon_be, ys, codes = [], [], [], [], []
    # 先把encode部分完成并处理
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
    for k, v in res.items():
        if v is not None:
            res[k] = v.detach().cpu().numpy()
    # 进行ICA
    print('FastICA beginning!!')
    if no_be_num is None:
        ica_use_data = res['codes']
    else:
        ica_use_data = res['codes'][:, no_be_num:]
    ica_estimator = FastICA()
    transfered_data = ica_estimator.fit_transform(ica_use_data)
    # 使用标签对ICA分解后的数据进行筛选，选择没有意义的成分
    _, pvals = f_classif(transfered_data, res['ys'][:, 1])
    mask_sig = pvals < 0.05
    print('total %d codes, significated %d, their indices are:'
          % (len(mask_sig), mask_sig.sum()))
    print(np.argwhere(mask_sig).squeeze())
    transfered_data[:, mask_sig] = 0
    # ICA逆转换回来
    filtered_data = ica_estimator.inverse_transform(transfered_data)
    if no_be_num is not None:
        filtered_data = np.concatenate(
            [res['codes'][:, :no_be_num], filtered_data], axis=1
        )
    filtered_data = data.TensorDataset(torch.tensor(filtered_data))
    filtered_data = data.DataLoader(
        filtered_data, batch_size=bs, num_workers=nw)
    # decode部分
    with torch.no_grad():
        for hidden, in tqdm(filtered_data, 'decode: '):
            hidden = hidden.to(device, torch.float)
            batch_x_recon = models['decoder'](hidden)
            x_recon.append(batch_x_recon)
    res['recons_no_batch'] = torch.cat(x_recon).detach().cpu().numpy()
    return res


def main():
    from config import Config
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('save', help="保存的结果的文件夹")
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
    pre_transfer = Normalization(save_json['data_norm'])
    if save_json['task'] == 'metabolic':
        subject_dat, qc_dat = get_metabolic_data(
            Config.metabolic_x_file, Config.metabolic_y_file,
            pre_transfer=pre_transfer
        )
    elif save_json['task'] == 'demo':
        subject_dat, qc_dat = get_demo_data(
            Config.demo_sub_file, Config.demo_qc_file, pre_transfer
        )

    # ----- 读取训练好的模型 -----
    models = os.path.join(task_path, 'models.pth')
    models = torch.load(models)

    # ----- 得到生成的数据 -----
    print('不使用ICA')
    all_res = generate(
        models, data.ConcatDataset([subject_dat, qc_dat]),
        save_json['no_batch_num'], bs=save_json['batch_size'],
        nw=save_json['num_workers'], device=torch.device('cuda:0')
    )
    # ----- 保存 -----
    for k, v in all_res.items():
        np.savetxt(os.path.join(task_path, 'all_res_%s.txt' % k), v)
    print('')


    # ----- 得到生成的数据 -----
    print('使用ICA')
    all_res = generate_ica(
        models, data.ConcatDataset([subject_dat, qc_dat]),
        save_json['no_batch_num'], bs=save_json['batch_size'],
        nw=save_json['num_workers'], device=torch.device('cuda:0')
    )
    # ----- 保存 -----
    # 原理上，也经过了检查，这里的结果出了recons_no_batch和上面的是一样的，就
    #   不进行保存了。
    for k, v in all_res.items():
        if k == 'recons_no_batch':
            np.savetxt(
                os.path.join(task_path, 'all_res_%s_ica.txt' % k), v)
    print('')


if __name__ == '__main__':
    main()
