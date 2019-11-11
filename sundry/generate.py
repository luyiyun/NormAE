import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.decomposition import FastICA
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

from datasets import get_demo_data, get_metabolic_data, ConcatData
from transfer import Normalization
from metrics import Loss
from train import BatchEffectTrainer


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
    # aov_use_data[:, idxs] = aov_use_data[:, idxs].mean(axis=0)
    aov_use_data[:, idxs] = 0
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
    # aov_use_data2[:, bool_index] = aov_use_data2[:, bool_index].mean(axis=0)
    aov_use_data2[:, bool_index] = 0
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
    print(save_json)

    # ----- 读取数据 -----
    pre_transfer = Normalization(save_json['data_normalization'])
    if save_json['task'] == 'demo':
        all_dat = get_demo_data(
            Config.demo_sub_file, Config.demo_qc_file, pre_transfer,
            sub_qc_split=False
        )
    else:
        all_dat = get_metabolic_data(
            Config.metabolic_x_files[save_json['task']],
            Config.metabolic_y_files[save_json['task']],
            pre_transfer=pre_transfer, sub_qc_split=False
        )

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
        #  norm_new = Normalization(save_json['data_normalization'])
        trainer = BatchEffectTrainer(
            all_dat.num_features, save_json['bottle_num'],
            save_json['be_num'], batch_label_num=all_dat.num_batch_labels,
            lrs=save_json['ae_disc_lr'], bs=save_json['batch_size'],
            nw=save_json['num_workers'], epoch=save_json['epoch'],
            device=torch.device('cuda:0'), l2=save_json['l2'], clip_grad=True,
            ae_disc_train_num=save_json['ae_disc_train_num'],
            disc_weight=save_json['disc_weight'],
            label_smooth=save_json['label_smooth'],
            train_with_qc=save_json['train_data' ]== 'all',
            spectral_norm=save_json['spectral_norm'],
            schedual_stones=save_json['schedual_stones'],
            cls_leastsquare=save_json['cls_leastsquare'],
            order_losstype=save_json['order_losstype'],
            cls_order_bio_weight=save_json['cls_order_bio_weight'],
            use_batch_for_order=save_json['use_batch_for_order'],
            visdom_port=save_json['visdom_port'],
            decoder_hiddens=save_json['ae_units'],
            encoder_hiddens=save_json['ae_units'][::-1],
            disc_hiddens=save_json['disc_units'],
            early_stop=save_json['early_stop'],
            net_type=save_json['net_type'],
            resnet_bottle_num=save_json['resnet_bottle_num'],
            optimizer=save_json['optim'],
            denoise=save_json['denoise'],
            reconst_loss=save_json['reconst_loss'],
            disc_weight_epoch=save_json['disc_weight_epoch'],
            early_stop_check_num=save_json['early_stop_check_num'],
            #  disc_bn=save_json['disc_bn']
        )
        trainer.load_model(os.path.join(task_path, 'models.pth'))
        all_res = trainer.generate(
            all_dat, verbose=True, compute_qc_loss=False)

        # ----- 保存 -----
        for k, v in all_res.items():
            if k not in ['Ys', 'Codes']:
                #  v, _ = norm_new(v, None)
                v, _ = pre_transfer.inverse_transform(v, None)
                v = v.T  # 列为样本，行为变量，读取时比较快速
                v.index.name = 'meta.name'
            v.to_csv(os.path.join(task_path, '%s.csv' % k))
        print('')


if __name__ == '__main__':
    main()
