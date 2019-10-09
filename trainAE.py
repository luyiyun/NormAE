import os
import copy
import json
from itertools import chain

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib; matplotlib.use('Pdf')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from datasets import get_metabolic_data, get_demo_data, ConcatData
from networks import SimpleCoder, ResNet
from transfer import Normalization
import metrics as mm
from visual import VisObj, pca_for_dict, pca_plot
from generate import generate


class AutoEncoderTrainer:
    def __init__(
        self, in_features, bottle_num, encoder_hiddens, decoder_hiddens,
        lr=0.01, bs=64, nw=6, epoch=100, device=torch.device('cuda:0'),
        schedual_stones=[2000], train_with_qc=True, net_type='simple',
        denoise=0.0
    ):
        '''
        in_features: the number of input features;
        bottle_num: the number of bottle neck layer's units in AE;
        lr: the learning rates, the first is for reconstruction of AE, the second
            is for discriminator, if it's float, they are same;
        bs: train batch size;
        nw: the number of workers;
        epoch: the number of training epochs;
        device: GPU or CPU;
        schedual_stones: the epoch of lrs multiply 0.1;
        train_with_qc: if true, the dataset of training is concatenated data of
            subject and qc;
        '''

        # 得到3个模型
        if net_type == 'simple':
            self.models = {
                'encoder': SimpleCoder(
                    [in_features] + encoder_hiddens + [bottle_num]).to(device),
                'decoder': SimpleCoder(
                    [bottle_num] + decoder_hiddens + [in_features]).to(device),
            }
        else:
            self.models = {
                'encoder': ResNet(
                    [in_features] + encoder_hiddens + [bottle_num],
                    resnet_bottle_num
                ).to(device),
                'decoder': ResNet(
                    [bottle_num] + decoder_hiddens + [in_features],
                    resnet_bottle_num
                ).to(device),
            }

        self.criterion = nn.MSELoss()

        if optimizer == 'rmsprop':
            optimizer_obj = optim.RMSprop
        elif optimizer == 'adam':
            optimizer_obj = optim.Adam
        else:
            raise ValueError
        self.optimizer = optimizer_obj(chain(
            self.models['encoder'].parameters(),
            self.models['decoder'].parameters()
        ), lr=lr)

        self.schedual = optim.lr_scheduler.MultiStepLR(
            self.optimizer, schedual_stones, gamma=0.1)

        # 初始化结果记录
        self.history = {"recon_loss": [], 'qc_loss': [], 'qc_distance': []}

        # 属性化
        self.epoch = epoch
        self.bs = bs
        self.nw = nw
        self.train_with_qc = train_with_qc
        self.device = device
        self.denoise = denoise

        # 可视化工具
        self.visobj = VisObj()

    def fit(self, datas):
        ''' datas是多个Dataset对象的可迭代对象 '''
        # 将Dataset对象变成Dataloader对象
        train_data = data.ConcatDataset([datas['subject'], datas['qc']]) if \
            self.train_with_qc else datas['subject']
        dataloaders = {
            'train': data.DataLoader(
                train_data, batch_size=self.bs, num_workers=self.nw,
                shuffle=True
            ),
            'qc': data.DataLoader(
                datas['qc'], batch_size=self.bs, num_workers=self.nw)
        }

        # 开始进行多个epoch训练
        for _ in tqdm(range(self.epoch), 'Epoch: '):
            ## train phase
            # 实例化3个loss对象，用于计算epoch loss
            recon_loss_obj = mm.Loss()
            # 训练时需要将模型更改至训练状态
            for model in self.models.values():
                model.train()
            # 循环每个batch进行训练
            for batch_x, _ in tqdm(dataloaders['train'], 'Train Batch: '):
                batch_x = batch_x.to(self.device, torch.float)
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    code = self.models['encoder'](batch_x)
                    recon = self.models['decoder'](code)
                    recon_loss = self.criterion(recon, batch_x)
                    recon_loss.backward()
                self.optimizer.step()
                # 记录loss(这样，就算update了多次loss，也只记录最后一次)
                recon_loss_obj.add(recon_loss, batch_x.size(0))
            # 看warning说需要把scheduler的step放在optimizer.step之后使用
            self.schedual.step()

            # 记录epoch loss
            self.history['recon_loss'].append(recon_loss_obj.value())

            # 可视化epoch loss
            self.visobj.add_epoch_loss(
                winname='recon_losses',
                recon=self.history['recon_loss'][-1],
            )

            # ## valid phase

            # # 使用当前训练的模型去得到去批次结果
            # all_data = ConcatData(datas['subject'], datas['qc'])
            # all_reses_dict = generate(
            #     self.models, all_data, no_be_num=self.no_be_num,
            #     device=self.device, bs=self.bs, nw=self.nw
            # )
            # # 对数据进行对应的pca
            # subject_pca, qc_pca = pca_for_dict(all_reses_dict)
            # # plot pca
            # pca_plot(subject_pca, qc_pca)
            # # display in visdom
            # self.visobj.vis.matplot(plt, win='PCA', opts={'title': 'PCA'})
            # plt.close()

        return self.models, self.history


def check_update_dirname(dirname, indx=0):
    if os.path.exists(dirname):
        if indx > 0:
            dirname = dirname[:-len(str(indx))]
        indx += 1
        dirname = dirname + str(indx)
        dirname = check_update_dirname(dirname, indx)
    else:
        os.makedirs(dirname)
    return dirname


def main():
    from config import Config

    # config
    config = Config()
    config.show()

    # ----- 读取数据 -----
    pre_transfer = Normalization(config.args.data_norm)
    if config.args.task == 'demo':
        subject_dat, qc_dat = get_demo_data(
            config.demo_sub_file, config.demo_qc_file, pre_transfer
        )
    else:
        subject_dat, qc_dat = get_metabolic_data(
            config.metabolic_x_files[config.args.task],
            config.metabolic_y_files[config.args.task],
            pre_transfer=pre_transfer
        )
    datas = {'subject': subject_dat, 'qc': qc_dat}

    # ----- 训练网络 -----
    trainer = AutoEncoderTrainer(
        subject_dat.num_features, config.args.bottle_num,
        lr=config.args.ae_disc_lr[0],
        bs=config.args.batch_size, nw=config.args.num_workers,
        epoch=config.args.epoch, device=torch.device('cuda:0'),
        train_with_qc=config.args.train_data == 'all',
    )

    best_models, hist = trainer.fit(datas)
    print('')

    # 保存结果
    dirname = check_update_dirname(config.args.save)
    torch.save(best_models, os.path.join(dirname, 'models.pth'))
    pd.DataFrame(hist).to_csv(os.path.join(dirname, 'train.csv'))
    config.save(os.path.join(dirname, 'config.json'))


if __name__ == "__main__":
    main()
