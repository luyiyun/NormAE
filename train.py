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
from networks import SimpleCoder, SmoothCERankLoss
from transfer import Normalization
import metrics as mm
from visual import VisObj, pca_for_dict, pca_plot
from generate import generate


class BatchEffectTrainer:
    def __init__(
        self, in_features, bottle_num, no_be_num, batch_label_num=None,
        lrs=0.01, bs=64, nw=6, epoch=100, device=torch.device('cuda:0'),
        l2=0.0, clip_grad=False, ae_disc_train_num=(1, 1),
        ae_disc_weight=(1.0, 1.0), supervise='both', label_smooth=0.2,
        train_with_qc=False, spectral_norm=False, schedual_stones=[2000],
        interconnect=False
    ):
        '''
        in_features: the number of input features;
        bottle_num: the number of bottle neck layer's units in AE;
        no_be_num: the number of bottle neck layers's units without batch effect
            information in AE;
        batch_label_num: the number of batchs, if supervise=rank, it isn't used;
        lrs: the learning rates, the first is for reconstruction of AE, the second
            is for discriminator, if it's float, they are same;
        bs: train batch size;
        nw: the number of workers;
        epoch: the number of training epochs;
        device: GPU or CPU;
        l2: the value of weight decay;
        clip_grad: whether to use gradient truncation;
        ae_disc_train_num: for one batch, the training number for AE and
            discriminator;
        ae_disc_weight: for reconstruction loss, the weights of reconstruction
            loss and discriminated loss;
        supervise: the type of discriminate;
        label_smooth: the label smooth parameter for disciminate;
        train_with_qc: if true, the dataset of training is concatenated data of
            subject and qc;
        spectral_norm: if true, use spectral normalization for all linear layers;
        schedual_stones: the epoch of lrs multiply 0.1;
        interconnect: if true, the connection of hiddens between encoder and
            decoder will be build;
        '''

        # 得到两个loss
        self.criterions = {'reconstruction': nn.MSELoss()}
        # according to supervise, choose classification criterion
        if supervise == 'both':
            logit_dim = batch_label_num + 1
            self.criterions['adversarial'] = SmoothCERankLoss(label_smooth)
        elif supervise == 'rank':
            logit_dim = 1
            self.criterions['adversarial'] = SmoothCERankLoss(ce_w=0.0)
        elif supervise == 'cls':
            logit_dim = batch_label_num
            self.criterions['adversarial'] = SmoothCERankLoss(
                label_smooth, rank_w=0.0)
        else:
            raise ValueError(
                "supervise must be one of 'both', 'rank' and 'cls'")
        self.supervise = supervise


        # 得到3个模型
        self.models = {
            'encoder': SimpleCoder(
                [in_features, 300, 300, 300, bottle_num], lrelu=True,
                last_act=None, norm=nn.BatchNorm1d, dropout=None,
                spectral_norm=spectral_norm
            ).to(device),
            'decoder': SimpleCoder(
                [bottle_num, 300, 300, 300, in_features], lrelu=True,
                last_act=None, norm=nn.BatchNorm1d, dropout=None,
                spectral_norm=spectral_norm
            ).to(device),
            'discriminator': SimpleCoder(
                [no_be_num, 300, 300, logit_dim], lrelu=False,
                norm=nn.BatchNorm1d, last_act=None,
                spectral_norm=spectral_norm, return_hidden=False
            ).to(device)
        }
        self.num_encoder_layers = len(self.models['encoder'].layers)

        # 得到两个optim
        if not isinstance(lrs, (tuple, list)):
            lrs = [lrs] * 2
        self.optimizers = {
            'autoencode': optim.Adam(
                chain(
                    self.models['encoder'].parameters(),
                    self.models['decoder'].parameters()
                ),
                lr=lrs[0], weight_decay=l2
            ),
            'discriminate': optim.Adam(
                self.models['discriminator'].parameters(), lr=lrs[1],
                weight_decay=l2
            )
        }
        self.scheduals = {
            'discriminate': optim.lr_scheduler.MultiStepLR(
                self.optimizers['discriminate'], schedual_stones, gamma=0.1
            ),
            'autoencode': optim.lr_scheduler.MultiStepLR(
                self.optimizers['autoencode'], schedual_stones, gamma=0.1
            )
        }

        # 初始化结果记录
        self.history = {('recon%d_loss' % i): []
                        for i in range(self.num_encoder_layers)}
        self.history.update({
            'discriminate_loss_train': [], 'discriminate_loss_fixed': []
        })

        # 属性化
        self.epoch = epoch
        self.device = device
        self.clip_grad = clip_grad
        self.autoencode_train_num, self.discriminate_train_num = \
            ae_disc_train_num
        self.ae_weight, self.disc_weight = ae_disc_weight
        self.bs = bs
        self.nw = nw
        self.no_be_num = no_be_num
        self.supervise = supervise
        self.train_with_qc = train_with_qc
        self.interconnect = interconnect

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
            # reset = True  # 用于visdom的方法，绘制batch的loss曲线，指示是否是append的
            # 实例化3个loss对象，用于计算epoch loss
            ad_loss_train_obj = mm.Loss()
            ad_loss_fixed_obj = mm.Loss()
            recon_loss_objs = [mm.Loss() for _ in
                               range(self.num_encoder_layers)]
            # 训练时需要将模型更改至训练状态
            for model in self.models.values():
                model.train()
            # 循环每个batch进行训练
            for batch_x, batch_y in tqdm(dataloaders['train'], 'Train Batch: '):
                batch_x = batch_x.to(self.device, torch.float)
                batch_y = batch_y.to(self.device, torch.float)
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad()
                for _ in range(self.discriminate_train_num):
                    ad_loss_train = self._forward_discriminate(
                        batch_x, batch_y)
                for _ in range(self.autoencode_train_num):
                    recon_losses, ad_loss_fixed = self._forward_autoencode(
                        batch_x, batch_y)
                # 记录loss(这样，就算update了多次loss，也只记录最后一次)
                ad_loss_train_obj.add(ad_loss_train, batch_x.size(0))
                ad_loss_fixed_obj.add(ad_loss_fixed, batch_x.size(0))
                for rl_loss, rl in zip(recon_loss_objs, recon_losses):
                    rl_loss.add(rl, batch_x.size(0))
            # 看warning说需要把scheduler的step放在optimizer.step之后使用
            for sche in self.scheduals.values():
                sche.step()

            # 记录epoch loss
            self.history['discriminate_loss_train'].append(
                ad_loss_train_obj.value())
            self.history['discriminate_loss_fixed'].append(
                ad_loss_fixed_obj.value())
            for i in range(self.num_encoder_layers):
                self.history['recon%d_loss' % i].append(
                    recon_loss_objs[i].value())
            # 可视化epoch loss
            # 因为两组loss的取值范围相差太大，所以分开来显示
            self.visobj.add_epoch_loss(
                winname='disc_losses',
                disc_train=self.history['discriminate_loss_train'][-1],
                disc_fixed=self.history['discriminate_loss_fixed'][-1],
            )
            recon_kwargs = {k: v[-1] for k, v in self.history.items()
                            if k.startswith('recon')}
            self.visobj.add_epoch_loss(winname='recon_losses', **recon_kwargs)

            ## valid phase

            # 使用当前训练的模型去得到去批次结果
            all_data = ConcatData(datas['subject'], datas['qc'])
            all_reses_dict = generate(
                self.models, all_data, no_be_num=self.no_be_num,
                device=self.device, bs=self.bs, nw=self.nw
            )
            # 对数据进行对应的pca
            subject_pca, qc_pca = pca_for_dict(all_reses_dict)
            # plot pca
            pca_plot(subject_pca, qc_pca)
            # display in visdom
            self.visobj.vis.matplot(plt, win='PCA', opts={'title': 'PCA'})
            plt.close()

        return self.models, self.history

    def _forward_autoencode(self, batch_x, batch_y):
        ''' autoencode进行训练的部分 '''
        with torch.enable_grad():
            # encoder
            encoder_hiddens = self.models['encoder'](batch_x)
            hidden = encoder_hiddens[-1]
            # decoder
            if self.train_with_qc:
                hidden_mask = torch.ones_like(hidden)
                hidden_mask[batch_y[:, 3] == 0, :self.no_be_num] = 0
                decoder_hiddens = self.models['decoder'](hidden)
            else:
                decoder_hiddens = self.models['decoder'](hidden)
            batch_x_recon = decoder_hiddens[-1]
            # discriminator, but not training
            logit = self.models['discriminator'](hidden[:, :self.no_be_num])
            # reconstruction losses
            recon_losses = [
                self.criterions['reconstruction'](batch_x_recon, batch_x)]
            for i in range(len(encoder_hiddens)-1):
                recon_loss1 = self.criterions['reconstruction'](
                    encoder_hiddens[i], decoder_hiddens[-i-2])
                recon_losses.append(recon_loss1)
            # discriminator loss
            adversarial_loss = self.criterions['adversarial'](logit, batch_y)
            # 组合这些loss来得到最终计算梯度使用的loss
            # 分类做的不好，说明这些维度中没有批次的信息，批次的信息都在后面的维度中
            recon_loss = 0.
            if self.interconnect:
                for rl in recon_losses:
                    recon_loss += rl
            else:
                recon_loss = recon_losses[0]
            all_loss = self.ae_weight * recon_loss - \
                self.disc_weight * adversarial_loss
        all_loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(
                chain(
                    self.models['encoder'].parameters(),
                    self.models['decoder'].parameters()
                ), max_norm=1
            )
        self.optimizers['autoencode'].step()
        return recon_losses, adversarial_loss

    def _forward_discriminate(self, batch_x, batch_y):
        ''' discriminator进行训练的部分 '''
        with torch.no_grad():
            hidden = self.models['encoder'](batch_x)[-1]
        with torch.enable_grad():
            logit = self.models['discriminator'](hidden[:, :self.no_be_num])
            adversarial_loss = self.criterions['adversarial'](logit, batch_y)
        adversarial_loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(
                self.models['discriminator'].parameters(), max_norm=1)
        self.optimizers['discriminate'].step()
        return adversarial_loss

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
    trainer = BatchEffectTrainer(
        subject_dat.num_features, config.args.bottle_num,
        config.args.no_batch_num, batch_label_num=subject_dat.num_batch_labels,
        lrs=config.args.ae_disc_lr,
        bs=config.args.batch_size, nw=config.args.num_workers,
        epoch=config.args.epoch, device=torch.device('cuda:0'),
        l2=config.args.l2, clip_grad=True,
        ae_disc_train_num=config.args.ae_disc_train_num,
        ae_disc_weight=config.args.ae_disc_weight,
        supervise=config.args.supervise,
        label_smooth=config.args.label_smooth,
        train_with_qc=config.args.train_data == 'all',
        spectral_norm=config.args.spectral_norm,
        schedual_stones=config.args.schedual_stones,
        interconnect=config.args.interconnect
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
