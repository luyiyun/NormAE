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

from datasets import MetaBatchEffect, Demo
from networks import SimpleCoder, RankLoss, CustomCrossEntropy, LabelSmoothing
import transfer as T
import metrics as mm
from visual import VisObj, pca_plot
from sklearn.decomposition import PCA


class BatchEffectTrainer:
    def __init__(
        self, in_features, bottle_num, no_be_num, lrs=0.01, bs=64, nw=6,
        epoch=100, device=torch.device('cuda:0'), l2=0.0, clip_grad=False,
        ae_disc_train_num=(1, 1), ae_disc_weight=(1.0, 5.0),
        supervise='both', l1=False, label_smooth=0.0
    ):

        # 是否进行label smoothing
        if label_smooth == 0.0:
            # 这个就是可以自动把label转换成long再送给ce
            cls_criterion = CustomCrossEntropy()
        else:
            cls_criterion = LabelSmoothing(4, label_smooth)

        # according to supervise, choose classification criterion
        if supervise == 'both':
            logit_dim = 5
            adversarial_criterion = RankLoss(classification=cls_criterion)
        elif supervise == 'rank':
            logit_dim = 1
            adversarial_criterion = RankLoss(classification=None)
        elif supervise == 'cls':
            logit_dim = 4
            adversarial_criterion = cls_criterion
        else:
            raise ValueError(
                "supervise must be one of 'both', 'rank' and 'cls'")
        self.supervise = supervise

        # 得到3个模型
        self.models = {
            'encoder': SimpleCoder(
                [in_features, 300, 300, 300, bottle_num], lrelu=True,
                last_act=None, norm=nn.BatchNorm1d, dropout=None
            ).to(device),
            'decoder': SimpleCoder(
                [bottle_num, 300, 300, 300, in_features], lrelu=True,
                last_act=None, norm=nn.BatchNorm1d, dropout=None
            ).to(device),
            'discriminator': SimpleCoder(
                [no_be_num, 300, 300, logit_dim], lrelu=False,
                norm=nn.BatchNorm1d, last_act=None
            ).to(device)
        }

        # 得到两个loss
        self.criterions = {
            'adversarial': adversarial_criterion,
            'reconstruction': nn.MSELoss()
        }

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
                self.optimizers['discriminate'], [2000], gamma=0.1
            ),
            'autoencode': optim.lr_scheduler.MultiStepLR(
                self.optimizers['autoencode'], [2000], gamma=0.1
            )
        }

        # 初始化结果记录
        self.history = {
            'reconstruction_loss': [], 'discriminate_loss_train': [],
            'discriminate_loss_fixed': []
        }

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

        # 可视化工具
        self.visobj = VisObj()

    def fit(self, datas):
        ''' datas是多个Dataset对象的可迭代对象 '''
        # 实例化一个pca estimator，用于训练时实时的将数据进行可视化
        pca = PCA(2)
        # 将Dataset对象变成Dataloader对象
        dataloaders = {}
        for k, v in datas.items():
            dataloaders[k] = data.DataLoader(
                v, batch_size=self.bs, num_workers=self.nw,
                shuffle=(k == 'train')
            )

        # 开始进行多个epoch训练
        for e in tqdm(range(self.epoch), 'Epoch: '):
            ## train phase
            # reset = True  # 用于visdom的方法，绘制batch的loss曲线，指示是否是append的
            # 实例化3个loss对象，用于计算epoch loss
            ad_loss_train_obj = mm.Loss()
            ad_loss_fixed_obj = mm.Loss()
            recon_loss_obj = mm.Loss()
            # 训练时需要将模型更改至训练状态
            for model in self.models.values():
                model.train()
            # 循环每个batch进行训练
            for batch_x, batch_y in tqdm(dataloaders['train'], 'Batch: '):
                batch_x = batch_x.to(self.device, torch.float)
                batch_y = batch_y.to(self.device, torch.float)
                if self.supervise == 'cls':
                    batch_y = batch_y[:, 1]
                elif self.supervise == 'both':
                    batch_y = batch_y[:, :2]
                elif self.supervise == 'rank':
                    batch_y = batch_y[:, 0]
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad()
                for _ in range(self.discriminate_train_num):
                    ad_loss_train = self._forward_discriminate(
                        batch_x, batch_y)
                for _ in range(self.autoencode_train_num):
                    recon_loss, ad_loss_fixed = self._forward_autoencode(
                        batch_x, batch_y)
                # 记录loss(这样，就算update了多次loss，也只记录最后一次)
                ad_loss_train_obj.add(ad_loss_train, batch_x.size(0))
                ad_loss_fixed_obj.add(ad_loss_fixed, batch_x.size(0))
                recon_loss_obj.add(recon_loss, batch_x.size(0))
            # 看warning说需要把scheduler的step放在optimizer.step之后使用
            for sche in self.scheduals.values():
                sche.step()

            # 记录并可视化epoch loss
            # 因为两组loss的取值范围相差太大，所以分开来显示
            self.history['discriminate_loss_train'].append(
                ad_loss_train_obj.value())
            self.history['discriminate_loss_fixed'].append(
                ad_loss_fixed_obj.value())
            self.history['reconstruction_loss'].append(recon_loss_obj.value())
            self.visobj.add_epoch_loss(
                winname='disc_losses',
                disc_train=self.history['discriminate_loss_train'][-1],
                disc_fixed=self.history['discriminate_loss_fixed'][-1],
            )
            self.visobj.add_epoch_loss(
                winname='recon_losses',
                recon=self.history['reconstruction_loss'][-1],
            )


            ## valid phase
            reses = self.transform(
                dataloaders['train'], dataloaders['qc'], return_recon=True,
                # 只在第一个epoch的时候计算
                return_ori=(e == 0), return_ys=(e == 0)
            )
            train_reses_dict, qc_reses_dict = list(reses)
            if e == 0:
                # get original_x and label(ys)
                x_ori_plot = np.concatenate([
                    train_reses_dict['original_x'], qc_reses_dict['original_x']
                ], axis=0)
                ys_plot = np.concatenate([
                    train_reses_dict['ys'], qc_reses_dict['ys']
                ], axis=0)
                # pca for original_x
                x_ori_plot_pca = pca.fit_transform(x_ori_plot)
            # get reconstructed datas
            x_recons_nobe = np.concatenate([
                train_reses_dict['recons_no_batch'],
                qc_reses_dict['recons_no_batch']
            ], axis=0)
            x_recons_be = np.concatenate([
                train_reses_dict['recons_all'], qc_reses_dict['recons_all']
            ], axis=0)
            # pca for reconstructed datas
            x_recons_nobe_pca = pca.fit_transform(x_recons_nobe)
            x_recons_be_pca = pca.fit_transform(x_recons_be)
            # indexes for plot
            subject_num = len(train_reses_dict['recons_all'])
            # plot pca
            pca_plot(
                x_ori_plot_pca, x_recons_nobe_pca, x_recons_be_pca,
                ys_plot, subject_num
            )
            self.visobj.vis.matplot(plt, win='PCA', opts={'title': 'PCA'})
            plt.close()

        return self.models, self.history

    def _forward_autoencode(self, batch_x, batch_y):
        with torch.enable_grad():
            hidden = self.models['encoder'](batch_x)
            batch_x_recon = self.models['decoder'](hidden)
            logit = self.models['discriminator'](hidden[:, :self.no_be_num])
            reconstruction_loss = self.criterions['reconstruction'](
                batch_x_recon, batch_x)
            adversarial_loss = self.criterions['adversarial'](logit, batch_y)
            # 分类做的不好，说明这写维度中没有批次的信息，批次的信息都在后面的维度中
            all_loss = self.ae_weight * reconstruction_loss - \
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
        return reconstruction_loss, adversarial_loss

    def _forward_discriminate(self, batch_x, batch_y):
        with torch.no_grad():
            hidden = self.models['encoder'](batch_x)
        with torch.enable_grad():
            logit = self.models['discriminator'](hidden[:, :self.no_be_num])
            adversarial_loss = self.criterions['adversarial'](logit, batch_y)
        adversarial_loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(
                self.models['discriminator'].parameters(), max_norm=1)
        self.optimizers['discriminate'].step()
        return adversarial_loss

    def transform(
        self, *data_loaders, return_ori=False, return_recon=False,
        return_ys=False
    ):
        '''
        data_loaders: Dataset对象或Dataloader对象，如果是Dataset则会利用实例化
            时的num_workers和batch_size来将其转换成一个Dataloader对象，可以输入
            多个；
        return_ori：If True，Original X tensor will be return；
        return_recon：If True，X reconstructed by AE will be return；
        return_ys：If True，Y tensor will be return；

        return：
            It's generator，the element is dict，the keys are "recons_no_batch、
            recons_all、original_x、ys”, the values are ndarrays
        '''
        for m in self.models.values():
            m.eval()
        for data_loader in data_loaders:
            if isinstance(data_loader, data.Dataset):
                data_loader = data.DataLoader(
                    data_loader, batch_size=self.bs, num_workers=self.nw
                )
            x_recon, x_ori, x_recon_be, ys = [], [], [], []
            with torch.no_grad():
                for batch_x, batch_y in data_loader:
                    if return_ori:
                        x_ori.append(batch_x)
                    if return_ys:
                        ys.append(batch_y)
                    batch_x = batch_x.to(self.device, torch.float)
                    hidden = self.models['encoder'](batch_x)
                    if return_recon:
                        x_recon_be.append(self.models['decoder'](hidden))
                    hidden[:, self.no_be_num:] = 0
                    batch_x_recon = self.models['decoder'](hidden)
                    x_recon.append(batch_x_recon)
            res = {
                'recons_no_batch': torch.cat(x_recon),
                'recons_all': (
                    torch.cat(x_recon_be) if return_recon else None
                ),
                'original_x': (
                    torch.cat(x_ori) if return_ori else None
                ),
                'ys': torch.cat(ys) if return_ys else None
            }
            for k, v in res.items():
                if v is not None:
                    res[k] = v.detach().cpu().numpy()
            yield res


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
    if config.args.data == 'metabolic':
        meta_data = MetaBatchEffect.from_csv(
            config.sample_file, config.meta_file,
            pre_transfer=T.Normalization(config.args.data_norm)
        )
        subject_dat, qc_dat = meta_data.split_qc()
        subject_dat = meta_data
    elif config.args.data == 'demo':
        subject_dat = Demo.from_csv(
            config.demo_sub_file,
            pre_transfer=T.Normalization(config.args.data_norm)
        )
        qc_dat = Demo.from_csv(
            config.demo_qc_file,
            pre_transfer=T.Normalization(config.args.data_norm)
        )
    datas = {'train': subject_dat, 'qc': qc_dat}

    # ----- 训练网络 -----
    trainer = BatchEffectTrainer(
        subject_dat.num_features, config.args.bottle_num,
        config.args.no_batch_num, lrs=config.args.ae_disc_lr,
        bs=config.args.batch_size, nw=config.args.num_workers,
        epoch=config.args.epoch, device=torch.device('cuda:0'),
        l2=config.args.l2, clip_grad=True,
        ae_disc_train_num=config.args.ae_disc_train_num,
        ae_disc_weight=config.args.ae_disc_weight,
        supervise=config.args.supervise, l1=False,
        label_smooth=config.args.label_smooth
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
