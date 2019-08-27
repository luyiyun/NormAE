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

from datasets import MetaBatchEffect
from networks import Coder
import transfer as T
import metrics as mm
from visual import VisObj
from sklearn.decomposition import PCA


class NoneScheduler:
    def __init__(self, optimizer):
        pass

    def step(self):
        pass


class BatchEffectTrainer:
    def __init__(
        self, in_features, bottle_num, no_be_num, lrs=0.01, bs=64, nw=6,
        epoch=100, device=torch.device('cuda:0'), l2=0.0, clip_grad=False,
        ae_disc_train_num=(1, 1), no_best=False, ae_disc_weight=(1.0, 5.0),
        mse=True
    ):

        # 得到3个模型
        self.models = {
            'encoder': Coder(
                in_features, bottle_num, block_num=4, norm=None, dropout=0.0
            ).to(device),
            'decoder': Coder(
                bottle_num, in_features, block_num=4, norm=None, dropout=0.0
            ).to(device),
            'discriminator': Coder(
                no_be_num, 4, block_num=2, spectral_norm=False, norm='BN'
            ).to(device)
            # 'discriminator': nn.Sequential(
            #     nn.Linear(no_be_num, 50),
            #     nn.LeakyReLU(),
            #     nn.Linear(50, 50),
            # ).to(device)
        }
        # self.models['discriminator'].in_f = no_be_num

        # 得到两个loss
        self.criterions = {
            'adversarial': nn.CrossEntropyLoss() if not mse else nn.MSELoss(),
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
                lr=lrs[0]
            ),
            'discriminate': optim.Adam(
                self.models['discriminator'].parameters(), lr=lrs[1]
            )
        }
        self.scheduals = optim.lr_scheduler.MultiStepLR(
            self.optimizers['discriminate'],
            [500], gamma=0.1
        )

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
        self.mse = mse

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
                v, batch_size=self.bs, num_workers=self.nw)

        # 开始进行多个epoch训练
        for _ in tqdm(range(self.epoch), 'Epoch: '):
            ## train phase
            # reset = True  # 用于visdom的方法，绘制batch的loss曲线，指示是否是append的
            # 实例化3个loss对象，用于计算epoch loss
            ad_loss_train_obj = mm.Loss()
            ad_loss_fixed_obj = mm.Loss()
            recon_loss_obj = mm.Loss()
            # 训练时需要将模型更改至训练状态
            for model in self.models.values():
                model.train()
            self.scheduals.step()
            # 循环每个batch进行训练
            for batch_x, batch_y in tqdm(dataloaders['train'], 'Batch: '):
                batch_x = batch_x.to(self.device, torch.float)
                batch_y = batch_y.to(self.device, torch.long).squeeze()
                if self.mse:
                    batch_y_oh = torch.arange(4).diag()[batch_y].float().to(
                        self.device)
                else:
                    batch_y_oh = batch_y
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad()
                for _ in range(self.discriminate_train_num):
                    ad_loss_train = self._forward_discriminate(
                        batch_x, batch_y_oh)
                for _ in range(self.autoencode_train_num):
                    recon_loss, ad_loss_fixed = self._forward_autoencode(
                        batch_x, batch_y_oh)
                # 记录loss(这样，就算update了多次loss，也只记录最后一次)
                ad_loss_train_obj.add(ad_loss_train, batch_x.size(0))
                ad_loss_fixed_obj.add(ad_loss_fixed, batch_x.size(0))
                recon_loss_obj.add(recon_loss, batch_x.size(0))
                # 可视化batch loss
                # self.visobj.add_batch_loss(
                #     reset, ad=ad_loss, ae=ae_loss, recon=recon_loss)
                # reset = False

            # 记录并可视化epoch loss
            self.history['discriminate_loss_train'].append(
                ad_loss_train_obj.value())
            self.history['discriminate_loss_fixed'].append(
                ad_loss_fixed_obj.value())
            self.history['reconstruction_loss'].append(recon_loss_obj.value())
            self.visobj.add_epoch_loss(
                disc_train=self.history['discriminate_loss_train'][-1],
                disc_fixed=self.history['discriminate_loss_fixed'][-1],
                recon=self.history['reconstruction_loss'][-1]
            )


            ## valid phase
            reses = self.transform(
                dataloaders['train'], dataloaders['qc'], return_ori=True)
            x_recons = []
            x_oris = []
            for (x_recon, x_ori), phase in zip(reses, ['train', 'qc']):
                x_recon = x_recon.cpu().numpy()
                x_ori = x_ori.cpu().numpy()
                x_recons.append(x_recon)
                x_oris.append(x_ori)
            x_oris = np.concatenate(x_oris)
            x_recons = np.concatenate(x_recons)
            x_oris_pca = pca.fit_transform(x_oris)
            x_recons_pca = pca.fit_transform(x_recons)

            train_index = slice(0, len(datas['train']))
            qc_index = slice(
                len(datas['train']),
                len(datas['train'])+len(datas['qc'])
            )
            _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
            ax1.scatter(
                x_oris_pca[train_index, 0], x_oris_pca[train_index, 1],
                c='skyblue', alpha=0.7, label='Original'
            )
            ax1.scatter(
                x_recons_pca[train_index, 0], x_recons_pca[train_index, 1],
                c='orange', alpha=0.7, label='Recon'
            )
            ax2 = plt.subplot(1, 2, 2)
            ax2.scatter(
                x_oris_pca[qc_index, 0], x_oris_pca[qc_index, 1],
                c='skyblue', alpha=0.7, label='Original'
            )
            ax2.scatter(
                x_recons_pca[qc_index, 0], x_recons_pca[qc_index, 1],
                c='orange', alpha=0.7, label='Recon'
            )
            ax1.set_title('Train')
            ax2.set_title('QC')
            ax1.legend()
            ax2.legend()
            self.visobj.vis.matplot(plt, win='PCA', opts={'title': 'PCA'})
            plt.close()

        return self.models, self.history

    def _forward_autoencode(self, batch_x, batch_y):
        with torch.set_grad_enabled(True):
            hidden = self.models['encoder'](batch_x)
            batch_x_recon = self.models['decoder'](hidden)
        with torch.set_grad_enabled(False):
            logit = self.models['discriminator'](hidden[:, :self.no_be_num])
        with torch.set_grad_enabled(True):
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
        with torch.set_grad_enabled(False):
            hidden = self.models['encoder'](batch_x)
        with torch.set_grad_enabled(True):
            logit = self.models['discriminator'](hidden[:, :self.no_be_num])
            adversarial_loss = self.criterions['adversarial'](logit, batch_y)
        adversarial_loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(
                self.models['discriminator'].parameters(), max_norm=1)
        self.optimizers['discriminate'].step()
        return adversarial_loss

    def transform(self, *dataloaders, return_ori=False):
        for m in self.models.values():
            m.eval()
        for dataloader in dataloaders:
            with torch.no_grad():
                x_recon = []
                if return_ori:
                    x_ori = []
                for batch_x in dataloader:
                    if isinstance(batch_x, (tuple, list)):
                        batch_x = batch_x[0]
                    batch_x = batch_x.to(self.device, torch.float)
                    hidden = self.models['encoder'](batch_x)
                    hidden[:, self.no_be_num:] = 0
                    batch_x_recon = self.models['decoder'](hidden)
                    x_recon.append(batch_x_recon)
                    if return_ori:
                        x_ori.append(batch_x)
                x_recon = torch.cat(x_recon)
                if return_ori:
                    x_ori = torch.cat(x_ori)
                    yield x_recon, x_ori
                else:
                    yield x_recon

    def _record_best_models(self, epoch_metric):
        if epoch_metric < self.best_metric:
            self.best_metric = epoch_metric
            self.best_models_wts = {
                k: copy.deepcopy(v.state_dict())
                for k, v in self.models.items()
            }


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
    meta_data = MetaBatchEffect.from_csv(
        config.sample_file, config.meta_file, 'batch',
        pre_transfer=T.MultiCompose(
            T.Normalization(), lambda x, y: (x, y - 1)
        )
    )
    subject_dat, qc_dat = meta_data.split_qc()
    datas = {'train': subject_dat, 'qc': qc_dat}

    # ----- 训练网络 -----
    trainer = BatchEffectTrainer(
        subject_dat.num_features, config.args.bottle_num,
        config.args.no_batch_num, lrs=config.args.ae_disc_lr,
        bs=config.args.batch_size, nw=config.args.num_workers,
        epoch=config.args.epoch, device=torch.device('cuda:0'),
        l2=0.0, clip_grad=True,
        ae_disc_train_num=config.args.ae_disc_train_num,
        no_best=False, ae_disc_weight=config.args.ae_disc_weight

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
