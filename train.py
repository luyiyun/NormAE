import os
import copy
import json
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from datasets import get_metabolic_data, get_demo_data, ConcatData
from networks import SimpleCoder, ClsOrderLoss, ResNet
from transfer import Normalization
import metrics as mm
from visual import VisObj, pca_for_dict, pca_plot


class BatchEffectTrainer:
    def __init__(
        self, in_features, bottle_num, be_num, batch_label_num=None,
        lrs=0.01, bs=64, nw=6, epoch=(200, 100, 1000),
        device=torch.device('cuda:0'), l2=0.0, clip_grad=False,
        ae_disc_train_num=(1, 1), disc_weight=1.0,
        label_smooth=0.2, train_with_qc=False, spectral_norm=False,
        schedual_stones=[3000], cls_leastsquare=False, order_losstype=False,
        cls_order_weight=(1.0, 1.0), use_batch_for_order=True,
        visdom_port=8097, encoder_hiddens=[300, 300, 300],
        decoder_hiddens=[300, 300, 300], disc_hiddens=[300, 300],
        early_stop=False, net_type='simple', resnet_bottle_num=50,
        optimizer='rmsprop', denoise=0.1, reconst_loss='mae',
        disc_weight_epoch=500, early_stop_check_num=100
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
        label_smooth: the label smooth parameter for disciminate;
        train_with_qc: if true, the dataset of training is concatenated data of
            subject and qc;
        spectral_norm: if true, use spectral normalization for all linear layers;
        schedual_stones: the epoch of lrs multiply 0.1;
        cls_leastsquare: if ture, use mse in ClswithLabelSmoothLoss;
        order_leastsquare: if ture, use mse in OrderLoss;
        cls_order_weight: weights for cls and order in ClsOrderLoss;
        use_batch_for_order: if use, compute rank loss with batch;
        '''

        # 用于构建网络的参数
        self.in_features = in_features
        self.encoder_hiddens = encoder_hiddens
        self.decoder_hiddens = decoder_hiddens
        self.disc_hiddens = disc_hiddens
        self.bottle_num, self.be_num = bottle_num, be_num
        self.batch_label_num = batch_label_num
        self.spectral_norm = spectral_norm
        self.device = device
        self.denoise = denoise
        self.net_type = net_type
        self.resnet_bottle_num = resnet_bottle_num
        # 用于构建loss
        if len(disc_weight) == 2:
            # 如果disc weight有两个，则在iter phase阶段，disc weight(lambda)
            #   是进行线性增长的(增长的epoch数量是disc_weight_epoch)，其端点就
            #   是这两个值，在iter phase的剩下的阶段中，weight是disc_weight[1],
            #   而在ae phase这个值是0(这个时候没有用到disc loss)，在disc phase
            #   值是1.
            iter_w = np.linspace(*disc_weight, num=disc_weight_epoch)
            self.disc_weight = np.concatenate(
                [
                    np.zeros(epoch[0]), np.ones(epoch[1]), iter_w,
                    np.full(epoch[2]-disc_weight_epoch, disc_weight[-1])
                ], axis=0)
        elif len(disc_weight) == 1:
            # 如果disc weight只有一个值，则在iter phase阶段则一直是这一个值
            self.disc_weight = np.concatenate([
                np.zeros(epoch[0]), np.ones(epoch[1]),
                np.full(epoch[2], disc_weight[0])
            ], axis=0)
        else:
            raise ValueError
        self.label_smooth = label_smooth
        self.cls_leastsquare = cls_leastsquare
        self.order_losstype = order_losstype
        self.cls_weight, self.order_weight = cls_order_weight
        self.use_batch_for_order = use_batch_for_order
        self.rec_type = reconst_loss
        # optimizer的参数
        self.lrs = [lrs] * 2 if isinstance(lrs, float) else lrs
        self.rec_lr, self.cls_lr = self.lrs
        self.l2, self.clip_grad = l2, clip_grad
        self.schedual_stones = schedual_stones
        self.optimizer = optimizer
        # 训练参数
        self.rec_epoch, self.cls_epoch, self.iter_epoch = epoch
        self.epoch = sum(epoch)
        self.rec_train_num, self.cls_train_num = ae_disc_train_num
        self.bs, self.nw = bs, nw
        self.train_with_qc = train_with_qc
        self.early_stop = early_stop
        # 其他属性
        self.visdom_port = visdom_port
        self.early_stop_check_num = early_stop_check_num

        # 构建模型、损失及优化器
        self._build_model()

        # 初始化结果记录
        self.history = {
            'cls_loss_cls': [], 'cls_loss_rec': [], 'rec_loss': [],
            'qc_rec_loss': [], 'qc_distance': []
        }
        # 可视化工具
        self.visobj = VisObj(self.visdom_port)
        # 提前停止使用
        self.early_stop_objs = {
            'best_epoch': -1, 'best_qc_loss': 1000, 'best_qc_distance': 1000,
            'best_models': None, 'index': 0, 'best_score': 2000
        }

    def fit(self, datas):
        ''' datas是多个Dataset对象的可迭代对象 '''
        # 将Dataset对象变成Dataloader对象
        train_data = data.ConcatDataset([datas['subject'], datas['qc']]) \
            if self.train_with_qc else datas['subject']
        dataloaders = {
            'train': data.DataLoader(train_data, batch_size=self.bs,
                                     num_workers=self.nw, shuffle=True),
            'qc': data.DataLoader(datas['qc'], batch_size=self.bs,
                                  num_workers=self.nw)
        }
        # 开始进行多个epoch训练
        bar = tqdm(total=self.epoch)
        for e in range(self.epoch):
            self.e = e
            # 根据e进入不同的phase
            if e < self.rec_epoch:
                self.phase = 'rec_pretrain'
            elif e < self.rec_epoch + self.cls_epoch:
                self.phase = 'cls_pretrain'
            else:
                self.phase = 'iter_train'
            bar.set_description(self.phase)
            ## train phase
            # 实例化3个loss对象，用于计算epoch loss
            cls_loss_cls_obj = mm.Loss()
            cls_loss_rec_obj = mm.Loss()
            rec_loss_obj = mm.Loss()
            # 训练时需要将模型更改至训练状态
            for model in self.models.values():
                model.train()
            # 循环每个batch进行训练
            for batch_x, batch_y in tqdm(dataloaders['train'], 'Batch: '):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad()
                if self.phase in ['cls_pretrain', 'iter_train']:
                    for _ in range(self.cls_train_num):
                        cls_loss_cls = self._forward_discriminate(
                            batch_x, batch_y)
                    cls_loss_cls_obj.add(cls_loss_cls, batch_x.size(0))
                if self.phase in ['rec_pretrain', 'iter_train']:
                    for _ in range(self.rec_train_num):
                        rec_loss, cls_loss_rec = self._forward_autoencode(
                            batch_x, batch_y)
                    cls_loss_rec_obj.add(cls_loss_rec, batch_x.size(0))
                    rec_loss_obj.add(rec_loss, batch_x.size(0))
            # 看warning说需要把scheduler的step放在optimizer.step之后使用
            if self.phase == 'iter_train':
                for sche in self.scheduals.values():
                    sche.step()
            # 记录epoch loss
            self.history['cls_loss_cls'].append(cls_loss_cls_obj.value())
            self.history['cls_loss_rec'].append(cls_loss_rec_obj.value())
            self.history['rec_loss'].append(rec_loss_obj.value())
            # 可视化epoch loss
            # 因为两组loss的取值范围相差太大，所以分开来显示
            self.visobj.add_epoch_loss(
                winname='disc_losses',
                disc_train=self.history['cls_loss_cls'][-1],
                disc_fixed=self.history['cls_loss_rec'][-1],
            )
            self.visobj.add_epoch_loss(
                winname='recon_losses',
                recon_loss=self.history['rec_loss'][-1]
            )
            self.visobj.add_epoch_loss(winname="disc_weight",
                                       disc_weight=self.disc_weight[self.e])

            ## valid phase
            # 使用当前训练的模型去得到去批次结果
            all_data = ConcatData(datas['subject'], datas['qc'])
            all_reses_dict, qc_loss = self.generate(
                all_data, verbose=False, compute_qc_loss=True)
            # 对数据进行对应的pca
            subject_pca, qc_pca = pca_for_dict(all_reses_dict)
            # plot pca
            pca_plot(subject_pca, qc_pca)
            # display in visdom
            self.visobj.vis.matplot(plt, win='PCA', opts={'title': 'PCA'})
            plt.close()

            ## early stopping
            qc_dist = mm.mean_distance(qc_pca['Rec_nobe'])
            self.history['qc_rec_loss'].append(qc_loss)
            self.history['qc_distance'].append(qc_dist)
            self.visobj.add_epoch_loss(winname='qc_rec_loss', qc_loss=qc_loss)
            self.visobj.add_epoch_loss(winname='qc_distance', qc_dist=qc_dist)
            if self.early_stop and e >= self.epoch - self.early_stop_check_num:
                self._check_qc(qc_dist, qc_loss)
            
            # progressbar
            bar.update(1)
        bar.close()

        if self.early_stop:
            print('')
            print('The best epoch is %d' % self.early_stop_objs['best_epoch'])
            print('The best qc loss is %.4f' %
                self.early_stop_objs['best_qc_loss'])
            print('The best qc distance is %.4f' %
                self.early_stop_objs['best_qc_distance'])
            for k, v in self.models.items():
                v.load_state_dict(self.early_stop_objs['best_models'][k])
            self.early_stop_objs.pop('best_models')
            return self.models, self.history, self.early_stop_objs
        return self.models, self.history

    def generate(self, data_loader, verbose=True, compute_qc_loss=False):
        # 准备数据集和模型
        for m in self.models.values():
            m.to(self.device).eval()
        if isinstance(data_loader, data.Dataset):
            data_loader = data.DataLoader(
                data_loader, batch_size=self.bs, num_workers=self.nw)
        x_ori, x_rec, x_rec_nobe, ys, codes = [], [], [], [], []
        qc_loss = mm.Loss()

        # encoding
        if verbose:
            print('----- encoding -----')
            iterator = tqdm(data_loader, 'encode and decode: ')
        else:
            iterator = data_loader
        with torch.no_grad():
            for batch_x, batch_y in iterator:
                # X和Y都备份一次
                x_ori.append(batch_x)
                ys.append(batch_y)
                # 计算hidden codes
                batch_x = batch_x.to(self.device, torch.float)
                hidden = self.models['encoder'](batch_x)
                codes.append(hidden)
                # AE重建
                x_rec.append(self.models['decoder'](hidden))
                # 去除批次重建
                hidden_copy = hidden.clone()
                hidden_copy[:, :self.be_num] = 0
                x_rec_nobe.append(self.models['decoder'](hidden_copy))
                # 是否计算qc的loss，在训练的时候有用
                if compute_qc_loss:
                    qc_index = batch_y[:, -1] == 0.
                    if qc_index.sum() > 0:
                        batch_qc_loss = self.criterions['rec'](
                            batch_x[qc_index], x_rec[-1][qc_index])
                        qc_loss.add(
                            batch_qc_loss,
                            qc_index.sum().detach().cpu().item()
                        )
                    else:
                        qc_loss.add(torch.tensor(0.), 0)
        # 数据整理成dataframe，并保存到一个dict中
        res = {
            'Ori': torch.cat(x_ori), 'Ys': torch.cat(ys),
            'Codes': torch.cat(codes), 'Rec': torch.cat(x_rec),
            'Rec_nobe': torch.cat(x_rec_nobe)
        }
        for k, v in res.items():
            if v is not None:
                if k == 'Ys':
                    res[k] = pd.DataFrame(
                        v.detach().cpu().numpy(),
                        index=data_loader.dataset.Y_df.index,
                        columns=data_loader.dataset.Y_df.columns
                    )
                elif k != 'Codes':
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

        if compute_qc_loss:
            return res, qc_loss.value()
        return res

    def load_model(self, model_file):
        saved_model = torch.load(model_file)
        for k in self.early_stop_objs:
            saved_model.pop(k, None)
        self.models = saved_model

    def _check_qc(self, qc_dist, qc_loss):
        early_stop_score = qc_dist + qc_loss * 100
        if early_stop_score < self.early_stop_objs['best_score']:
            self.early_stop_objs['best_epoch'] = self.e
            self.early_stop_objs['best_models'] = {
                k: copy.deepcopy(v.state_dict())
                for k, v in self.models.items()
            }
            self.early_stop_objs['best_qc_loss'] = qc_loss
            self.early_stop_objs['best_qc_distance'] = qc_dist
            self.early_stop_objs['best_score'] = early_stop_score
            self.early_stop_objs['index'] = 0
        else:
            self.early_stop_objs['index'] += 1

    def _build_model(self):
        '''
        构建模型，损失函数，优化器以及lr_scheduler
        '''
        # according to cls_order_weight, choose classification criterion
        if self.cls_weight > 0.0 and self.order_weight > 0.0:
            self.logit_dim = self.batch_label_num + 1
        elif self.cls_weight > 0.0:
            self.logit_dim = self.batch_label_num
        elif self.order_weight > 0.0:
            self.logit_dim = 1
        else:
            raise ValueError(
                'cls_weight and order_weight must not be all zero')
        # 得到3个模型
        if self.net_type == 'simple':
            self.models = {
                'encoder': SimpleCoder(
                    [self.in_features] + self.encoder_hiddens +\
                    [self.bottle_num]
                ).to(self.device),
                'decoder': SimpleCoder(
                    [self.bottle_num] + self.decoder_hiddens +\
                    [self.in_features]
                ).to(self.device),
                'disc': SimpleCoder(
                    [self.bottle_num-self.be_num] + self.disc_hiddens +\
                    [self.logit_dim], bn=False
                ).to(self.device)
            }
        else:
            self.models = {
                'encoder': ResNet(
                    [self.in_features] + self.encoder_hiddens +\
                    [self.bottle_num], self.resnet_bottle_num
                ).to(device),
                'decoder': ResNet(
                    [self.bottle_num] + self.decoder_hiddens +\
                    [self.in_features], self.resnet_bottle_num
                ).to(device),
                'disc': ResNet(
                    [self.bottle_num-self.be_num] + self.disc_hiddens +\
                    [self.logit_dim], self.resnet_bottle_num
                ).to(device)
            }
        # 构建loss
        self.criterions = {
            'rec': nn.L1Loss() if self.rec_type == 'mae' else nn.MSELoss(),
            'cls': ClsOrderLoss(
                self.cls_leastsquare, self.order_losstype, self.label_smooth),
        }
        # 构建optim
        if self.optimizer == 'rmsprop':
            optimizer_obj = partial(optim.RMSprop, momentum=0.5)
        elif self.optimizer == 'adam':
            optimizer_obj = partial(optim.Adam, betas=(0.5, 0.9))
        else:
            raise ValueError
        self.optimizers = {
            'rec': optimizer_obj(
                chain(
                    self.models['encoder'].parameters(),
                    self.models['decoder'].parameters()
                ),
                lr=self.rec_lr, weight_decay=self.l2
            ),
            'cls': optimizer_obj(
                self.models['disc'].parameters(), lr=self.cls_lr,
                weight_decay=self.l2
            )
        }
        self.scheduals = {
            'cls': optim.lr_scheduler.MultiStepLR(
                self.optimizers['cls'], self.schedual_stones,
                gamma=0.1
            ),
            'rec': optim.lr_scheduler.MultiStepLR(
                self.optimizers['rec'], self.schedual_stones,
                gamma=0.1
            )
        }

    def _forward_autoencode(self, batch_x, batch_y):
        ''' autoencode进行训练的部分 '''
        with torch.enable_grad():
            # encoder
            if self.denoise is not None and self.denoise > 0.0:
                noise = torch.randn(*batch_x.shape).to(batch_x) * self.denoise
                batch_x_noise = batch_x + noise
            else:
                batch_x_noise = batch_x

            hidden = self.models['encoder'](batch_x_noise)
            # decoder
            batch_x_rec = self.models['decoder'](hidden)
            # reconstruction losses
            recon_loss = self.criterions['rec'](batch_x_rec, batch_x)
            all_loss = recon_loss.clone()
            if self.phase == "iter_train":
                # discriminator, but not training
                logit = self.models['disc'](hidden[:, self.be_num:])
                # discriminator loss
                ad_loss_args = [None] * 5 + [self.cls_weight, self.order_weight]
                if self.cls_weight > 0.0:
                    ad_loss_args[0] = logit[:, :self.batch_label_num]
                    ad_loss_args[1] = batch_y[:, 1]
                if self.order_weight > 0.0:
                    ad_loss_args[2] = logit[:, -1]
                    ad_loss_args[3] = batch_y[:, 0]
                if self.use_batch_for_order:
                    ad_loss_args[4] = batch_y[:, 1]
                adversarial_loss = self.criterions['cls'](*ad_loss_args)
                # 组合这些loss来得到最终计算梯度使用的loss
                # 分类做的不好，说明这些维度中没有批次的信息，批次的信息都在后面的维度中
                all_loss -= self.disc_weight[self.e] * adversarial_loss

        all_loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(
                chain(
                    self.models['encoder'].parameters(),
                    self.models['decoder'].parameters()
                ), max_norm=1
            )
        self.optimizers['rec'].step()
        if self.phase == 'iter_train':
            return recon_loss, adversarial_loss
        return recon_loss, torch.tensor(0.)

    def _forward_discriminate(self, batch_x, batch_y):
        ''' discriminator进行训练的部分 '''
        with torch.no_grad():
            if self.denoise is not None and self.denoise > 0.0:
                noise = torch.randn(*batch_x.shape).to(batch_x) * self.denoise
                batch_x_noise = batch_x + noise
            else:
                batch_x_noise = batch_x

            hidden = self.models['encoder'](batch_x_noise)
        with torch.enable_grad():
            logit = self.models['disc'](hidden[:, self.be_num:])
            ad_loss_args = [None] * 5 + [self.cls_weight, self.order_weight]
            if self.cls_weight > 0.0:
                ad_loss_args[0] = logit[:, :self.batch_label_num]
                ad_loss_args[1] = batch_y[:, 1]
            if self.order_weight > 0.0:
                ad_loss_args[2] = logit[:, -1]
                ad_loss_args[3] = batch_y[:, 0]
            if self.use_batch_for_order:
                ad_loss_args[4] = batch_y[:, 1]
            adversarial_loss = self.criterions['cls'](*ad_loss_args)
        adversarial_loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(
                self.models['disc'].parameters(), max_norm=1)
        self.optimizers['cls'].step()
        return adversarial_loss


def main():
    from config import Config

    # config
    config = Config()
    config.show()

    # ----- 读取数据 -----
    pre_transfer = Normalization(config.args.data_normalization)
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
        config.args.be_num, batch_label_num=subject_dat.num_batch_labels,
        lrs=config.args.ae_disc_lr, bs=config.args.batch_size,
        nw=config.args.num_workers, epoch=config.args.epoch,
        device=torch.device('cuda:0'), l2=config.args.l2, clip_grad=True,
        ae_disc_train_num=config.args.ae_disc_train_num,
        disc_weight=config.args.disc_weight,
        label_smooth=config.args.label_smooth,
        train_with_qc=config.args.train_data == 'all',
        spectral_norm=config.args.spectral_norm,
        schedual_stones=config.args.schedual_stones,
        cls_leastsquare=config.args.cls_leastsquare,
        order_losstype=config.args.order_losstype,
        cls_order_weight=config.args.cls_order_weight,
        use_batch_for_order=config.args.use_batch_for_order,
        visdom_port=config.args.visdom_port,
        decoder_hiddens=config.args.ae_units,
        encoder_hiddens=config.args.ae_units[::-1],
        disc_hiddens=config.args.disc_units,
        early_stop=config.args.early_stop,
        net_type=config.args.net_type,
        resnet_bottle_num=config.args.resnet_bottle_num,
        optimizer=config.args.optim,
        denoise=config.args.denoise,
        reconst_loss=config.args.reconst_loss,
        disc_weight_epoch=config.args.disc_weight_epoch,
        early_stop_check_num=config.args.early_stop_check_num
    )
    if config.args.load_model != '':
        trainer.load_model(config.args.load_model)
    if config.args.early_stop:
        best_models, hist, early_stop_objs = trainer.fit(datas)
    else:
        best_models, hist = trainer.fit(datas)
    print('')

    # 保存结果
    if os.path.exists(config.args.save):
        dirname = input("%s has been already exists, please input New: " %
                        config.args.save)
    else:
        os.makedirs(config.args.save)
        dirname = config.args.save
    torch.save(best_models, os.path.join(dirname, 'models.pth'))
    pd.DataFrame(hist).to_csv(os.path.join(dirname, 'train.csv'))
    config.save(os.path.join(dirname, 'config.json'))
    if config.args.early_stop:
        with open(os.path.join(dirname, 'early_stop_info.json'), 'w') as f:
            json.dump(early_stop_objs, f)
    

if __name__ == "__main__":
    main()
