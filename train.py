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
from generate import generate


class BatchEffectTrainer:
    def __init__(
        self, in_features, bottle_num, no_be_num, batch_label_num=None,
        lrs=0.01, bs=64, nw=6, epoch=100, device=torch.device('cuda:0'),
        l2=0.0, clip_grad=False, ae_disc_train_num=(1, 1),
        ae_disc_weight=(1.0, 1.0), label_smooth=0.2,
        train_with_qc=False, spectral_norm=False, schedual_stones=[3000],
        cls_leastsquare=False, order_leastsquare=False,
        cls_order_weight=(1.0, 1.0), use_batch_for_order=True,
        visdom_port=8097, encoder_hiddens=[300, 300, 300],
        decoder_hiddens=[300, 300, 300], disc_hiddens=[300, 300],
        early_stop=False, net_type='simple', resnet_bottle_num=50,
        optimizer='rmsprop', denoise=0.1
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

        # 得到两个loss
        self.cls_weight, self.order_weight = cls_order_weight
        self.criterions = {
            'reconstruction': nn.MSELoss(),
            'adversarial_train': ClsOrderLoss(
                int(self.cls_weight != 0), int(self.order_weight != 0),
                cls_leastsquare, order_leastsquare, label_smooth),
            'adversarial_ae': ClsOrderLoss(
                self.cls_weight, self.order_weight, cls_leastsquare,
                order_leastsquare, label_smooth),
        }
        # according to cls_order_weight, choose classification criterion
        if self.cls_weight > 0.0 and self.order_weight > 0.0:
            logit_dim = batch_label_num + 1
        elif self.cls_weight > 0.0:
            logit_dim = batch_label_num
        elif self.order_weight > 0.0:
            logit_dim = 1
        else:
            raise ValueError(
                'cls_weight and order_weight must not be all zero')

        # 得到3个模型
        if net_type == 'simple':
            self.models = {
                'encoder': SimpleCoder(
                    [in_features] + encoder_hiddens + [bottle_num]).to(device),
                'decoder': SimpleCoder(
                    [bottle_num] + decoder_hiddens + [in_features]).to(device),
                'discriminator': SimpleCoder(
                    [no_be_num] + disc_hiddens + [logit_dim]).to(device)
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
                'discriminator': ResNet(
                    [no_be_num] + disc_hiddens + [logit_dim],
                    resnet_bottle_num
                ).to(device)
            }

        # 得到两个optim
        if not isinstance(lrs, (tuple, list)):
            lrs = [lrs] * 2
        if optimizer == 'rmsprop':
            optimizer_obj = partial(optim.RMSprop, momentum=0.5)
        elif optimizer == 'adam':
            optimizer_obj = partial(optim.Adam, betas=(0.5, 0.999))
        else:
            raise ValueError
        self.optimizers = {
            'autoencode': optimizer_obj(
                chain(
                    self.models['encoder'].parameters(),
                    self.models['decoder'].parameters()
                ),
                lr=lrs[0], weight_decay=l2
            ),
            'discriminate': optimizer_obj(
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
        self.history = {
            'discriminate_loss_train': [], 'discriminate_loss_fixed': [],
            'reconstruction_loss': [], 'qc_loss': [], 'qc_distance': []
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
        self.train_with_qc = train_with_qc
        self.use_batch_for_order = use_batch_for_order
        self.early_stop = early_stop
        self.denoise = denoise
        self.batch_label_num = batch_label_num

        # 可视化工具
        self.visobj = VisObj(visdom_port)

        # 提前停止使用
        self.early_stop_objs = {
            'best_epoch': -1, 'best_qc_loss': 1000, 'best_qc_distance': 1000,
            'best_models': None, 'index': 0, 'best_score': 2000
        }

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
        for e in tqdm(range(self.epoch), 'Epoch: '):
            ## train phase
            # 实例化3个loss对象，用于计算epoch loss
            ad_loss_train_obj = mm.Loss()
            ad_loss_fixed_obj = mm.Loss()
            recon_loss_obj = mm.Loss()
            # 训练时需要将模型更改至训练状态
            for model in self.models.values():
                model.train()
            # 循环每个batch进行训练
            for batch_x, batch_y in tqdm(dataloaders['train'], 'Train Batch: '):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
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

            # 记录epoch loss
            self.history['discriminate_loss_train'].append(
                ad_loss_train_obj.value())
            self.history['discriminate_loss_fixed'].append(
                ad_loss_fixed_obj.value())
            self.history['reconstruction_loss'].append(recon_loss_obj.value())
            # 可视化epoch loss
            # 因为两组loss的取值范围相差太大，所以分开来显示
            self.visobj.add_epoch_loss(
                winname='disc_losses',
                disc_train=self.history['discriminate_loss_train'][-1],
                disc_fixed=self.history['discriminate_loss_fixed'][-1],
            )
            self.visobj.add_epoch_loss(
                winname='recon_losses',
                recon_loss=self.history['reconstruction_loss'][-1]
            )

            ## valid phase

            # 使用当前训练的模型去得到去批次结果
            all_data = ConcatData(datas['subject'], datas['qc'])
            all_reses_dict, qc_loss = generate(
                self.models, all_data, no_be_num=self.no_be_num,
                device=self.device, bs=self.bs, nw=self.nw,
                verbose=False, ica=False, compute_qc_loss=True
            )
            # 对数据进行对应的pca
            subject_pca, qc_pca = pca_for_dict(all_reses_dict)
            # plot pca
            pca_plot(subject_pca, qc_pca)
            # display in visdom
            self.visobj.vis.matplot(plt, win='PCA', opts={'title': 'PCA'})
            plt.close()

            ## early stopping
            qc_dist = mm.mean_distance(qc_pca['recons_no_batch'])
            self.history['qc_loss'].append(qc_loss)
            self.history['qc_distance'].append(qc_dist)
            self.visobj.add_epoch_loss(
                winname='qc_loss', qc_loss=qc_loss)
            self.visobj.add_epoch_loss(
                winname='qc_distance', qc_dist=qc_dist
            )
            if self.early_stop and e >= self.epoch - 100:
                early_stop_index = self.check_qc(e, qc_dist, qc_loss)
        if self.early_stop:
            print('')
            print('The best epoch is %d' %
                self.early_stop_objs['best_epoch'])
            print('The best qc loss is %.4f' %
                self.early_stop_objs['best_qc_loss'])
            print('The best qc distance is %.4f' %
                self.early_stop_objs['best_qc_distance'])
            for k, v in self.models.items():
                v.load_state_dict(self.early_stop_objs['best_models'][k])
            self.models.update(self.early_stop_objs)

        return self.models, self.history

    def check_qc(self, e, qc_dist, qc_loss):
        early_stop_score = qc_dist + qc_loss * 100
        if early_stop_score < self.early_stop_objs['best_score']:
            self.early_stop_objs['best_epoch'] = e
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

        #  if self.early_stop_objs['index'] > 200:
            #  return True
        return False

    def _forward_autoencode(self, batch_x, batch_y):
        ''' autoencode进行训练的部分 '''
        with torch.enable_grad():
            # encoder
            if self.denoise is not None and self.denoise > 0.0:
                noise = torch.randn(*batch_x.shape).to(batch_x) * self.denoise
                batch_x_noise = batch_x + noise
            else:
                batch_x_noise = batch_x

            encoder_hiddens = self.models['encoder'](batch_x_noise)
            hidden = encoder_hiddens
            # decoder
            decoder_hiddens = self.models['decoder'](hidden)
            batch_x_recon = decoder_hiddens
            # discriminator, but not training
            logit = self.models['discriminator'](hidden[:, :self.no_be_num])
            # reconstruction losses
            recon_loss = self.criterions['reconstruction'](
                batch_x_recon, batch_x)
            # discriminator loss
            ad_loss_args = [None] * 5
            if self.cls_weight > 0.0:
                ad_loss_args[0] = logit[:, :self.batch_label_num]
                ad_loss_args[1] = batch_y[:, 1]
            if self.order_weight > 0.0:
                ad_loss_args[2] = logit[:, -1]
                ad_loss_args[3] = batch_y[:, 0]
            if self.use_batch_for_order:
                ad_loss_args[4] = batch_y[:, 1]
            adversarial_loss = self.criterions['adversarial_ae'](*ad_loss_args)
            # 组合这些loss来得到最终计算梯度使用的loss
            # 分类做的不好，说明这些维度中没有批次的信息，批次的信息都在后面的维度中
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
        return recon_loss, adversarial_loss

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
            logit = self.models['discriminator'](hidden[:, :self.no_be_num])
            ad_loss_args = [None] * 5
            if self.cls_weight > 0.0:
                ad_loss_args[0] = logit[:, :self.batch_label_num]
                ad_loss_args[1] = batch_y[:, 1]
            if self.order_weight > 0.0:
                ad_loss_args[2] = logit[:, -1]
                ad_loss_args[3] = batch_y[:, 0]
            if self.use_batch_for_order:
                ad_loss_args[4] = batch_y[:, 1]
            adversarial_loss = self.criterions['adversarial_train'](
                *ad_loss_args)
        adversarial_loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(
                self.models['discriminator'].parameters(), max_norm=1)
        self.optimizers['discriminate'].step()
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
        config.args.no_batch_num, batch_label_num=subject_dat.num_batch_labels,
        lrs=config.args.ae_disc_lr, bs=config.args.batch_size,
        nw=config.args.num_workers, epoch=config.args.epoch,
        device=torch.device('cuda:0'), l2=config.args.l2, clip_grad=True,
        ae_disc_train_num=config.args.ae_disc_train_num,
        ae_disc_weight=config.args.ae_disc_weight,
        label_smooth=config.args.label_smooth,
        train_with_qc=config.args.train_data == 'all',
        spectral_norm=config.args.spectral_norm,
        schedual_stones=config.args.schedual_stones,
        cls_leastsquare=config.args.cls_leastsquare,
        order_leastsquare=config.args.order_leastsquare,
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
        denoise=config.args.denoise
    )

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
    

if __name__ == "__main__":
    main()
