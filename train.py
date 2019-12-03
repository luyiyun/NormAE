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
from networks import SimpleCoder, OrderLoss
from transfer import Normalization
import metrics as mm
from visual import VisObj, pca_for_dict, pca_plot


class BatchEffectTrainer:
    def __init__(
        self, in_features, batch_label_num, device, pre_transfer, opts
        #  bottle_num, be_num, batch_label_num=None,
        #  lrs=0.01, bs=64, nw=6, epoch=(200, 100, 1000),
        #  device=torch.device('cuda:0'), l2=0.0, clip_grad=False,
        #  ae_disc_train_num=(1, 1), disc_weight=1.0,
        #  label_smooth=0.2, train_with_qc=False, spectral_norm=False,
        #  schedual_stones=[3000], cls_leastsquare=False, order_losstype=False,
        #  cls_order_bio_weight=(1.0, 1.0, 1.0), use_batch_for_order=True,
        #  visdom_port=8097, encoder_hiddens=[300, 300, 300],
        #  decoder_hiddens=[300, 300, 300], disc_hiddens=[300, 300],
        #  early_stop=False, net_type='simple', resnet_bottle_num=50,
        #  optimizer='rmsprop', denoise=0.1, reconst_loss='mae',
        #  disc_weight_epoch=500, early_stop_check_num=100,
        #  dropouts=(0., 0., 0., 0., 0.), pre_transfer=None, visdom_env='main',
    ):

        # architecture
        self.in_features = in_features
        self.batch_label_num = batch_label_num
        self.device = device
        self.encoder_hiddens = opts.ae_encoder_units
        self.decoder_hiddens = opts.ae_decoder_units
        self.disc_b_hiddens = opts.disc_b_units
        self.disc_o_hiddens = opts.disc_o_units
        self.bottle_num = opts.bottle_num
        self.dropouts = opts.dropouts

        # loss
        #  self.cls_weight =
        #  , self.order_weight, self.bio_weight = cls_order_bio_weight
        self.use_batch_for_order = opts.use_batch_for_order
        self.lambda_b, self.lambda_o = opts.lambda_b, opts.lambda_o

        # optimizer
        self.lr_rec = opts.lr_rec
        self.lr_disc_b = opts.lr_disc_b
        self.lr_disc_o = opts.lr_disc_o
        #  self.lrs = [lrs] * 2 if isinstance(lrs, float) else lrs
        #  self.rec_lr, self.cls_lr = self.lrs
        #  self.l2, self.clip_grad = l2, clip_grad
        #  self.schedual_stones = schedual_stones
        #  self.optimizer = optimizer

        # training
        self.rec_epoch, self.cls_epoch, self.iter_epoch = opts.epoch
        #  self.epoch = sum(epoch)
        #  self.rec_train_num, self.cls_train_num = ae_disc_train_num
        self.bs, self.nw = opts.batch_size, opts.num_workers
        self.train_with_qc = opts.train_data == "all"
        #  self.train_with_qc = train_with_qc
        #  self.early_stop = early_stop

        # other
        self.visdom_port = opts.visdom_port
        self.visdom_env = opts.visdom_env
        self.pre_transfer = pre_transfer
        #  self.early_stop_check_num = early_stop_check_num

        # build model
        self._build_model()

        # training record
        self.history = {
            'disc_cls_loss': [], 'disc_order_loss': [], "disc_bio_loss": [],
            'adv_cls_loss': [], 'adv_order_loss': [], "adv_bio_loss": [],
            'rec_loss': [], 'qc_rec_loss': [], 'qc_distance': []
        }
        # visdom
        self.visobj = VisObj(self.visdom_port, env=self.visdom_env)
        #  self.visobj = Visdom(port=self.visdom_port, env=self.visdom_env)
        # early stop
        self.early_stop_objs = {
            'best_epoch': -1, 'best_qc_loss': 1000, 'best_qc_distance': 1000,
            'best_models': None, 'index': 0, 'best_score': 2000
        }

    def fit(self, datas):
        # get dataloaders
        train_data = data.ConcatDataset([datas['subject'], datas['qc']]) \
            if self.train_with_qc else datas['subject']
        dataloaders = {
            'train': data.DataLoader(train_data, batch_size=self.bs,
                                     num_workers=self.nw, shuffle=True),
            'qc': data.DataLoader(datas['qc'], batch_size=self.bs,
                                  num_workers=self.nw)
        }
        # begin training
        pbar = tqdm(total=self.epoch)
        for e in range(self.epoch):
            self.e = e
            if e < self.rec_epoch:
                self.phase = 'rec_pretrain'
            elif e < self.rec_epoch + self.cls_epoch:
                self.phase = 'cls_pretrain'
            else:
                self.phase = 'iter_train'
            pbar.set_description(self.phase)

            ## train phase
            for model in self.models.values():
                model.train()
            disc_cls_loss_obj = mm.Loss()
            disc_order_loss_obj = mm.Loss()
            adv_cls_loss_obj = mm.Loss()
            adv_order_loss_obj = mm.Loss()
            rec_loss_obj = mm.Loss()
            # 循环每个batch进行训练
            for batch_x, batch_y in tqdm(dataloaders['train'], 'Batch: '):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                bs0 = batch_x.size(0)
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad()
                if self.phase in ['cls_pretrain', 'iter_train']:
                    disc_cls_loss, disc_order_loss = \
                        self._forward_discriminate(batch_x, batch_y)
                    disc_cls_loss_obj.add(disc_cls_loss, bs0)
                    disc_order_loss_obj.add(disc_order_loss, bs0)
                if self.phase in ['rec_pretrain', 'iter_train']:
                    rec_loss, adv_cls_loss, adv_order_loss = \
                        self._forward_autoencode(batch_x, batch_y)
                    rec_loss_obj.add(rec_loss, bs0)
                    adv_cls_loss_obj.add(adv_cls_loss, bs0)
                    adv_order_loss_obj.add(adv_order_loss, bs0)
            #  if self.phase == 'iter_train':
            #      for sche in self.scheduals.values():
            #          sche.step()
            # record loss
            self.history['disc_cls_loss'].append(disc_cls_loss_obj.value())
            self.history['disc_order_loss'].append(disc_order_loss_obj.value())
            self.history['adv_cls_loss'].append(adv_cls_loss_obj.value())
            self.history['adv_order_loss'].append(adv_order_loss_obj.value())
            self.history['rec_loss'].append(rec_loss_obj.value())
            # visual epoch loss
            self.visobj.add_epoch_loss(
                winname='disc_losses',
                disc_cls_loss=self.history['disc_cls_loss'][-1],
                disc_order_loss=self.history['disc_order_loss'][-1],
                disc_bio_loss=self.history['disc_bio_loss'][-1],
                adv_cls_loss=self.history['adv_cls_loss'][-1],
                adv_order_loss=self.history['adv_order_loss'][-1],
                adv_bio_loss=self.history['adv_bio_loss'][-1]
            )
            self.visobj.add_epoch_loss(
                winname='recon_losses',
                recon_loss=self.history['rec_loss'][-1]
            )

            ## valid phase
            all_data = ConcatData(datas['subject'], datas['qc'])
            all_reses_dict, qc_loss = self.generate(
                all_data, verbose=False, compute_qc_loss=True)
            # pca
            subject_pca, qc_pca = pca_for_dict(all_reses_dict, 3)
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
            if e >= self.epoch - 200:
                self._check_qc(qc_dist, qc_loss)
            
            # progressbar
            pbar.update(1)
        pbar.close()

        # early stop information and save visdom env
        if self.visdom_env != 'main':
            self.visobj.vis.save([self.visdom_env])
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

    def generate(self, data_loader, verbose=True, compute_qc_loss=False):
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
                # return x and y
                x_ori.append(batch_x)
                ys.append(batch_y)
                # return latent representation
                batch_x = batch_x.to(self.device, torch.float)
                batch_y = batch_y.to(self.device, torch.float)
                hidden = self.models['encoder'](batch_x)
                codes.append(hidden)
                # return rec with and without batch effects
                batch_ys = [
                    torch.eye(self.cls_logit_dim)[batch_y[:, 1].long()].to(
                        hidden),
                    batch_y[:, [0]]
                ]
                batch_ys = torch.cat(batch_ys, dim=1)
                hidden_be = hidden + self.models['map'](batch_ys)
                x_rec.append(self.models['decoder'](hidden_be))
                x_rec_nobe.append(self.models['decoder'](hidden))
                # return qc loss
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

        # return dataframe
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
                    res[k] = self.pre_transfer.inverse_transform(
                        res[k], None)[0]
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
        logit_dim = self.batch_label_num + 1
        # build models
        self.models = {
            'encoder': SimpleCoder(
                [self.in_features] + self.encoder_hiddens +
                [self.bottle_num], dropout=self.dropouts[0]
            ).to(self.device),
            'decoder': SimpleCoder(
                [self.bottle_num] + self.decoder_hiddens +
                [self.in_features], dropout=self.dropouts[1],
                final_act=None
            ).to(self.device),
            'map': SimpleCoder(
                [logit_dim] + [500] + [self.bottle_num],
            ).to(self.device),
            'disc_b': SimpleCoder(
                [self.bottle_num] + self.disc_b_hiddens +
                [self.batch_label_num], bn=True, dropout=self.dropouts[2]
            ).to(self.device),
            "disc_o": SimpleCoder(
                [self.bottle_num] + self.disc_o_hiddens + [1],
                bn=False, dropout=self.dropouts[3]
            ).to(self.device)
        }
        # build loss
        self.criterions = {
            'cls': nn.CrossEntropyLoss(),
            'order': OrderLoss(),
            "rec": nn.L1Loss()
        }
        # build optim
        optimizer_obj = partial(optim.Adam, betas=(0.5, 0.9))
        self.optimizers = {
            'rec': optimizer_obj(
                chain(
                    self.models['encoder'].parameters(),
                    self.models['decoder'].parameters(),
                    self.models['map'].parameters()
                ), lr=self.lr_rec
            ),
            "cls": optimizer_obj(self.models['disc_cls'].parameters(),
                                 lr=self.lr_disc_b),
            "order": optimizer_obj(self.models['disc_order'].parameters(),
                                   lr=self.lr_disc_o)
        }
        #  self.scheduals = {
        #      'rec': optim.lr_scheduler.MultiStepLR(
        #          self.optimizers['rec'], self.schedual_stones,
        #          gamma=0.1
        #      ),
        #  }
        #  if self.cls_logit_dim > 0:
        #      self.scheduals['cls'] = optim.lr_scheduler.MultiStepLR(
        #          self.optimizers['cls'], self.schedual_stones,
        #          gamma=0.1
        #      )
        #  if self.order_logit_dim > 0:
        #      self.scheduals['order'] = optim.lr_scheduler.MultiStepLR(
        #          self.optimizers['order'], self.schedual_stones,
        #          gamma=0.1
        #      )
        #  if self.bio_logit_dim > 0:
        #      self.scheduals["bio"] = optim.lr_scheduler.MultiStepLR(
        #          self.optimizers['bio'], self.schedual_stones,
        #          gamma=0.1
        #      )

    def _forward_autoencode(self, batch_x, batch_y):
        ''' autoencode进行训练的部分 '''
        res = [None, None, None, None]
        with torch.enable_grad():
            # encoder
            #  if self.denoise is not None and self.denoise > 0.0:
            #      noise = torch.randn(*batch_x.shape).to(batch_x) * self.denoise
            #      batch_x_noise = batch_x + noise
            #      batch_x_noise = batch_x_noise.clamp(0, 1)
            #  else:
            #      batch_x_noise = batch_x
            hidden = self.models['encoder'](batch_x)
            # decoder
            batch_ys = [
                torch.eye(self.cls_logit_dim)[batch_y[:, 1].long()].to(hidden),
                batch_y[:, [0]]

            ]
            batch_ys = torch.cat(batch_ys, dim=1)
            hidden_be = hidden + self.models['map'](batch_ys)
            batch_x_rec = self.models['decoder'](hidden_be)
            # reconstruction losses
            recon_loss = self.criterions['rec'](batch_x_rec, batch_x)
            # adversarial regularizations (disc_b)
            logit_cls = self.models['disc_b'](hidden)
            loss_cls = self.criterions['cls'](logit_cls, batch_y[:, 1])
            all_loss -= self.lambda_b * loss_cls
            # adversarial regularizations (disc_o)
            if self.use_batch_for_order:
                group = batch_y[:, 1]
            else:
                group = None
            logit_order = self.models['disc_order'](hidden)
            loss_order = self.criterions['order'](logit_order, batch_y[:, 0],
                                                  group)
            all_loss -= self.lambda_o * loss_order
        all_loss.backward()
        self.optimizers['rec'].step()
        return [recon_loss, loss_cls, loss_order]

    def _forward_discriminate(self, batch_x, batch_y):
        with torch.no_grad():
            #  if self.denoise is not None and self.denoise > 0.0:
            #      noise = torch.randn(*batch_x.shape).to(batch_x) * self.denoise
            #      batch_x_noise = batch_x + noise
            #      batch_x_noise = batch_x_noise.clamp(0, 1)
            #  else:
            #      batch_x_noise = batch_x
            hidden = self.models['encoder'](batch_x)
        with torch.enable_grad():
            # disc_b
            logit_cls = self.models['disc_cls'](hidden)
            adv_cls_loss = self.criterions['cls'](logit_cls, batch_y[:, 1])
            adv_cls_loss.backward()
            self.optimizers['cls'].step()
            logit_order = self.models['disc_order'](hidden)
            # disc_o
            if self.use_batch_for_order:
                group = batch_y[:, 1]
            else:
                group = None
            adv_order_loss = self.criterions['order'](
                logit_order, batch_y[:, 0], group)
            adv_order_loss.backward()
        return [adv_cls_loss, adv_order_loss]


