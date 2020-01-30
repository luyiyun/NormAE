""" NormAE estimator, sklearn style """
import copy
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets import ConcatData
from networks import SimpleCoder, OrderLoss
import metrics as mm
from visual import VisObj, pca_for_dict, pca_plot


class BatchEffectTrainer:
    def __init__(
        self, in_features, batch_label_num, device, pre_transfer, opts
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
        self.use_batch_for_order = opts.use_batch_for_order
        self.lambda_b, self.lambda_o = opts.lambda_b, opts.lambda_o

        # optimizer
        self.lr_rec = opts.lr_rec
        self.lr_disc_b = opts.lr_disc_b
        self.lr_disc_o = opts.lr_disc_o

        # training
        self.epoch = sum(opts.epoch)
        self.rec_epoch, self.disc_epoch, self.iter_epoch = opts.epoch
        self.bs, self.nw = opts.batch_size, opts.num_workers
        self.train_with_qc = opts.train_data == "all"

        # other
        self.visdom_port = opts.visdom_port
        self.visdom_env = opts.visdom_env
        self.pre_transfer = pre_transfer

        # build model
        self._build_model()

        # training record
        self.history = {
            'disc_b_loss': [], 'disc_o_loss': [], 'adv_b_loss': [],
            'adv_o_loss': [], 'rec_loss': [], 'qc_rec_loss': [],
            'qc_distance': []
        }
        # visdom
        self.visobj = VisObj(self.visdom_port, env=self.visdom_env)
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
            elif e < self.rec_epoch + self.disc_epoch:
                self.phase = 'disc_pretrain'
            else:
                self.phase = 'iter_train'
            pbar.set_description(self.phase)

            # --- train phase ---
            for model in self.models.values():
                model.train()
            disc_b_loss_obj = mm.Loss()
            disc_o_loss_obj = mm.Loss()
            adv_b_loss_obj = mm.Loss()
            adv_o_loss_obj = mm.Loss()
            rec_loss_obj = mm.Loss()
            for batch_x, batch_y in tqdm(dataloaders['train'], 'Batch: '):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                bs0 = batch_x.size(0)
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad()
                if self.phase in ['disc_pretrain', 'iter_train']:
                    disc_b_loss, disc_o_loss = \
                        self._forward_discriminate(batch_x, batch_y)
                    disc_b_loss_obj.add(disc_b_loss, bs0)
                    disc_o_loss_obj.add(disc_o_loss, bs0)
                if self.phase in ['rec_pretrain', 'iter_train']:
                    rec_loss, adv_b_loss, adv_o_loss = \
                        self._forward_autoencode(batch_x, batch_y)
                    rec_loss_obj.add(rec_loss, bs0)
                    adv_b_loss_obj.add(adv_b_loss, bs0)
                    adv_o_loss_obj.add(adv_o_loss, bs0)
            # record loss
            self.history['disc_b_loss'].append(disc_b_loss_obj.value())
            self.history['disc_o_loss'].append(disc_o_loss_obj.value())
            self.history['adv_b_loss'].append(adv_b_loss_obj.value())
            self.history['adv_o_loss'].append(adv_o_loss_obj.value())
            self.history['rec_loss'].append(rec_loss_obj.value())
            # visual epoch loss
            self.visobj.add_epoch_loss(
                winname='disc_losses',
                disc_b_loss=self.history['disc_b_loss'][-1],
                disc_o_loss=self.history['disc_o_loss'][-1],
                adv_b_loss=self.history['adv_b_loss'][-1],
                adv_o_loss=self.history['adv_o_loss'][-1],
            )
            self.visobj.add_epoch_loss(
                winname='recon_losses',
                recon_loss=self.history['rec_loss'][-1]
            )

            # --- valid phase ---
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

            # --- early stopping ---
            qc_dist = mm.mean_distance(qc_pca['Rec_nobe'])
            self.history['qc_rec_loss'].append(qc_loss)
            self.history['qc_distance'].append(qc_dist)
            self.visobj.add_epoch_loss(winname='qc_rec_loss', qc_loss=qc_loss)
            self.visobj.add_epoch_loss(winname='qc_distance', qc_dist=qc_dist)
            if e >= (self.epoch - 200):
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
                    torch.eye(self.batch_label_num)[batch_y[:, 1].long()].to(
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
                        index=data_loader.dataset.y_df.index,
                        columns=data_loader.dataset.y_df.columns
                    )
                elif k != 'Codes':
                    res[k] = pd.DataFrame(
                        v.detach().cpu().numpy(),
                        index=data_loader.dataset.x_df.index,
                        columns=data_loader.dataset.x_df.columns
                    )
                    res[k] = self.pre_transfer.inverse_transform(
                        res[k], None)[0]
                else:
                    res[k] = pd.DataFrame(
                        v.detach().cpu().numpy(),
                        index=data_loader.dataset.x_df.index,
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
            'disc_b': nn.CrossEntropyLoss(),
            'disc_o': OrderLoss(),
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
            "disc_b": optimizer_obj(self.models['disc_b'].parameters(),
                                    lr=self.lr_disc_b),
            "disc_o": optimizer_obj(self.models['disc_o'].parameters(),
                                    lr=self.lr_disc_o)
        }

    def _forward_autoencode(self, batch_x, batch_y):
        ''' autoencode进行训练的部分 '''
        res = [None, None, None]
        with torch.enable_grad():
            all_loss = 0.
            hidden = self.models['encoder'](batch_x)
            # decoder
            batch_ys = [
                torch.eye(self.batch_label_num)[batch_y[:, 1].long()].to(
                    hidden),
                batch_y[:, [0]]

            ]
            batch_ys = torch.cat(batch_ys, dim=1)
            hidden_be = hidden + self.models['map'](batch_ys)
            batch_x_rec = self.models['decoder'](hidden_be)
            # reconstruction losses
            recon_loss = self.criterions['rec'](batch_x_rec, batch_x)
            all_loss += recon_loss
            res[0] = recon_loss
            if self.phase == "iter_train":
                # adversarial regularizations (disc_b)
                logit_b = self.models['disc_b'](hidden)
                loss_b = self.criterions['disc_b'](logit_b,
                                                   batch_y[:, 1].long())
                all_loss -= self.lambda_b * loss_b
                res[1] = loss_b
                # adversarial regularizations (disc_o)
                if self.use_batch_for_order:
                    group = batch_y[:, 1]
                else:
                    group = None
                logit_o = self.models['disc_o'](hidden)
                loss_o = self.criterions['disc_o'](logit_o, batch_y[:, 0],
                                                   group)
                all_loss -= self.lambda_o * loss_o
                res[2] = loss_o
        all_loss.backward()
        nn.utils.clip_grad_norm_(
            chain(self.models["encoder"].parameters(),
                  self.models["decoder"].parameters(),
                  self.models["map"].parameters()),
            max_norm=1
        )
        self.optimizers['rec'].step()
        return res

    def _forward_discriminate(self, batch_x, batch_y):
        with torch.no_grad():
            hidden = self.models['encoder'](batch_x)
        with torch.enable_grad():
            # disc_b
            logit_b = self.models['disc_b'](hidden)
            adv_b_loss = self.criterions['disc_b'](logit_b,
                                                   batch_y[:, 1].long())
            adv_b_loss.backward()
            nn.utils.clip_grad_norm_(self.models["disc_b"].parameters(),
                                     max_norm=1)
            self.optimizers['disc_b'].step()
            # disc_o
            logit_o = self.models['disc_o'](hidden)
            if self.use_batch_for_order:
                group = batch_y[:, 1]
            else:
                group = None
            adv_o_loss = self.criterions['disc_o'](
                logit_o, batch_y[:, 0], group)
            adv_o_loss.backward()
            nn.utils.clip_grad_norm_(self.models["disc_b"].parameters(),
                                     max_norm=1)
            self.optimizers['disc_o'].step()
        return [adv_b_loss, adv_o_loss]
