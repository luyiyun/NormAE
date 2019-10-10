#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

from datasets import get_metabolic_data
from transfer import Normalization
from networks import OrderLoss
from metrics import Loss
from visual import VisObj


def dcg(ys_true, ys_pred):
    pred_index = ys_pred.argsort(descending=True, dim=0)
    ys_true_sorted = ys_true[pred_index]
    ret = 0
    for i, l in enumerate(ys_true_sorted, 1):
        ret += (2 ** l - 1) / torch.log2(torch.tensor(1 + i).to(ys_pred))
    return ret


def ndcg(ys_true, ys_pred):
    ideal_dcg = dcg(ys_true, ys_true)
    pred_dcg = dcg(ys_true, ys_pred)
    return pred_dcg / ideal_dcg

class NDCG:
    def __init__(self):
        self.preds = []
        self.targets = []

    def add(self, pred, target):
        self.preds.append(pred)
        self.targets.append(target)

    def value(self):
        self.preds = torch.cat(self.preds, dim=0)
        self.targets = torch.cat(self.targets, dim=0)
        return ndcg(self.preds, self.targets).detach().cpu().numpy()

class RankPredictor:
    def __init__(self, loss_type, in_f, lr, device=torch.device('cuda:0')):
        self.model = nn.Sequential(
            nn.Linear(in_f, 2000),
            nn.BatchNorm1d(2000),
            nn.LeakyReLU(),
            nn.Linear(2000, 2000),
            nn.BatchNorm1d(2000),
            nn.LeakyReLU(),
            nn.Linear(2000, 1)
        ).to(device)
        self.criterion = OrderLoss(loss_type)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr, betas=(0.5, 0.999))

        self.history = {'train_loss': [], 'train_ndcg': [],
                        'valid_loss': [], 'valid_ndcg': []}

        self.device = device
        self.visobj = VisObj()


    def fit(self, dats, bs, nw, epoch):
        loaders = {}
        for k, v in dats.items():
            loaders[k] = data.DataLoader(
                v, batch_size=bs, num_workers=nw, shuffle=(k=='train'))


        for e in tqdm(range(epoch), 'Epoch: '):
            loss_objs = {'train': Loss(), 'valid': Loss()}
            ndcg_objs = {'train': NDCG(), 'valid': NDCG()}
            for phase in ['train', 'valid']:
                for batch_x, batch_y in tqdm(loaders[phase], 'batch: '):
                    batch_x = batch_x.to(self.device).float()
                    batch_y = batch_y.to(self.device).float()

                    with torch.set_grad_enabled(phase == 'train'):
                        pred_score = self.model(batch_x).squeeze()
                        loss = self.criterion(pred_score, batch_y[:, 0])
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    with torch.no_grad():
                        loss_objs[phase].add(loss, batch_x.size(0))              
                        ndcg_objs[phase].add(pred_score, batch_y[:, 0])
                
            self.history['train_loss'].append(loss_objs['train'].value())
            self.history['valid_loss'].append(loss_objs['valid'].value())
            self.history['train_ndcg'].append(ndcg_objs['train'].value())
            self.history['valid_ndcg'].append(ndcg_objs['valid'].value())

            self.visobj.add_epoch_loss(
                'epoch_loss', train=self.history['train_loss'][-1],
                valid=self.history['valid_loss'][-1]
            )
            self.visobj.add_epoch_loss(
                'epoch_ndcg', train=self.history['train_ndcg'][-1],
                valid=self.history['valid_ndcg'][-1]
            )

        return self.history

def main():
    from config import Config

    loss_type = sys.argv[1]

    pre_transfer = Normalization('standard')
    subject_dat, qc_dat = get_metabolic_data(
        Config.metabolic_x_files['Amide'],
        Config.metabolic_y_files['Amide'],
        pre_transfer=pre_transfer
    )
    datas = {'train': subject_dat, 'valid': qc_dat}

    trainer = RankPredictor(
        loss_type='listnet', in_f=subject_dat.num_features,
        lr=0.0001, device=torch.device('cuda:0')
    )
    history = trainer.fit(datas, bs=64, nw=12, epoch=1000)

    with open('./RESULTS/'+loss_type+'.json', w) as f:
        json.dump(history, f)

if __name__ == '__main__':
    main()
