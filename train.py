import os
import copy
import json
from itertools import chain

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import progressbar as pb
import pandas as pd

from datasets import MetaBatchEffect
from networks import Coder, RankLoss
import transfer as T
import metrics as mm


class NoneScheduler:
    def __init__(self, optimizer):
        pass

    def step(self):
        pass


class BatchEffectTrainer:
    def __init__(
        self, models, criterions, optimizers, scheduler=NoneScheduler(None),
        epoch=100, device=torch.device('cuda:0'), l2=0.0, clip_grad=False,
        ae_disc_train_num=(1, 2), no_best=False
    ):
        # 初始化使用的标准和保存的参数
        if not no_best:
            self.best_metric = 0.0
            self.best_models_wts = {
                k: copy.deepcopy(v.state_dict())
                for k, v in models.items()
            }

        # 得到3个模型
        self.models = {k: v.to(device) for k, v in models.items()}

        # 得到两个loss
        self.criterions = criterions

        # 得到两个optim
        self.optimizers = optimizers

        # 初始化结果记录
        self.history = {
            'train_autoencode_loss': [], 'train_discriminate_loss': [],
            'qc_discriminate_loss': [], 'qc_distance': []
        }

        # 属性化
        self.epoch = epoch
        self.device = device
        self.scheduler = scheduler
        self.clip_grad = clip_grad
        self.autoencode_train_num, self.discriminate_train_num = \
            ae_disc_train_num
        self.no_best = no_best

    def fit(self, dataloaders):
        epoch_metric = 0.0
        for e in range(self.epoch):
            for phase in ['train', 'qc']:
                if phase == 'train':
                    if isinstance(
                        self.scheduler, optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.scheduler.step(epoch_metric)
                    else:
                        self.scheduler.step()
                    for model in self.models.values():
                        model.train()
                    prefix = "Train: "
                else:
                    for model in self.models.values():
                        model.eval()
                    prefix = "Valid: "
                # progressbar
                batch_text = pb.FormatCustomText(
                    'Batch: %(batch)s  Loss: %(loss).4f',
                    dict(batch='discriminate', loss=0.)
                )
                widgets = [
                    prefix, " ", pb.Percentage(),
                    ' ', pb.SimpleProgress(
                        format='(%s)' % pb.SimpleProgress.DEFAULT_FORMAT
                    ),
                    ' ', pb.Bar(),
                    ' ', pb.Timer(),
                    ' ', pb.AdaptiveETA(),
                    ' ', batch_text
                ]
                iterator = pb.progressbar(dataloaders[phase], widgets=widgets)

                # 创建两个计数的变量，用来决定进行autoencode还是discriminator的训练
                discriminate_index, autoencode_index = 0, 0

                # train和valid阶段创建不同的metrics来记录loss和评价指标
                if phase == 'train':
                    self.metrics = {
                        'train_autoencode_loss': mm.Loss(),
                        'train_discriminate_loss': mm.Loss(),
                    }
                else:
                    self.metrics = {
                        'qc_discriminate_loss': mm.Loss(),
                        'qc_distance': mm.MeanDistance()
                    }

                # 对每个batch进行相应的操作
                batch_phase = 'discriminate'
                for batch_x, batch_y in iterator:
                    batch_x = batch_x.to(self.device, torch.float)
                    batch_y = batch_y.to(self.device, torch.float)
                    if phase == 'train':
                        for optimizer in self.optimizers.values():
                            optimizer.zero_grad()
                        if batch_phase == 'discriminate':
                            self._forward_discriminate(
                                batch_x, batch_y, batch_text)
                            discriminate_index += 1
                            if (
                                discriminate_index >=
                                self.discriminate_train_num
                            ):
                                batch_phase = 'autoencode'
                        elif batch_phase == 'autoencode':
                            self._forward_autoencode(
                                batch_x, batch_y, batch_text)
                            autoencode_index += 1
                            if (
                                autoencode_index >=
                                self.autoencode_train_num
                            ):
                                batch_phase = 'discriminate'
                    else:
                        self._forward_qc(batch_x, batch_y, batch_text)

                if phase == 'train':
                    self.history['train_autoencode_loss'].append(
                        self.metrics['train_autoencode_loss'].value())
                    self.history['train_discriminate_loss'].append(
                        self.metrics['train_discriminate_loss'].value())
                else:
                    self.history['qc_discriminate_loss'].append(
                        self.metrics['qc_discriminate_loss'].value())
                    self.history['qc_distance'].append(
                        self.metrics['qc_distance'].value())
                    epoch_metric = self.history['qc_distance'][-1]
                    print('Epoch: %d, qc_distance: %.4f' % (e, epoch_metric))

            if not self.no_best:
                self._record_best_models(epoch_metric)

        if not self.no_best:
            print("Best metric: %.4f" % self.best_metric)
            for k, v in self.models.items():
                v.load_state_dict(self.best_models_wts[k])

        return self.models, self.history

    def _forward_autoencode(self, batch_x, batch_y, batch_text):
        with torch.set_grad_enabled(True):
            hidden = self.models['encoder'](batch_x)
            batch_x_recon = self.models['decoder'](hidden)
        with torch.set_grad_enabled(False):
            no_batch_num = self.models['discriminator'].in_f
            logit = self.models['discriminator'](hidden[:, :no_batch_num])
        with torch.set_grad_enabled(True):
            reconstruction_loss = self.criterions['reconstruction'](
                batch_x, batch_x_recon)
            adversarial_loss = self.criterions['adversarial'](batch_y, logit)
            # 排序做的不好，说明这写维度中没有批次的信息，批次的信息都在后面的维度中
            all_loss = reconstruction_loss - adversarial_loss
        all_loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(
                chain(
                    self.models['encoder'].parameters(),
                    self.models['decoder'].parameters()
                ), max_norm=1
            )
        self.optimizers['autoencode'].step()
        self.metrics['train_autoencode_loss'].add(all_loss, batch_x.size(0))
        batch_text.update_mapping(loss=all_loss.item(), batch='autoencode')

    def _forward_discriminate(self, batch_x, batch_y, batch_text):
        with torch.set_grad_enabled(False):
            hidden = self.models['encoder'](batch_x)
        with torch.set_grad_enabled(True):
            no_batch_num = self.models['discriminator'].in_f
            logit = self.models['discriminator'](hidden[:, :no_batch_num])
            adversarial_loss = self.criterions['adversarial'](batch_y, logit)
        adversarial_loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(
                self.models['discriminator'].parameters(), max_norm=1)
        self.optimizers['discriminate'].step()
        self.metrics['train_discriminate_loss'].add(
            adversarial_loss, batch_x.size(0))
        batch_text.update_mapping(
            loss=adversarial_loss.item(), batch='discriminate')

    def _forward_qc(self, batch_x, batch_y, batch_text):
        with torch.no_grad():
            no_batch_num = self.models['discriminator'].in_f
            hidden = self.models['encoder'](batch_x)
            hidden[:, no_batch_num:] = 0.
            batch_x_recon = self.models['decoder'](hidden)
            logit = self.models['discriminator'](hidden[:, :no_batch_num])
            adversarial_loss = self.criterions['adversarial'](batch_y, logit)
        self.metrics['qc_discriminate_loss'].add(
            adversarial_loss, batch_x.size(0))
        self.metrics['qc_distance'].add(batch_x_recon)
        batch_text.update_mapping(
            loss=adversarial_loss.item(), batch='autoencode')

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
    # pre_transfer = T.MultiCompose(*[
    #     T.Normalization()
    # ])
    meta_data = MetaBatchEffect.from_csv(
        config.sample_file, config.meta_file, ['injection.order', 'batch'],
        pre_transfer=T.Normalization()
    )
    subject_dat, qc_dat = meta_data.split_qc()
    dataloaders = {
        'train': data.DataLoader(
            subject_dat, batch_size=config.args.batch_size,
            num_workers=config.args.num_workers, shuffle=True
        ),
        'qc': data.DataLoader(
            qc_dat, batch_size=config.args.batch_size,
            num_workers=config.args.num_workers, shuffle=False
        ),
    }

    # ----- 构建网络和优化器 -----
    in_f = meta_data.num_features
    encoder = Coder(in_f, config.args.bottle_num)
    decoder = Coder(config.args.bottle_num, in_f)
    discriminator = Coder(
        config.args.no_batch_num, 1, block_num=2,
        spectral_norm=True
    )
    models = {
        'encoder': encoder, 'decoder': decoder, 'discriminator': discriminator}

    adversarial_criterion = RankLoss()
    reconstruction_criterion = nn.MSELoss()
    criterions = {
        'adversarial': adversarial_criterion,
        'reconstruction': reconstruction_criterion
    }

    ae_lr, disc_lr = config.args.ae_disc_lr
    reconstruction_optim = optim.Adam(
        chain(encoder.parameters(), decoder.parameters()),
        lr=ae_lr
    )
    adversarial_optim = optim.Adam(
        discriminator.parameters(), lr=disc_lr)
    optimizers = {
        'autoencode': reconstruction_optim, 'discriminate': adversarial_optim}

    # ----- 训练网络 -----
    trainer = BatchEffectTrainer(
        models, criterions, optimizers, epoch=config.args.epoch,
        ae_disc_train_num=config.args.ae_disc_train_num,
        no_best=config.args.no_best
    )
    best_models, hist = trainer.fit(dataloaders)
    print('')

    # 保存结果
    dirname = check_update_dirname(config.args.save)
    torch.save(best_models, os.path.join(dirname, 'models.pth'))
    pd.DataFrame(hist).to_csv(os.path.join(dirname, 'train.csv'))
    config.save(os.path.join(dirname, 'config.json'))


if __name__ == "__main__":
    main()
