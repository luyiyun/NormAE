import os
import json
import time

import argparse
import torch.nn as nn
import torch.optim as optim


class NoneScheduler:
    def __init__(self, optimizer):
        pass

    def step(self):
        pass


class Config:

    sample_file = "./DATA/metabolic/sample.information.T3.csv"
    meta_file = "./DATA/metabolic/data_T3原始数据.csv"

    def __init__(self, pred=False):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            '-s', '--save', default='./save',
            help='保存的文件夹路径，如果有重名，会在其后加-来区别'
        )
        self.parser.add_argument(
            '-bs', '--batch_size', default=64, type=int,
            help='batch size，默认时64'
        )
        self.parser.add_argument(
            '-nw', '--num_workers', default=12, type=int,
            help='多进程数目，默认时12'
        )
        self.parser.add_argument(
            '-e', '--epoch', default=200, type=int,
            help='epoch 数量，默认是200'
        )
        self.parser.add_argument(
            '--bottle_num', type=int, default=100,
            help="瓶颈层的节点数量，默认是100"
        )
        self.parser.add_argument(
            '--no_batch_num', type=int, default=90,
            help="瓶颈层中不包含batch effect信息的节点数量，默认是90"
        )
        self.parser.add_argument(
            '--ae_disc_train_num', type=int, default=(1, 2), nargs=2,
            help="autoencode部分和discriminate部分训练次数的比例，默认是1:2"
        )
        self.parser.add_argument(
            '--ae_disc_lr', type=float, default=(0.01, 0.1), nargs=2,
            help=(
                "autoencode部分和discriminate部"
                "分训练时使用的lr，默认是0.01和0.1"
            )
        )
        self.parser.add_argument(
            '--no_best', action='store_true',
            help="使用此参数则表示不根据metric进行选择得到最好的模型"
        )
        if pred:
            self.parser.add_argument(
                '-d', '--dir', help="想要预测的模型，指的是训练完保存的文件夹")
        self.args = self.parser.parse_args()

    def save(self, fname):
        self.save_dict = self.args.__dict__
        with open(fname, 'w') as f:
            json.dump(self.save_dict, f)

    def show(self):
        print('')
        print('此次训练使用的参数是：')
        for k, v in self.args.__dict__.items():
            print('%s:  %s' % (k, str(v)))
        print('')

