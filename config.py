import os
import json
import time

import argparse
import torch.nn as nn
import torch.optim as optim


class Config:

    metabolic_y_files = {
        i: os.path.join('./DATA', i, 'sample.information.csv')
        for i in ['Amide', 'T3']
    }
    metabolic_x_files = {
        i: os.path.join('./DATA', i, 'meta.csv')
        for i in ['Amide', 'T3']
    }
    demo_sub_file = './DATA/Demo/sample.csv'
    demo_qc_file = './DATA/Demo/qc.csv'

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # 经常进行更改的
        self.parser.add_argument(
            '-s', '--save', default='./save',
            help='保存的文件夹路径，如果有重名，会在其后加-来区别'
        )
        self.parser.add_argument(
            '-t', '--task', default='T3',
            help="使用哪个数据，默认是T3，还可以是demo, Amide"
        )
        self.parser.add_argument(
            '-td', '--train_data', default='subject',
            help=("使用哪些数据作为训练数据，默认是subject，也可以是all，"
                  "即使用所有数据来train")
        )
        self.parser.add_argument(
            '-e', '--epoch', default=5000, type=int,
            help='epoch 数量，默认是5000'
        )
        self.parser.add_argument(
            '--no_batch_num', type=int, default=90,
            help="瓶颈层中不包含batch effect信息的节点数量，默认是90"
        )
        self.parser.add_argument(
            '--ae_disc_weight', type=float, default=(1, 1), nargs=2,
            help="重建误差权重和对抗权重，默认是1.0和1.0"
        )
        # self.parser.add_argument(
        #     '--supervise', default='both',
        #     help=(
        #         "用于discriminator的标签，如果是order则"
        #         "只有排序，如果是cls则只有分类，如果是both则两者都有，默认是"
        #         "both"
        #     )
        # )
        self.parser.add_argument(
            '--cls_order_weight', default=(1.0, 1.0), nargs=2, type=int,
            help="cls loss和order loss在判别是所占的比例，默认是1:1"
        )
        self.parser.add_argument(
            '--label_smooth', default=0.2, type=float,
            help='label smoothing, default 0.2'
        )
        self.parser.add_argument(
            '--schedual_stones', type=int, nargs='+',
            default=[3000], help="epochs of lrs multiply 0.1, default [2000]"
        )
        self.parser.add_argument(
            '--cls_leastsquare', action='store_true',
            help="if use, mse rather than ce in cls loss."
        )
        self.parser.add_argument(
            '--order_leastsquare', action='store_true',
            help="if use, mse rather than ce in order loss."
        )
        self.parser.add_argument(
            '--use_batch_for_order', action='store_true',
            help="if use, compute rank loss with batch"
        )

        # 已经基本确定下来不动的
        self.parser.add_argument(
            '--data_normalization', default='standard',
            help="数据标准化类型，默认是standard，还可以是minmax、maxabs或robust"
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
            '--bottle_num', type=int, default=100,
            help="瓶颈层的节点数量，默认是100"
        )
        self.parser.add_argument(
            '--ae_disc_train_num', type=int, default=(1, 1), nargs=2,
            help="autoencode部分和discriminate部分训练次数的比例，默认是1:1"
        )
        self.parser.add_argument(
            '--ae_disc_lr', type=float, default=(0.0001, 0.01), nargs=2,
            help=(
                "autoencode部分和discriminate部"
                "分训练时使用的lr，默认是0.0001和0.01"
            )
        )
        self.parser.add_argument(
            '--l2', default=0.0, type=float,
            help="weight decay, default 0.0"
        )
        self.parser.add_argument(
            '--spectral_norm', action='store_true',
            help='if use, linear layer will be spectral normalized.'
        )

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

