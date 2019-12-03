import json

import argparse


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # dataset
        self.parser.add_argument(
            "--meta_data", help="the path of metabolomics data"
        )
        self.parser.add_argument(
            "--sample_data", help="the path of sample information"
        )
        self.parser.add_argument(
            '-td', '--train_data', default='all',
            help=("the training data, subject or all (default)"
                  "即使用所有数据来train")
        )

        # save results
        self.parser.add_argument(
            '-s', '--save', default='./save',
            help='the path to save results, default ./save'
        )

        # architecture
        self.parser.add_argument(
            '--ae_encoder_units', default=[1000, 1000], type=int, nargs="+",
            help="the hidden units of encoder, default 1000, 1000"
        )
        self.parser.add_argument(
            '--ae_decoder_units', default=[1000, 1000], type=int, nargs="+",
            help="the hidden units of decoder, default 1000, 1000"
        )
        self.parser.add_argument(
            "--disc_b_units", default=[250, 250], type=int, nargs="+",
            help="the hidden units of disc_b, default 250, 250"
        )
        self.parser.add_argument(
            "--disc_o_units", default=[250, 250], type=int, nargs="+",
            help="the hidden units of disc_b, default 250, 250"
        )
        self.parser.add_argument(
            '--bottle_num', type=int, default=500,
            help="the number of bottle neck units, default 500"
        )
        self.parser.add_argument(
            '--dropouts', default=(0.3, 0.1, 0.3, 0.3), type=float, nargs=4,
            help=("the dropout rates of encoder, decoder, disc_b, disc_o,"
                  "default 0.3, 0.1, 0.3, 0.3")
        )

        # regularization
        self.parser.add_argument(
            '--lambda_b', type=float, default=1.0,
            help="the weight of adversarial loss for batch labels, default 1"
        )
        self.parser.add_argument(
            '--lambda_0', type=float, default=1.0,
            help=("the weight of adversarial loss for injection order,"
                  " default 1")
        )

        # training
        self.parser.add_argument(
            "--lr_rec", type=float, default=0.0002,
            help="the learning rate of AE training, default 0.0002"
        )
        self.parser.add_argument(
            "--lr_disc_b", type=float, default=0.005,
            help="the leanring rate of disc_b training, default 0.005"
        )
        self.parser.add_argument(
            "--lr_disc_o", type=float, default=0.0005,
            help="the leanring rate of disc_o training, default 0.0005"
        )
        self.parser.add_argument(
            '-e', '--epoch', default=(200, 100, 1000), type=int, nargs=3,
            help=("ae pretrain, disc pretrain, "
                  "iteration train epochs，default (1000, 10, 700)")
        )
        self.parser.add_argument(
            '--use_batch_for_order', default=True, type=bool,
            help="if compute rank loss with batch ?, default True"
        )
        self.parser.add_argument(
            '-bs', '--batch_size', default=64, type=int,
            help='batch size，default 64'
        )

        # other
        self.parser.add_argument(
            '--visdom_env', default='main',
            help="if use visdom, it is the env name,default main"
        )
        self.parser.add_argument(
            '--visdom_port', default=8097, type=int,
            help="if use visdom, it is the port, default 8097"
        )
        self.parser.add_argument(
            '-nw', '--num_workers', default=12, type=int,
            help='the number of multi cores, default 12'
        )

        self.args = self.parser.parse_args()

    def init(self):
        return self.args

    def save(self, fname):
        self.save_dict = self.args.__dict__
        with open(fname, 'w') as f:
            json.dump(self.save_dict, f)

    def show(self):
        print('')
        print('the settings of training：')
        for k, v in self.args.__dict__.items():
            print('%s:  %s' % (k, str(v)))
        print('')

