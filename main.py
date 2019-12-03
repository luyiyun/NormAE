#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json

import pandas as pd
import torch

from config import Config
from transfer import Normalization
from datasets import get_metabolic_data
from train import BatchEffectTrainer


def main():
    # config
    config = Config()
    opts = config.init()
    config.show()

    # ----- read data -----
    pre_transfer = Normalization("standard")
    subject_dat, qc_dat = get_metabolic_data(opts.meta_data,
                                             opts.sample_data,
                                             pre_transfer=pre_transfer)
    datas = {'subject': subject_dat, 'qc': qc_dat}

    # ----- training -----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = BatchEffectTrainer(
        subject_dat.num_features, subject_dat.num_batch_labels,
        device, pre_transfer, opts)
    #  if config.args.load_model != '':
    #      trainer.load_model(config.args.load_model)
    best_models, hist, early_stop_objs = trainer.fit(datas)
    print('')

    # ----- save models and results -----
    if os.path.exists(opts.save):
        dirname = input("%s has been already exists, please input New: " %
                        config.args.save)
        os.makedirs(dirname)
    else:
        os.makedirs(config.args.save)
    torch.save(best_models, os.path.join(opts.save, 'models.pth'))
    pd.DataFrame(hist).to_csv(os.path.join(opts.save, 'train.csv'))
    config.save(os.path.join(opts.save, 'config.json'))
    with open(os.path.join(opts.save, 'early_stop_info.json'), 'w') as f:
        json.dump(early_stop_objs, f)


if __name__ == "__main__":
    main()
