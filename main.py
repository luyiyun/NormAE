#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
from time import perf_counter

import pandas as pd
import torch

from config import Config
from transfer import Normalization
from datasets import get_metabolic_data, ConcatData
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
                                             pre_transfer=pre_transfer,
                                             use_log=opts.use_log,
                                             use_batch=opts.use_batch,
                                             use_samples_size=opts.sample_size,
                                             random_seed=opts.random_seed)
    datas = {'subject': subject_dat, 'qc': qc_dat}

    # build estimator
    if opts.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if opts.device == "GPU" else "cpu")
    trainer = BatchEffectTrainer(
        subject_dat.num_features, subject_dat.num_batch_labels,
        device, pre_transfer, opts)
    # load models
    if opts.load is not None:
        model_file = os.path.join(opts.load, "models.pth") \
            if os.path.isdir(opts.load) else opts.load
        trainer.load_model(model_file)
    if opts.task == "train":
        # ----- training -----
        fit_time1 = perf_counter()
        best_models, hist, early_stop_objs = trainer.fit(datas)
        fit_time2 = perf_counter()
        early_stop_objs["fit_duration"] = fit_time2 - fit_time1
        print('')
        # ----- save models and results -----
        if os.path.exists(opts.save):
            dirname = input("%s has been already exists, please input New: " %
                            config.args.save)
            os.makedirs(dirname)
        else:
            os.makedirs(opts.save)
        torch.save(best_models, os.path.join(opts.save, 'models.pth'))
        pd.DataFrame(hist).to_csv(os.path.join(opts.save, 'train.csv'))
        config.save(os.path.join(opts.save, 'config.json'))
        with open(os.path.join(opts.save, 'early_stop_info.json'), 'w') as f:
            json.dump(early_stop_objs, f)
    elif opts.task == "remove":
        # ----- remove batch effects -----
        all_dat = ConcatData(subject_dat, qc_dat)
        all_res = trainer.generate(all_dat, verbose=True,
                                   compute_qc_loss=False)
        # ----- save results -----
        for k, v in all_res.items():
            if k not in ['Ys', 'Codes']:
                v, _ = pre_transfer.inverse_transform(v, None)
                v = v.T
                v.index.name = 'meta.name'
            v.to_csv(os.path.join(opts.save, '%s.csv' % k))
        print('')
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
