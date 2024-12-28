from argparse import ArgumentParser
import os.path as osp
import os

import pandas as pd
import numpy as np

from .estimator import NormAE
from .utils import plot_pca


def read_and_preprocess(
    meta_csv: str,
    sample_csv: str,
    mz_row: str,
    rt_row: str,
    qc_indicator_col: str,
    qc_indicator_value: str,
    order_indicator_col: str,
    batch_indicator_col: str,
    filter_zero_thre: float,
    log_transform: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    info_df = pd.read_csv(sample_csv, index_col=0)
    for col in [qc_indicator_col, order_indicator_col, batch_indicator_col]:
        if col and col not in info_df.columns:
            raise ValueError(f"{col} not in sample_csv")

    meta_df = pd.read_csv(meta_csv, index_col=0)
    drop_rows = []
    if mz_row:
        assert mz_row in meta_df.columns, f"{mz_row} not in meta_df"
        drop_rows.append(mz_row)
    if rt_row:
        assert rt_row in meta_df.columns, f"{rt_row} not in meta_df"
        drop_rows.append(rt_row)
    if len(drop_rows) > 0:
        meta_df = meta_df.drop(columns=drop_rows).T

    print("Preprocessing data...")
    indice_inter = info_df.index.intersection(meta_df.index)
    meta_df = meta_df.loc[indice_inter]
    info_df = info_df.loc[indice_inter]

    # remove peaks that has most zero values in all samples
    mask1 = (meta_df == 0).mean(axis=0) < filter_zero_thre
    meta_df = meta_df.loc[:, mask1]
    # remove peaks that has most zero values in QCs
    if qc_indicator_col:
        qc_mask = info_df[qc_indicator_col] == qc_indicator_value
        qc_meta_df = meta_df.loc[qc_mask, :]
        mask2 = (qc_meta_df == 0).mean(axis=0) < 0.2
        meta_df = meta_df.loc[:, mask2]

    # for each peak, impute the zero values with the half of minimum values
    def impute_zero(peak):
        zero_mask = peak == 0
        if zero_mask.any():
            new_x = peak.copy()
            impute_value = peak.loc[~zero_mask].min()
            new_x[zero_mask] = impute_value / 2
            return new_x
        return peak

    meta_df = meta_df.apply(impute_zero, axis=0)

    # extract the useful information from y_file
    if batch_indicator_col:
        info_df[batch_indicator_col] = (
            info_df[batch_indicator_col].astype("category").cat.codes
        )
    if qc_indicator_col:
        info_df[qc_indicator_col] = info_df[qc_indicator_col] == qc_indicator_value

    if log_transform:
        meta_df = meta_df.map(np.log)

    print("Data preprocessing finished.")
    print(f"meta_df shape: {meta_df.shape}, info_df shape: {info_df.shape}")

    return meta_df, info_df


def boolean_parser(arg: str):
    if arg.lower() in ["true", "1", "t", "y", "yes"]:
        return True
    elif arg.lower() in ["false", "0", "f", "n", "no"]:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {arg}")


def app():
    parser = ArgumentParser(prog="NormAE CLI")
    parser.add_argument(
        "--meta_csv",
        required=True,
        type=str,
        help="csv file contained metabolomics values (must >= 0), each column is a sample and each row is a metabolite.",
    )
    parser.add_argument(
        "--sample_csv",
        required=True,
        type=str,
        help="csv file contained sample information, each row is a sample and each column is a feature.",
    )
    parser.add_argument(
        "--mz_row",
        type=str,
        default="mz",
        help="the name of the row in meta_df that contains the m/z values. Default is mz. Empty string means no mz row.",
    )
    parser.add_argument(
        "--rt_row",
        type=str,
        default="rt",
        help="the name of the row in meta_df that contains the retention time values. Default is rt. Empty string means no rt row.",
    )
    parser.add_argument(
        "--qc_indicator_col",
        type=str,
        default="class",
        help="the name of the column in sample_df that contains the QC indicator. Default is class. Empty string which means no QC indicator.",
    )
    parser.add_argument(
        "--qc_indicator_value",
        type=str,
        default="QC",
        help="the value of the QC indicator that indicates a QC sample. Default is QC.",
    )
    parser.add_argument(
        "--order_indicator_col",
        type=str,
        default="injection.order",
        help="the name of the column in sample_df that contains the sample injection order. Default is injection.order. Empty string which means no order indicator.",
    )
    parser.add_argument(
        "--batch_indicator_col",
        type=str,
        default="batch",
        help="the name of the column in sample_df that contains the sample batch indicator. Default is batch. Empty string which means no batch indicator.",
    )
    parser.add_argument(
        "--filter_zero_thre",
        type=float,
        default=0.2,
        help="the threshold of the percentage of zero values in a peak to filter it. Default is 0.2.",
    )
    parser.add_argument(
        "--log_transform",
        default=True,
        type=boolean_parser,
        help="whether to log transform the metabolomics values. Default is True",
    )
    parser.add_argument(
        "--n_latents",
        type=int,
        default=100,
        help="the number of latent variables. Default is 100.",
    )
    parser.add_argument(
        "--enc_hiddens",
        type=int,
        nargs="+",
        default=(300, 300),
        help="the number of hidden units in the encoder. Default is (300, 300).",
    )
    parser.add_argument(
        "--dec_hiddens",
        type=int,
        nargs="+",
        default=(300, 300),
        help="the number of hidden units in the decoder. Default is (300, 300).",
    )
    parser.add_argument(
        "--disc_batch_hiddens",
        type=int,
        nargs="+",
        default=(250,),
        help="the number of hidden units in the discriminator for batch effect. Default is (250,).",
    )
    parser.add_argument(
        "--disc_order_hiddens",
        type=int,
        nargs="+",
        default=(250,),
        help="the number of hidden units in the discriminator for order effect. Default is (250,).",
    )
    parser.add_argument(
        "--enc_bn",
        default=True,
        type=boolean_parser,
        help="whether to use batch normalization in the encoder. Default is True",
    )
    parser.add_argument(
        "--dec_bn",
        default=True,
        type=boolean_parser,
        help="whether to use batch normalization in the decoder. Default is True",
    )
    parser.add_argument(
        "--disc_batch_bn",
        default=True,
        type=boolean_parser,
        help="whether to use batch normalization in the discriminator for batch effect. Default is True",
    )
    parser.add_argument(
        "--disc_order_bn",
        default=False,
        type=boolean_parser,
        help="whether to use batch normalization in the discriminator for order effect. Default is False",
    )
    parser.add_argument(
        "--enc_dropout",
        type=float,
        default=0.3,
        help="the dropout rate of the encoder. Default is 0.3.",
    )
    parser.add_argument(
        "--dec_dropout",
        type=float,
        default=0.1,
        help="the dropout rate of the decoder. Default is 0.1.",
    )
    parser.add_argument(
        "--disc_batch_dropout",
        type=float,
        default=0.3,
        help="the dropout rate of the discriminator for batch effect. Default is 0.3.",
    )
    parser.add_argument(
        "--disc_order_dropout",
        type=float,
        default=0.3,
        help="the dropout rate of the discriminator for order effect. Default is 0.3.",
    )
    parser.add_argument(
        "--lambda_batch",
        type=float,
        default=1.0,
        help="the weight of the batch effect in the loss function. Default is 1.0.",
    )
    parser.add_argument(
        "--lambda_order",
        type=float,
        default=1.0,
        help="the weight of the order effect in the loss function. Default is 1.0.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="the device to run the model. Default is cpu, can also be cuda.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="the batch size. Default is 128.",
    )
    parser.add_argument(
        "--lr_rec",
        type=float,
        default=2e-4,
        help="the learning rate of the reconstruction loss. Default is 2e-4.",
    )
    parser.add_argument(
        "--lr_disc_batch",
        type=float,
        default=5e-3,
        help="the learning rate of the discriminator for batch effect. Default is 5e-3.",
    )
    parser.add_argument(
        "--lr_disc_order",
        type=float,
        default=5e-4,
        help="the learning rate of the discriminator for order effect. Default is 5e-4.",
    )
    parser.add_argument(
        "--n_epochs_rec_pretrain",
        type=int,
        default=100,
        help="the number of epochs for pretraining the reconstruction network. Default is 100.",
    )
    parser.add_argument(
        "--n_epochs_disc_pretrain",
        type=int,
        default=10,
        help="the number of epochs for pretraining the discriminator networks. Default is 10.",
    )
    parser.add_argument(
        "--n_epochs_iter_train",
        type=int,
        default=500,
        help="the number of epochs for training the model in adversarial iteration. Default is 500.",
    )
    parser.add_argument(
        "--min_n_epochs_iter_train",
        type=int,
        default=400,
        help="the minimum number of epochs for training the model in adversarial iteration. Default is 400.",
    )
    parser.add_argument(
        "--early_stop",
        type=boolean_parser,
        default=True,
        help="whether to use early stopping. Default is True",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=20,
        help="the patience of early stopping. Default is 20.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="the directory to save the trained results. Default is current directory.",
    )
    args = parser.parse_args()

    if not args.order_indicator_col and not args.batch_indicator_col:
        raise ValueError(
            "At least one of batch_indicator_col and order_indicator_col should be provided."
        )

    meta_df, info_df = read_and_preprocess(
        args.meta_csv,
        args.sample_csv,
        args.mz_row,
        args.rt_row,
        args.qc_indicator_col,
        args.qc_indicator_value,
        args.order_indicator_col,
        args.batch_indicator_col,
        args.filter_zero_thre,
        args.log_transform,
    )

    model = NormAE(
        n_latents=args.n_latents,
        enc_hiddens=args.enc_hiddens,
        dec_hiddens=args.dec_hiddens,
        disc_batch_hiddens=args.disc_batch_hiddens,
        disc_order_hiddens=args.disc_order_hiddens,
        enc_bn=args.enc_bn,
        dec_bn=args.dec_bn,
        disc_batch_bn=args.disc_batch_bn,
        disc_order_bn=args.disc_order_bn,
        enc_dropout=args.enc_dropout,
        dec_dropout=args.dec_dropout,
        disc_batch_dropout=args.disc_batch_dropout,
        disc_order_dropout=args.disc_order_dropout,
        lambda_batch=args.lambda_batch,
        lambda_order=args.lambda_order,
        device=args.device,
        batch_size=args.batch_size,
        lr_rec=args.lr_rec,
        lr_disc_batch=args.lr_disc_batch,
        lr_disc_order=args.lr_disc_order,
        n_epochs_rec_pretrain=args.n_epochs_rec_pretrain,
        n_epochs_disc_pretrain=args.n_epochs_disc_pretrain,
        n_epochs_iter_train=args.n_epochs_iter_train,
        min_n_epochs_iter_train=args.min_n_epochs_iter_train,
        early_stop=args.early_stop,
        early_stop_patience=args.early_stop_patience,
    )

    if args.qc_indicator_col:
        qc_indice = info_df[args.qc_indicator_col]
        dfs = {
            "subject": (
                meta_df[~qc_indice],
                info_df[~qc_indice],
            ),
            "qc": (meta_df[qc_indice], info_df[qc_indice]),
        }
    else:
        dfs = {
            "subject": (meta_df, info_df),
        }
    fit_kwargs = {}
    for k in dfs.keys():
        fit_kwargs["X" if k == "subject" else "X_qc"] = dfs[k][0].values
        if args.batch_indicator_col:
            fit_kwargs["y" if k == "subject" else "y_qc"] = dfs[k][1][
                args.batch_indicator_col
            ].values
        if args.order_indicator_col:
            fit_kwargs["z" if k == "subject" else "z_qc"] = dfs[k][1][
                args.order_indicator_col
            ].values
    print("Training model...")
    model.fit(**fit_kwargs)

    print("Remove the batch effects and save the clean data...")
    X_clean = model.transform(meta_df.values)
    X_clean_df = pd.DataFrame(X_clean, index=meta_df.index, columns=meta_df.columns)

    os.makedirs(args.output_dir, exist_ok=True)
    model.plot_history(osp.join(args.output_dir, "history.png"))
    fig, _ = plot_pca(
        X_clean,
        qc=info_df[args.qc_indicator_col].values if args.qc_indicator_col else None,
        batch=info_df[args.batch_indicator_col].values
        if args.batch_indicator_col
        else None,
        injection=info_df[args.order_indicator_col].values
        if args.order_indicator_col
        else None,
    )
    fig.savefig(osp.join(args.output_dir, "pca.png"))
    X_clean_df.T.to_csv(osp.join(args.output_dir, "X_clean.csv"))
