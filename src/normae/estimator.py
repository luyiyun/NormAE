from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
# import seaborn as sns

from .model import NormAENet
from .train import train, generate


class NormAEDataSet(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ):
        super().__init__()
        self.X = X
        self.y = y
        self.z = z

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        res = {"X": self.X[idx]}
        if self.y is not None:
            res["y"] = self.y[idx]
        if self.z is not None:
            res["z"] = self.z[idx]
        return res


@dataclass
class NormAE:
    n_latents: int = 500
    enc_hiddens: tuple[int] = (1000, 1000)
    dec_hiddens: tuple[int] = (1000, 1000)
    disc_batch_hiddens: list[int] = (250, 250)
    disc_order_hiddens: list[int] = (250, 250)
    enc_bn: bool = True
    dec_bn: bool = True
    disc_batch_bn: bool = True
    disc_order_bn: bool = False
    enc_dropout: float = 0.3
    dec_dropout: float = 0.1
    disc_batch_dropout: float = 0.3
    disc_order_dropout: float = 0.3
    grouped_order_loss: bool = True
    lambda_batch: float = 1.0
    lambda_order: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    lr_rec: float = 2e-4
    lr_disc_batch: float = 5e-3
    lr_disc_order: float = 5e-4
    n_epochs_rec_pretrain: int = 1000
    n_epochs_disc_pretrain: int = 10
    n_epochs_iter_train: int = 700
    early_stop: bool = False
    early_stop_patience: int = 10
    grad_clip: bool = True
    grad_clip_norm: float = 1.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        z: np.ndarray | None = None,
        X_qc: np.ndarray | None = None,
        y_qc: np.ndarray | None = None,
        z_qc: np.ndarray | None = None,
    ):
        if y is None and z is None:
            raise ValueError("Either y or z must be provided.")
        if X_qc is not None:
            if y is not None:
                assert (
                    y_qc is not None
                ), "y_qc must be provided if y is provided."
            if z is not None:
                assert (
                    z_qc is not None
                ), "z_qc must be provided if z is provided."

        device = torch.device(self.device)

        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        yt = (
            None
            if y is None
            else torch.tensor(y, dtype=torch.long, device=device)
        )
        zt = (
            None
            if z is None
            else torch.tensor(z, dtype=torch.float32, device=device)
        )
        dat = NormAEDataSet(Xt, yt, zt)
        dataloader = torch.utils.data.DataLoader(
            dat, batch_size=self.batch_size, shuffle=True
        )

        if X_qc is not None:
            Xt_qc = torch.tensor(X_qc, dtype=torch.float32, device=device)
            yt_qc = (
                None
                if y_qc is None
                else torch.tensor(y_qc, dtype=torch.long, device=device)
            )
            zt_qc = (
                None
                if z_qc is None
                else torch.tensor(z_qc, dtype=torch.float32, device=device)
            )
            qc_dat = NormAEDataSet(Xt_qc, yt_qc, zt_qc)
            qc_dataloader = torch.utils.data.DataLoader(
                qc_dat,
                batch_size=self.batch_size,
                shuffle=True,
            )

        self.model_ = NormAENet(
            n_features=X.shape[1],
            n_latents=self.n_latents,
            n_batches=np.unique(y).shape[0] if y is not None else None,
            enc_hiddens=self.enc_hiddens,
            dec_hiddens=self.dec_hiddens,
            disc_batch_hiddens=self.disc_batch_hiddens
            if y is not None
            else None,
            disc_order_hiddens=self.disc_order_hiddens
            if z is not None
            else None,
            act=nn.ReLU(),
            enc_bn=self.enc_bn,
            dec_bn=self.dec_bn,
            disc_batch_bn=self.disc_batch_bn,
            disc_order_bn=self.disc_order_bn,
            enc_dropout=self.enc_dropout,
            dec_dropout=self.dec_dropout,
            disc_batch_dropout=self.disc_batch_dropout,
            disc_order_dropout=self.disc_order_dropout,
            grouped_order_loss=self.grouped_order_loss,
            lambda_batch=self.lambda_batch,
            lambda_order=self.lambda_order,
        ).to(device)

        self.history_ = train(
            model=self.model_,
            train_loader=dataloader,
            device=device,
            n_epochs_rec_pretrain=self.n_epochs_rec_pretrain,
            n_epochs_disc_pretrain=self.n_epochs_disc_pretrain,
            n_epochs_iter_train=self.n_epochs_iter_train,
            lr_rec=self.lr_rec,
            lr_disc_batch=self.lr_disc_batch,
            lr_disc_order=self.lr_disc_order,
            early_stop=self.early_stop,
            early_stop_patience=self.early_stop_patience,
            qc_loader=qc_dataloader if X_qc is not None else None,
            grad_clip=self.grad_clip,
            grad_clip_norm=self.grad_clip_norm,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        device = torch.device(self.device)
        X = torch.tensor(X, dtype=torch.float32, device=device)
        dat = NormAEDataSet(X)
        dataloader = torch.utils.data.DataLoader(
            dat, batch_size=self.batch_size, shuffle=False
        )
        X_clean = generate(self.model_, dataloader, device=self.device)
        return X_clean

    def plot_history(
        self, fig_name: str = "history.png"
    ) -> tuple[plt.Figure, plt.Axes]:
        phases = self.history_["phase"].unique()
        if phases.shape[0] > 2:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        else:
            fig, axs = plt.subplots(
                nrows=1, ncols=phases.shape[0], figsize=(10, 5), squeeze=False
            )
        axs = axs.flatten()

        for i, phase in enumerate(phases):
            ax = axs[i]
            dfi = self.history_[self.history_["phase"] == phase]
            dfi = dfi.dropna(axis=1, how="all")
            x = dfi["phase_epoch"].values
            for c in dfi.columns:
                if c in ["epoch", "phase_epoch", "phase"]:
                    continue
                y = dfi[c].values
                ax.plot(x, y, label=c)
            ax.set_title(phase)
            ax.set_xlabel("epoch")
            ax.legend()

        fig.tight_layout()
        if fig_name is not None:
            fig.savefig(fig_name)
        else:
            plt.show()

        return fig, axs
