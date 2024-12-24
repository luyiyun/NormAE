from itertools import chain
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import scipy.spatial.distance as dist

from .model import NormAENet


class LossAccumulator:
    def __init__(self):
        self.init()

    def init(self):
        self.loss_dict = {}
        self.cnt_dict = {}

    def update(
        self,
        batch_size: int,
        main_loss: torch.Tensor,
        loss_dict: dict[str, torch.Tensor],
        prefix: str = "",
    ):
        self.loss_dict[f"{prefix}main"] = (
            self.loss_dict.get(f"{prefix}main", 0)
            + main_loss.item() * batch_size
        )
        self.cnt_dict[f"{prefix}main"] = (
            self.cnt_dict.get(f"{prefix}main", 0) + batch_size
        )
        for k, v in loss_dict.items():
            self.loss_dict[f"{prefix}{k}"] = (
                self.loss_dict.get(f"{prefix}{k}", 0) + v.item() * batch_size
            )
            self.cnt_dict[f"{prefix}{k}"] = (
                self.cnt_dict.get(f"{prefix}{k}", 0) + batch_size
            )

    def calulate(self) -> dict[str, float]:
        return {k: v / self.cnt_dict[k] for k, v in self.loss_dict.items()}


class QCEvaluator:
    def __init__(self, pca_n_components: int):
        self.pca_n_components = pca_n_components
        self.init()

    def init(self):
        self.rec_clean_all = []
        self.y_all = []

    def update(self, rec_clean: torch.Tensor, y: torch.Tensor):
        self.rec_clean_all.append(rec_clean)
        self.y_all.append(y)

    def calculate(self) -> float:
        rec_clean_all = torch.cat(self.rec_clean_all, dim=0)
        # y_all = torch.cat(self.y_all, dim=0)
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=self.pca_n_components)),
            ]
        )
        pca = estimator.fit_transform(rec_clean_all.cpu().numpy())
        distance = dist.pdist(pca)
        md = np.mean(distance)

        return md


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.init()

    def init(self):
        self.best_score = float("inf")
        self.cnt = 0
        self.best_epoch = -1

    def __call__(self, epoch: int, score: float) -> bool:
        if score < self.best_score:
            self.best_score = score
            self.cnt = 0
            self.best_epoch = epoch
        else:
            self.cnt += 1
            if self.cnt >= self.patience:
                return True
        return False

    def best_info(self) -> str:
        return (
            f"best_score: {self.best_score:.2f}, best_epoch: {self.best_epoch}"
        )


class BestModelSaver:
    def __init__(self):
        self.best_score = float("inf")
        self.best_model = None

    def see(self, model: NormAENet, score: float):
        if score < self.best_score:
            self.best_score = score
            self.best_model = deepcopy(model.state_dict())

    def load_best_model(self, model: NormAENet):
        model.load_state_dict(self.best_model)


class ChainDataLoader:
    def __init__(self, *loaders):
        self.loaders = loaders

    def __iter__(self):
        return chain(*self.loaders)

    def __len__(self):
        return sum(len(loader) for loader in self.loaders)


def get_x_y_z(
    b: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    x = b["X"].to(device)
    y = b.get("y", None)
    z = b.get("z", None)
    if y is not None:
        y = y.to(device)
    if z is not None:
        z = z.to(device)

    return x, y, z


def train(
    model: NormAENet,
    train_loader: DataLoader,
    device: torch.device,
    lr_rec: float,
    lr_disc_batch: float,
    lr_disc_order: float,
    n_epochs_rec_pretrain: int,
    n_epochs_disc_pretrain: int,
    n_epochs_iter_train: int,
    qc_loader: DataLoader | None = None,
    early_stop: bool = False,
    early_stop_patience: int = 10,
    grad_clip: bool = False,
    grad_clip_norm: float = 1.0,
) -> pd.DataFrame:
    if early_stop:
        assert qc_loader is not None, (
            "QC loader is required for early stop, please set early_stop=False"
            " if you don't want to use early stop"
        )

    model.to(device)
    optimizer_rec = torch.optim.Adam(
        model.get_parameters("autoencoder"),
        lr=lr_rec,
        betas=(0.5, 0.9),
    )
    params = []
    if model.disc_b_hiddens is not None:
        params.append(
            {"params": model.disc_b.parameters(), "lr": lr_disc_batch}
        )
    if model.disc_o_hiddens is not None:
        params.append(
            {"params": model.disc_o.parameters(), "lr": lr_disc_order}
        )
    optimizer_disc = torch.optim.Adam(
        params,
        betas=(0.5, 0.9),
    )
    loss_accumulator = LossAccumulator()
    best_saver = BestModelSaver()
    if qc_loader is not None:
        pca_estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=3)),
            ]
        )
        # qc_evaluator = QCEvaluator(pca_n_components=3)
    if early_stop:
        early_stopper = EarlyStopper(patience=early_stop_patience)

    loader = (
        ChainDataLoader(train_loader, qc_loader)
        if qc_loader is not None
        else train_loader
    )

    history = []
    for e in tqdm(
        range(n_epochs_rec_pretrain), desc="Epoch(pretrain autoencoder): "
    ):
        model.train()
        loss_accumulator.init()
        for b in tqdm(loader, desc="Batch: ", leave=False):
            x, y, z = get_x_y_z(b, device)
            loss, losses = model(x, y, z, phase="reconstruct_pretrain")
            optimizer_rec.zero_grad()
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(
                    model.get_parameters("autoencoder"),
                    max_norm=grad_clip_norm,
                )
            optimizer_rec.step()

            loss_accumulator.update(x.size(0), loss, losses)

        loss_dict = loss_accumulator.calulate()
        tqdm.write(
            f"rec_pretrain, epoch {e}, "
            + ", ".join(f"{k}: {v:.2f}" for k, v in loss_dict.items())
        )
        loss_dict.update(
            {"epoch": e, "phase_epoch": e, "phase": "reconstruct_pretrain"}
        )
        history.append(loss_dict)

    for e in tqdm(
        range(n_epochs_disc_pretrain), desc="Epoch(pretrain discriminator): "
    ):
        model.train()
        loss_accumulator.init()
        for b in tqdm(loader, desc="Batch: ", leave=False):
            x, y, z = get_x_y_z(b, device)
            loss, losses = model(x, y, z, phase="discriminate")
            optimizer_disc.zero_grad()
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(
                    model.get_parameters("discriminator"),
                    max_norm=grad_clip_norm,
                )
            optimizer_disc.step()

            loss_accumulator.update(x.size(0), loss, losses)

        loss_dict = loss_accumulator.calulate()
        tqdm.write(
            f"disc_pretrain, epoch {e}, "
            + ", ".join(f"{k}: {v:.2f}" for k, v in loss_dict.items())
        )
        loss_dict.update(
            {
                "epoch": e + n_epochs_rec_pretrain,
                "phase_epoch": e,
                "phase": "discriminate",
            }
        )
        history.append(loss_dict)

    for e in tqdm(
        range(n_epochs_iter_train), desc="Epoch(iterative training): "
    ):
        model.train()
        loss_accumulator.init()
        for b in tqdm(loader, desc="Batch: ", leave=False):
            x, y, z = get_x_y_z(b, device)
            loss, losses = model(x, y, z, phase="discriminate")
            optimizer_disc.zero_grad()
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(
                    model.get_parameters("discriminator"),
                    max_norm=grad_clip_norm,
                )
            optimizer_disc.step()
            loss_accumulator.update(
                x.size(0), loss, losses, prefix="iterdisc_"
            )

            loss, losses = model(x, y, z, phase="reconstruct")
            optimizer_rec.zero_grad()
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(
                    model.get_parameters("autoencoder"),
                    max_norm=grad_clip_norm,
                )
            optimizer_rec.step()
            loss_accumulator.update(x.size(0), loss, losses, prefix="iterrec_")

        loss_dict = loss_accumulator.calulate()
        tqdm.write(
            f"iter_train, epoch {e}, "
            + ", ".join(f"{k}: {v:.2f}" for k, v in loss_dict.items())
        )
        loss_dict.update(
            {
                "epoch": e + n_epochs_rec_pretrain + n_epochs_disc_pretrain,
                "phase_epoch": e,
                "phase": "iter_train",
            }
        )
        history.append(loss_dict)

        # --- validation ---
        if qc_loader is not None:
            model.eval()
            loss_accumulator.init()
            # qc_evaluator.init()
            rec_clean_qc, rec_clean_train = [], []
            with torch.no_grad():
                for b in tqdm(qc_loader, desc="QC Batch: ", leave=False):
                    x, y, z = get_x_y_z(b, device)
                    qc_loss, rec_clean = model(
                        x, y, z, phase="reconstruct_valid"
                    )
                    loss_accumulator.update(
                        x.size(0), qc_loss, {}, prefix="qc_"
                    )
                    rec_clean_qc.append(rec_clean)
                rec_clean_qc = torch.cat(rec_clean_qc, dim=0)
                for b in tqdm(
                    train_loader, desc="QC Batch(Subject): ", leave=False
                ):
                    x, y, z = get_x_y_z(b, device)
                    rec_clean = model(x, y, z, phase="generate")
                    rec_clean_train.append(rec_clean)
                rec_clean_train = torch.cat(rec_clean_train, dim=0)

            # comute qc_mean_distance
            rec_clean_all = torch.cat([rec_clean_qc, rec_clean_train], dim=0)
            embed = pca_estimator.fit_transform(rec_clean_all.cpu().numpy())
            qc_md = dist.pdist(embed[: len(rec_clean_qc)]).mean()

            loss_dict = loss_accumulator.calulate()
            score = qc_md + loss_dict["qc_main"] * 100
            tqdm.write(
                f"qc_valid, epoch {e}, score: {score:.2f}, "
                + f"qc_mean_distance: {qc_md:.2f}, "
                + ", ".join(f"{k}: {v:.2f}" for k, v in loss_dict.items())
            )
            loss_dict.update(
                {
                    "epoch": e
                    + n_epochs_rec_pretrain
                    + n_epochs_disc_pretrain,
                    "phase_epoch": e,
                    "phase": "iter_train_valid",
                    "qc_md": qc_md,
                }
            )
            history.append(loss_dict)
            best_saver.see(model, score)
            if early_stop and early_stopper(e, score):
                break

    if early_stop:
        tqdm.write(early_stopper.best_info())
    best_saver.load_best_model(model)

    return pd.DataFrame.from_records(history)


def generate(
    model: NormAENet,
    loader: DataLoader,
    device: str,
) -> np.ndarray:
    model.to(device)
    model.eval()
    res = []
    with torch.no_grad():
        for b in tqdm(loader, desc="Generating: ", leave=True):
            x, y, z = get_x_y_z(b, device)
            x_clean = model(x, y, z, phase="generate")
            res.append(x_clean)
    res = torch.cat(res, dim=0).cpu().numpy()
    return res
