from typing import Literal
from itertools import chain

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        n_inpt: int,
        n_oupt: int,
        hiddens: tuple[int],
        act: nn.Module = nn.ReLU(),
        bn: bool = False,
        dropout: float = 0.0,
        final_act: nn.Module | None = None,
    ):
        super().__init__()
        layers = []
        for i, o in zip([n_inpt] + list(hiddens[:-1]), hiddens):
            layers.append(nn.Linear(i, o))
            if bn:
                layers.append(nn.BatchNorm1d(o))
            layers.append(act)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hiddens[-1], n_oupt))
        if final_act is not None:
            layers.append(final_act)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OrderLoss(nn.Module):
    """OrderLoss"""

    def __init__(self):
        super(OrderLoss, self).__init__()
        self.cross_entropy = nn.BCEWithLogitsLoss()

    def forward(self, pred, target, group=None):
        """forward

        :param pred: predicted rank score;
        :param target: injection orders;
        :param group: batch labels;
        """
        pred = pred.squeeze()
        target = target.squeeze().float()
        low, high = self._comparable_pairs(target, group)
        if low.size(0) == 0:
            return torch.tensor(0.0, device=pred.device)
        low_pred, high_pred = pred[low], pred[high]
        res = self.cross_entropy(
            high_pred - low_pred,
            torch.ones(high_pred.size(0), device=pred.device),
        )
        return res

    @staticmethod
    def _comparable_pairs(true_rank, group=None):
        """_comparable_pairs
        Get all cmparable sample pairs

        :param true_rank: true injection orders;
        :param group: batch labels;
        """
        batch_size = len(true_rank)
        indx = torch.arange(batch_size, device=true_rank.device)
        pairs1 = indx.repeat(batch_size)
        pairs2 = indx.repeat_interleave(batch_size, dim=0)
        # get pairs that first > second
        time_mask = true_rank[pairs1] < true_rank[pairs2]
        pairs1, pairs2 = pairs1[time_mask], pairs2[time_mask]
        # if group is not None, just using pairs that both in one group
        if group is not None:
            batch_mask = group[pairs1] == group[pairs2]
            pairs1, pairs2 = pairs1[batch_mask], pairs2[batch_mask]
        return pairs1, pairs2


class NormAENet(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_latents: int,
        n_batches: int | None = None,
        enc_hiddens: tuple[int] = (128,),
        dec_hiddens: tuple[int] = (128,),
        disc_batch_hiddens: tuple[int] | None = None,
        disc_order_hiddens: tuple[int] | None = None,
        act: nn.Module = nn.ReLU(),
        enc_bn: bool = True,
        dec_bn: bool = True,
        disc_batch_bn: bool = True,
        disc_order_bn: bool = False,
        enc_dropout: float = 0.0,
        dec_dropout: float = 0.0,
        disc_batch_dropout: float = 0.0,
        disc_order_dropout: float = 0.0,
        grouped_order_loss: bool = True,
        lambda_batch: float = 1.0,
        lambda_order: float = 1.0,
    ):
        if n_batches is None and disc_batch_hiddens is not None:
            raise ValueError(
                "n_batches should be provided " "if disc_batch_hiddens is not None"
            )
        if n_batches is not None and disc_batch_hiddens is None:
            raise ValueError(
                "disc_batch_hiddens should be provided " "if n_batches is not None"
            )
        if n_batches is None and disc_order_hiddens is None:
            raise ValueError(
                "at least one of disc_batch_hiddens and disc_order_hiddens "
                "should be provided"
            )

        super().__init__()
        self.n_batches = n_batches
        self.disc_b_hiddens = disc_batch_hiddens
        self.disc_o_hiddens = disc_order_hiddens
        self.grouped_order_loss = grouped_order_loss
        self.lambda_batch = lambda_batch
        self.lambda_order = lambda_order

        self.encoder = MLP(
            n_features,
            n_latents,
            enc_hiddens,
            act,
            enc_bn,
            enc_dropout,
        )
        self.decoder = MLP(
            n_latents,
            n_features,
            dec_hiddens,
            act,
            dec_bn,
            dec_dropout,
        )
        self.mapper = MLP(
            int(disc_order_hiddens is not None)
            if n_batches is None
            else n_batches + int(disc_order_hiddens is not None),
            n_latents,
            (500,),
            act,
            True,
            dropout=0.0,
        )

        if disc_batch_hiddens is not None:
            self.disc_b = MLP(
                n_latents,
                n_batches,
                disc_batch_hiddens,
                act,
                disc_batch_bn,
                disc_batch_dropout,
            )
        if disc_order_hiddens is not None:
            self.disc_o = MLP(
                n_latents,
                1,
                disc_order_hiddens,
                act,
                disc_order_bn,
                disc_order_dropout,
            )

        self.criterion_rec = nn.L1Loss()
        self.criterion_disc_b = nn.CrossEntropyLoss()
        self.criterion_disc_o = OrderLoss()

    def get_parameters(self, module: Literal["autoencoder", "discriminator"]):
        if module == "autoencoder":
            return chain(
                self.encoder.parameters(),
                self.decoder.parameters(),
                self.mapper.parameters(),
            )
        elif module == "discriminator":
            params = []
            if self.disc_b_hiddens is not None:
                params.append(self.disc_b.parameters())
            if self.disc_o_hiddens is not None:
                params.append(self.disc_o.parameters())
            return chain(*params)
        else:
            raise ValueError(f"Invalid module: {module}")

    def _forward_reconstruct(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        disc: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        latent = self.encoder(x)
        batch = []
        if self.disc_b_hiddens is not None:
            batch.append(torch.eye(self.n_batches, device=y.device)[y])
        if self.disc_o_hiddens is not None:
            batch.append(z[:, None])
        batch = torch.cat(batch, dim=1)

        latent_w_batch = latent + self.mapper(batch)
        x_rec = self.decoder(latent_w_batch)
        loss_rec = self.criterion_rec(x_rec, x)

        loss = loss_rec
        losses = {"rec": loss_rec}
        if disc:
            if self.disc_b_hiddens is not None:
                disc_b_logits = self.disc_b(latent)
                loss_disc_b = self.criterion_disc_b(disc_b_logits, y)
                loss = loss - self.lambda_batch * loss_disc_b
                losses["disc_b"] = loss_disc_b
            if self.disc_o_hiddens is not None:
                disc_o_logits = self.disc_o(latent)
                loss_disc_o = self.criterion_disc_o(
                    disc_o_logits,
                    z,
                    group=y if self.grouped_order_loss else None,
                )
                loss = loss - self.lambda_order * loss_disc_o
                losses["disc_o"] = loss_disc_o

        return loss, losses

    def _forward_discriminate(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():  # NOTE: 可能会产生冲突
            latent = self.encoder(x)
        loss, loss_dict = 0.0, {}
        if self.disc_b_hiddens is not None:
            disc_b_logits = self.disc_b(latent)
            loss_disc_b = self.criterion_disc_b(disc_b_logits, y)
            loss = loss + loss_disc_b
            loss_dict["disc_b"] = loss_disc_b
        if self.disc_o_hiddens is not None:
            disc_o_logits = self.disc_o(latent)
            loss_disc_o = self.criterion_disc_o(
                disc_o_logits, z, group=y if self.grouped_order_loss else None
            )
            loss = loss + loss_disc_o
            loss_dict["disc_o"] = loss_disc_o

        return loss, loss_dict

    def _forward_generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def _forward_reconstruct_valid(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        rec_clean = self.decoder(latent)

        batch = []
        if self.disc_b_hiddens is not None:
            batch.append(torch.eye(self.n_batches, device=y.device)[y])
        if self.disc_o_hiddens is not None:
            batch.append(z[:, None])
        batch = torch.cat(batch, dim=1)

        latent_w_batch = latent + self.mapper(batch)
        x_rec = self.decoder(latent_w_batch)
        loss_rec = self.criterion_rec(x_rec, x)

        return loss_rec, rec_clean

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        phase: Literal[
            "reconstruct",
            "discriminate",
            "generate",
            "reconstruct_pretrain",
            "reconstruct_valid",
        ] = "reconstruct",
    ) -> (
        tuple[torch.Tensor, dict[str, torch.Tensor]]
        | torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
    ):
        if phase == "reconstruct":
            return self._forward_reconstruct(x, y, z)
        elif phase == "reconstruct_pretrain":
            return self._forward_reconstruct(x, y, z, disc=False)
        elif phase == "discriminate":
            return self._forward_discriminate(x, y, z)
        elif phase == "generate":
            return self._forward_generate(x)
        elif phase == "reconstruct_valid":
            return self._forward_reconstruct_valid(x, y, z)
        else:
            raise ValueError(f"Invalid phase: {phase}")
