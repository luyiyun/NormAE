import copy

import visdom
import torch



class VisObj:
    def __init__(self,):
        self.epoch_idx = -1
        self.batch_losses = {}
        self.epoch_losses = {}
        self.vis = visdom.Visdom()

    def add_batch_loss(self, reset=True, **loss_dict):
        if reset:
            self.batch_idx = 0
            update = None
        else:
            self.batch_idx += 1
            update = 'append'
        ks, vs = self._dict2tensor(**loss_dict)
        self.vis.line(
            X=torch.tensor([self.batch_idx]),
            Y=vs, update=update, win='batch_losses',
            opts={'title': "Batch Loss", 'legend': ks}
        )

    def add_epoch_loss(self, **loss_dict):
        if self.epoch_idx == -1:
            update = None
        else:
            update = 'append'
        self.epoch_idx += 1
        ks, vs = self._dict2tensor(**loss_dict)
        self.vis.line(
            X=torch.tensor([self.epoch_idx]),
            Y=vs, update=update, win='epoch_losses',
            opts={'title': "Epoch Loss", 'legend': ks}
        )

    @staticmethod
    def _dict2tensor(**loss_dict):
        ks = []
        vs = []
        for k, v in loss_dict.items():
            ks.append(k)
            vs.append(v)
        vs = torch.tensor(vs).squeeze().unsqueeze(0)
        return ks, vs
