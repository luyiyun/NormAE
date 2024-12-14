import torch
import torch.nn as nn


class SimpleCoder(nn.Module):
    ''' MLP '''
    def __init__(
            self, units, act=nn.LeakyReLU(), bn=True, dropout=0.0,
            final_act=None
    ):
        '''
        args:
            units: list, the units for each layers, contain input layer and
                output layer;
            act: activity function for hidden layers;
            bn: bool, using BatchNorm if true;
            dropout: float, dropout rate;
            final_act: the activity function for output layer;
        '''
        super(SimpleCoder, self).__init__()
        self.layers = nn.ModuleList()
        for i, (u_i, u_o) in enumerate(zip(units[:-1], units[1:])):
            one_layer = []
            linear_layer = nn.Linear(u_i, u_o)
            one_layer.append(linear_layer)
            if i < (len(units) - 2):  # input in units
                if bn:
                    one_layer.append(nn.BatchNorm1d(u_o))
                one_layer.append(act)
                if dropout > 0.0:
                    one_layer.append(nn.Dropout(dropout))
            one_layer = nn.Sequential(*one_layer)
            self.layers.append(one_layer)
        if final_act is not None:
            self.layers.append(final_act)

    def forward(self, x):
        """ forward """
        for layer in self.layers:
            x = layer(x)
        return x

    def init_parameters(self, init, **init_params):
        """init_parameters

        :param init: init funtion;
        :param **init_params: other params for init funtion;
        """
        for one_m in self.modules():
            if isinstance(one_m, nn.Linear):
                init(one_m.weight, **init_params)


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
        low_pred, high_pred = pred[low], pred[high]
        res = self.cross_entropy(
            high_pred - low_pred,
            torch.ones(high_pred.size(0), device=pred.device)
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
        indx = torch.arange(batch_size)
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
