import torch
import torch.nn as nn
import torch.nn.functional as F


''' 网络结构部分 '''


def bottle_linear(
    in_f, out_f, bottle_f, act=nn.LeakyReLU(), dropout=0.0,
    spectual_norm=False
):
    ''' in_f-bottle_f --> act --> dropout --> bottle_f-out_f '''
    if spectual_norm:
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(in_f, bottle_f)),
            act, nn.Dropout(dropout, inplace=True),
            nn.utils.spectral_norm(nn.Linear(bottle_f, out_f))
        )
    return nn.Sequential(
        nn.Linear(in_f, bottle_f), act, nn.Dropout(dropout, inplace=True),
        nn.Linear(bottle_f, out_f)
    )


class ResBotBlock(nn.Module):
    '''
    x --> (bottle_neck_layer --> act)*(n-1) --> bottle_neck_layer --> +x --> bn
    '''
    def __init__(
        self, input_shape, act, bottle_units, bottle_act=None, dropout=0.5,
        spectral_norm=False, norm=None
    ):
        super(ResBotBlock, self).__init__()
        if not isinstance(bottle_units, (tuple, list)):
            bottle_units = [bottle_units]
        if bottle_act is None:
            bottle_act = act
        self.bottle_modules = []
        for i, bu in enumerate(bottle_units):
            self.bottle_modules.append(
                bottle_linear(
                    input_shape, input_shape, bu, bottle_act, dropout,
                    spectral_norm
                )
            )
            if i != (len(bottle_units) - 1):
                self.bottle_modules.append(act)
        self.bottle_modules = nn.Sequential(*self.bottle_modules)
        if norm == 'BN':
            self.bn = nn.BatchNorm1d(input_shape)
        elif norm == 'IN':
            self.bn = nn.InstanceNorm1d(input_shape)
        else:
            self.bn = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.bottle_modules(x)
        out += identity
        return self.bn(out)


class Coder(nn.Module):
    def __init__(
        self, in_f, out_f, hidden_unit=500, block_num=5, bottle_unit=50,
        dropout=0.5, spectral_norm=False, norm=None
    ):
        super(Coder, self).__init__()
        self.in_f = in_f
        act = nn.LeakyReLU()
        bottle_units = [bottle_unit] * 2
        modules = [nn.Linear(in_f, hidden_unit), act]
        for _ in range(block_num):
            modules.append(
                ResBotBlock(
                    hidden_unit, act, bottle_units, dropout=dropout,
                    spectral_norm=spectral_norm, norm=norm
                )
            )
            modules.append(act)
        last_layer = nn.Linear(hidden_unit, out_f)
        if spectral_norm:
            last_layer = nn.utils.spectral_norm(last_layer)
        modules.append(last_layer)
        self.f = nn.Sequential(*modules)

    def forward(self, x):
        return self.f(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


''' Loss '''


class RankLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RankLoss, self).__init__()
        if reduction == 'mean':
            self.agg_func = torch.mean
        elif reduction == 'sum':
            self.agg_func = torch.sum

    def forward(self, pred_rank, rank_batch):
        # rank loss
        low, high = self._comparable_pairs(rank_batch)
        low_pred, high_pred = pred_rank[low], pred_rank[high]
        rank_loss = self.agg_func(
            (1 - high_pred + low_pred).clamp(min=0) ** 2)
        return rank_loss

    @staticmethod
    def _comparable_pairs(rank_batch):
        ''' 得到所有样本对，其中并把排序在前的放在前面 '''
        truth_rank, batch_index = rank_batch[:, 0], rank_batch[:, 1]
        batch_size = len(truth_rank)
        indx = torch.arange(batch_size)
        pairs1 = indx.repeat(batch_size)
        pairs2 = indx.repeat_interleave(batch_size, dim=0)
        # 选择第一个小于第二个的元素
        time_mask = truth_rank[pairs1] < truth_rank[pairs2]
        pairs1, pairs2 = pairs1[time_mask], pairs2[time_mask]
        # 选择在一个batch的pairs
        batch_mask = batch_index[pairs1] == batch_index[pairs2]
        pairs1, pairs2 = pairs1[batch_mask], pairs2[batch_mask]
        return pairs1, pairs2


def test():
    x = torch.randn(32, 1000)
    net = Coder(1000, 100)
    res = net(x)
    print(res.shape)


    truth = torch.randperm(100)
    pred = torch.rand(100)
    criterion = RankLoss()
    print(criterion(truth, pred))
    print(truth)
    print(pred)
    pred, _ = pred.sort()
    truth, _ = truth.sort()
    print(criterion(truth, pred))
    print(truth)
    print(pred)


if __name__ == '__main__':
    test()
