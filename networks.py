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
        dropout=0.5, spectral_norm=False, norm=None, lrelu=True
    ):
        super(Coder, self).__init__()
        self.in_f = in_f
        act = nn.LeakyReLU() if lrelu else nn.ReLU()
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


class SimpleCoder(nn.Module):
    def __init__(
        self, units, dropout=None, norm=None, lrelu=True, last_act=None
    ):
        '''
        units是节点个数，其中第一个是输入维度，最后一个是输出维度
        '''
        super(SimpleCoder, self).__init__()
        model = []
        for i, (u1, u2) in enumerate(zip(units[:-1], units[1:])):
            model.append(nn.Linear(u1, u2))
            if i < (len(units) - 2):  # 因为units是包括了输入层的
                model.append(nn.LeakyReLU() if lrelu else nn.ReLU())
                if norm is not None:
                    model.append(norm(u2))
                # dropout可以是None(没有dropout)，也可以是float(除了最后一
                # 层，其他层都加这个dropout)，也可以是list(表示第几层指定的
                # dropout是多少，None表示这一层不加)
                if isinstance(dropout, float):
                    model.append(nn.Dropout(dropout))
                elif isinstance(dropout, (list, tuple)):
                    dropout_i = dropout[i]
                    model.append(nn.Dropout(dropout_i))
            else:
                if last_act is not None:
                    model.append(last_act)
                # 有可能dropout为list的时候还会指定最后一层加dropout
                if isinstance(dropout, (list, tuple)):
                    model.append(nn.Dropout(dropout[len(units)-2]))
        self.model = nn.Sequential(*model)


    def forward(self, x):
        return self.model(x)


''' Loss '''


class RankLoss(nn.Module):
    def __init__(self, reduction='mean', classification=None):
        super(RankLoss, self).__init__()
        if reduction == 'mean':
            self.agg_func = torch.mean
        elif reduction == 'sum':
            self.agg_func = torch.sum
        self.classification = (classification is not None)
        if self.classification:
            self.ce = classification

    def forward(self, pred, rank_batch):
        if self.classification:
            pred_rank, pred_cls = pred[:, 0], pred[:, 1:]
        else:
            pred_rank = pred
        # rank loss
        low, high = self._comparable_pairs(rank_batch)
        low_pred, high_pred = pred_rank[low], pred_rank[high]
        diff = (1 - high_pred + low_pred).clamp(min=0)
        diff = diff ** 2
        rank_loss = self.agg_func(diff)
        # cls loss
        if self.classification:
            batch_index = rank_batch[:, 1].long()
            cls_loss = self.ce(pred_cls, batch_index)
            return rank_loss + cls_loss
        return rank_loss

    @staticmethod
    def _comparable_pairs(rank_batch):
        ''' 得到所有样本对，其中并把排序在前的放在前面 '''
        # 这里接受的rank_batch可能只有rank
        batch_index = None
        if rank_batch.dim() == 1:
            truth_rank = rank_batch
        elif rank_batch.size(1) == 1:
            truth_rank = rank_batch[:, 0]
        else:
            truth_rank, batch_index = rank_batch[:, 0], rank_batch[:, 1]
        batch_size = len(truth_rank)
        indx = torch.arange(batch_size)
        pairs1 = indx.repeat(batch_size)
        pairs2 = indx.repeat_interleave(batch_size, dim=0)
        # 选择第一个小于第二个的元素
        time_mask = truth_rank[pairs1] < truth_rank[pairs2]
        pairs1, pairs2 = pairs1[time_mask], pairs2[time_mask]
        # 选择在一个batch的pairs
        if batch_index is not None:
            batch_mask = batch_index[pairs1] == batch_index[pairs2]
            pairs1, pairs2 = pairs1[batch_mask], pairs2[batch_mask]
        return pairs1, pairs2


class ClassicalMSE(nn.Module):
    '''
    这个是和普通的mse没有区别，如果说唯一的区别，就是其接受的target是01234这样
    的数字标签，而pred是batch x 4的tensor，所以需要先把target one-hot一下而已
    '''
    def __init__(self, cate_num, l1=True, **kwargs):
        super(ClassicalMSE, self).__init__()
        self.mse = nn.L1Loss(**kwargs) if l1 else \
            nn.MSELoss(**kwargs)
        self.identity_matrix = torch.arange(cate_num).diag()

    def forward(self, pred, target):
        target_oh = self.identity_matrix[target.squeeze().long()].to(pred)
        return self.mse(pred, target_oh)


class CustomCrossEntropy(nn.Module):
    def __init__(self, **kwargs):
        super(CustomCrossEntropy, self).__init__()
        self.ce = nn.CrossEntropyLoss(**kwargs)

    def forward(self, pred, target):
        return self.ce(pred, target.to(torch.long).squeeze())


class LabelSmoothing(nn.Module):
    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.size = size
        self.smoothing = smoothing
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        self.pseudo_identity = torch.empty((self.size, self.size))
        self.pseudo_identity.uniform_(0, self.smoothing)
        self.pseudo_identity = torch.eye(self.size) - self.pseudo_identity
        self.pseudo_identity.abs_()
        target = target.long()
        true_dist = self.pseudo_identity[target].to(input)
        return self.criterion(self.log_softmax(input), true_dist)


def test():
    # x = torch.randn(32, 1000)
    # net = Coder(1000, 100)
    # res = net(x)
    # print(res.shape)


    # truth = torch.randperm(100)
    # pred = torch.rand(100)
    # criterion = RankLoss()
    # print(criterion(truth, pred))
    # print(truth)
    # print(pred)
    # pred, _ = pred.sort()
    # truth, _ = truth.sort()
    # print(criterion(truth, pred))
    # print(truth)
    # print(pred)
    crit = LabelSmoothing(size=5, smoothing=0.1)
    predict = torch.tensor([
        [0, 0.2, 0.7, 0.1, 0],
        [0, 0.9, 0.2, 0.1, 0],
        [1, 0.2, 0.7, 0.1, 0]
    ])
    v = crit(predict.log(), torch.tensor([2, 1, 0], dtype=torch.long))
    print(v)


if __name__ == '__main__':
    test()
