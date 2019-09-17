import torch
import torch.nn as nn


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
        self, units, dropout=None, norm=None, lrelu=True, last_act=None,
        spectral_norm=False
    ):
        '''
        units是节点个数，其中第一个是输入维度，最后一个是输出维度
        '''
        super(SimpleCoder, self).__init__()
        model = []
        for i, (u1, u2) in enumerate(zip(units[:-1], units[1:])):
            if spectral_norm:
                model.append(nn.utils.spectral_norm(nn.Linear(u1, u2)))
            else:
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


class CEwithLabelSmooth(nn.Module):
    '''
    带有标签平滑的判别loss，如果smoothing=0则就是普通的ce，如果不=0，则对于当前
    样本，如果其标签是0, 则标签使用(1-lambda, lambda, lambda, lambda)，其中
    lambda是从uniform(o, smoothing)中采样得到
    '''
    def __init__(self, smoothing=0.0):
        super(CEwithLabelSmooth, self).__init__()
        self.smoothing = smoothing
        self.log_softmax = nn.LogSoftmax(dim=1)
        if smoothing == 0.0:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, target):
        ''' 注意，这里的pred是预测的分类部分 '''
        use_target = target[:, 1].long()
        if self.smoothing == 0.0:
            return self.criterion(pred, use_target)
        else:
            size = pred.size(1)
            pseudo_identity = torch.empty((size, size))
            pseudo_identity.uniform_(0, self.smoothing)
            pseudo_identity = torch.eye(size) - pseudo_identity
            pseudo_identity.abs_()
            true_dist = pseudo_identity[use_target].to(pred)
            return self.criterion(self.log_softmax(pred), true_dist)


class RankLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RankLoss, self).__init__()
        if reduction == 'mean':
            self.agg_func = torch.mean
        elif reduction == 'sum':
            self.agg_func = torch.sum

    def forward(self, pred, target):
        ''' 注意，这里的pred是预测的排序部分 '''
        # rank loss
        low, high = self._comparable_pairs(target[:, 0], target[:, 1])
        low_pred, high_pred = pred[low], pred[high]
        diff = (1 - high_pred + low_pred).clamp(min=0)
        diff = diff ** 2
        rank_loss = self.agg_func(diff)
        return rank_loss

    @staticmethod
    def _comparable_pairs(true_rank, true_batch=None):
        '''
        true_rank: 真实的每个样本的排序；
        true_batch: 每个样本属于的批次标签；

        这里有true_batch的意思是，只有在同一个批次内样本才有比较的必要，不在同
        一个批次没有比较的必要。注意到，实际上我们不需要判断是否需要true_batch，
        如果没有true_batch标签，在dataset的步骤里将其填补成-1，大家都是-1，则
        所有的pairs都会被考虑，和不考虑true_batch的结果是一样的，但可能速度会
        慢一些。
        '''
        # 这里接受的rank_batch可能只有rank
        batch_size = len(true_rank)
        indx = torch.arange(batch_size)
        pairs1 = indx.repeat(batch_size)
        pairs2 = indx.repeat_interleave(batch_size, dim=0)
        # 选择第一个小于第二个的元素
        time_mask = true_rank[pairs1] < true_rank[pairs2]
        pairs1, pairs2 = pairs1[time_mask], pairs2[time_mask]
        # 选择在一个batch的pairs
        if true_batch is not None:
            batch_mask = true_batch[pairs1] == true_batch[pairs2]
            pairs1, pairs2 = pairs1[batch_mask], pairs2[batch_mask]
        return pairs1, pairs2


class SmoothCERankLoss(nn.Module):
    def __init__(self, smoothing=0.2, reduction='mean', ce_w=1.0, rank_w=1.0):
        ''' 如果ce_w=0.0默认pred中没有分类的部分，rank也是一样 '''
        assert ce_w > 0.0 or rank_w > 0.0
        super(SmoothCERankLoss, self).__init__()
        self.ce_w, self.rank_w = ce_w, rank_w
        if ce_w > 0.:
            self.ce = CEwithLabelSmooth(smoothing)
        if rank_w > 0.:
            self.rank = RankLoss(reduction)

    def forward(self, pred, target):
        if self.ce_w > 0. and self.rank_w > 0.:
            pred_rank, pred_cls = pred[:, 0], pred[:, 1:]
            return self.rank(pred_rank, target) + self.ce(pred_cls, target)
        elif self.ce_w > 0.:
            return self.ce(pred, target)
        elif self.rank_w > 0.:
            return self.rank(pred, target)


def test():
    # x = torch.randn(32, 1000)
    # net = Coder(1000, 100)
    # res = net(x)
    # print(res.shape)


    # rank and ce
    pass


if __name__ == '__main__':
    test()
