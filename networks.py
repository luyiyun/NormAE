import torch
import torch.nn as nn


''' 网络结构部分 '''


class ResBlock(nn.Module):
    '''
    x(in_f) --bottle_linear--> h(inter_f) --bottle_linear--> h(inter_f) \
        ... --bottle_linear--> out(out_f)+x(in_f) --bn--> <return>
    '''
    def __init__(
        self, in_f, out_f, act, bottle_units, bottle_act=None, dropout=0.0
    ):
        super(ResBlock, self).__init__()
        # 保存属性
        self.act = act
        self.bottle_act = bottle_act if bottle_act is not None else act
        self.dropout = dropout
        # 对bottle_units进行处理
        if isinstance(bottle_units, int):
            bottle_units = [bottle_units]
        elif not isinstance(bottle_units, (tuple, list)):
            raise ValueError
        # 如果bottle neck只有一个，则只需要一层bottle_linear即可，所以
        #   中间层就是输出层，如果还有其他的，则使用输入和输出的平均值
        inter_f = out_f if len(bottle_units) == 1 else int((in_f + out_f) / 2)
        # 开始搭建网络结构
        self.bottle_modules = []
        for i, bu in enumerate(bottle_units):
            # 除了最后一层，其他层后面都需要加上激活函数
            if len(self.bottle_modules) > 0:
                self.bottle_modules.append(self.act)
            # 第一层和最后一层的输入和输出有所不同
            if i == 0:
                self.bottle_modules.append(
                    self.bottle_linear(in_f, inter_f, bu))
            elif i == len(bottle_units) - 1:
                self.bottle_modules.append(
                    self.bottle_linear(inter_f, out_f, bu))
            else:
                self.bottle_modules.append(
                    self.bottle_linear(inter_f, inter_f, bu))
        self.bottle_modules = nn.Sequential(*self.bottle_modules)
        self.bn = nn.BatchNorm1d(out_f)

    def forward(self, x):
        out = self.bottle_modules(x)
        in_shape, out_shape = x.size(1), out.size(1)
        if in_shape == out_shape:
            return self.bn(x + out)
        elif in_shape > out_shape:
            # 如果输出比较小，则只是用输入的一部分
            return self.bn(x[:, :out_shape] + out)
        # 如果输入比较小，则使用0来进行填补
        zero_tensor = torch.zeros((x.size(0), out_shape-in_shape)).to(x)
        x = torch.cat([x, zero_tensor], dim=1)
        return self.bn(x + out)

    def bottle_linear(self, in_f, out_f, bottle_f):
        ''' in_f --linear--> bottle_f --> --act-dropout-linear--> out_f '''
        return nn.Sequential(
            nn.Linear(in_f, bottle_f), self.act,
            nn.Dropout(self.dropout, inplace=True),
            nn.Linear(bottle_f, out_f)
        )


class ResNet(nn.Module):
    def __init__(self, units, bottle_units=(50, 50), dropout=0.0):
        super(ResNet, self).__init__()
        in_f, out_f, act = units[0], units[-1], nn.LeakyReLU()
        # 开始搭建网络
        self.layers = nn.ModuleList()
        # 首先第一层，简单的fc，将维度映射下去
        self.layers.append(nn.Sequential(nn.Linear(in_f, units[1]), act))
        # 中间的层，每一层是一个ResBlock，后面再接一个激活函数
        for i, j in zip(units[1:-2], units[2:-1]):
            one_layer = [
                ResBlock(i, j, act, bottle_units, dropout=dropout),
                act
            ]
            self.layers.append(nn.Sequential(*one_layer))
        # 最后一层，无激活函数的fc
        self.layers.append(nn.Sequential(nn.Linear(units[-2], out_f)))
        # 重新初始化参数
        self.init_parameters(nn.init.normal_, std=0.02)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def init_parameters(self, init, **init_params):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init(m.weight, **init_params)


class SimpleCoder(nn.Module):
    def __init__(self, units, act=nn.LeakyReLU(), bn=True):
        '''
        构建多层全连接

        args:
            units: list，表示每一次的节点数，注意，第一个元素是输入层，最后一个
                元素是输出层，其他是隐藏层；
        '''
        super(SimpleCoder, self).__init__()
        self.layers = nn.ModuleList()
        for i, (u1, u2) in enumerate(zip(units[:-1], units[1:])):
            one_layer = []
            linear_layer = nn.Linear(u1, u2)
            one_layer.append(linear_layer)
            if i < (len(units) - 2):  # 因为units是包括了输入层的
                one_layer.append(act)
                if bn:
                    one_layer.append(nn.BatchNorm1d(u2))
            one_layer = nn.Sequential(*one_layer)
            self.layers.append(one_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


''' Loss '''


class ClswithLabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0, leastsquare=False):
        '''
        主要用于discriminator
        带有标签平滑的判别loss，如果smoothing=0且leastsquare=False则就是普通的
        ce，如果smoothing!=0且leastsquare=False，则对于当前样本，如果其标签是0,
        则标签使用(1-lambda, lambda, lambda, lambda)，其中lambda是从uniform
        (o, smoothing)中采样得到，然后计算输出和此标签间的交叉熵；如果smoothing!=
        0且leastsquare=True，则计算输出和此平滑标签间的mse；如果smoothing=0且
        leastsquare=True则计算的是真实标签和输出间的mse。

        args:
        smoothing: 标签平滑使用的小扰动的大小;
        leastsquare: Boolean，是否使用mse来代替cross entropy；
        '''
        super(ClswithLabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
        self.leastsquare = leastsquare
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.kld = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, target):
        '''

        args:
            pred: 是预测的分类部分，即如果是4分类，则此tensor的维度就是batch x 4;
            target: shape=(batch,)，储存标签信息(1,2,3,4);
        '''
        # 使用mse时，不论是否使用标签平滑，都需要使用long格式的标签来进行
        #   one-hot转变或加噪声
        # 而对于ce，则也都需要long格式的标签来直接输入或者加噪声
        target = target.long()
        size = pred.size(1)

        if self.smoothing == 0.0 and not self.leastsquare:
            # 没有标签平滑且不使用最小二乘回归
            return self.ce(pred, target)
        elif self.smoothing == 0.0:
            # 没有标签平滑但需要使用最小二乘回归
            identity = torch.eye(size)
            one_hot_label = identity[target].to(pred)
            return self.mse(pred, one_hot_label)
        else:
            # 需要使用标签平滑
            pseudo_identity = torch.empty((size, size))
            pseudo_identity.uniform_(0, self.smoothing)
            pseudo_identity = torch.eye(size) - pseudo_identity
            pseudo_identity.abs_()
            true_dist = pseudo_identity[target].to(pred)
            # 如果是使用最小二乘回归
            if self.leastsquare:
                return self.mse(pred, true_dist)
        # 如果使用标签平滑且不使用最小二乘回归
        return self.kld(self.log_softmax(pred), true_dist)


class OrderLoss(nn.Module):
    def __init__(self, loss_type='paired_ce'):
        super(OrderLoss, self).__init__()
        self.loss_type=loss_type
        self.softmax = nn.Softmax(dim=0)
        self.ce = nn.BCEWithLogitsLoss()
        self.log_softmax = nn.LogSoftmax(dim=0)
        self.kld = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, target, group=None):
        '''
        args:
            pred: 是预测的order部分，batch x 1;
            target: size=(batch,)，储存injection.order信息;
            group: tensor, shape=(batch,)，用于计算rank loss使用;
        '''
        pred = pred.squeeze()
        target = target.squeeze().float()
        if self.loss_type == 'listnet':
            if group is None:
                #  pred = self.softmax(pred)
                #  target = self.softmax(target)
                #  return -(target * pred.log()).sum()
                pred = self.log_softmax(pred)
                target = self.log_softmax(target)
                return self.kld(pred, target)
            else:
                # 对于提供了group，则需要对每个group的值进行计算后再加在一起
                unique_group = torch.unique(group)
                res = 0.
                for g in unique_group:
                    # 因为这个g是unique得到的，所以不会出现不存在的现象
                    #  pred_g = self.softmax(pred[group == g])
                    #  target_g = self.softmax(target[group == g])
                    #  res -= (target_g * pred_g.log()).sum()
                    pred_g = self.log_softmax(pred[group == g])
                    target_g = self.log_softmax(target[group == g])
                    res += self.kld(pred_g, target_g)
                return res
        elif self.loss_type == 'listmle':
            if group is None:
                sort_index = target.argsort()
                sort_pred = pred[sort_index]
                sort_pred_exp = sort_pred.exp()
                return torch.mean(sort_pred_exp.cumsum(0).log() - sort_pred)
            else:
                # 对于提供了group，则需要对每个group的值进行计算后再加在一起
                unique_group = torch.unique(group)
                res = 0.
                for g in unique_group:
                    # 因为这个g是unique得到的，所以不会出现不存在的现象
                    sort_index_g = target[group == g].argsort()
                    sort_pred_g = pred[group == g][sort_index_g]
                    sort_pred_exp_g = sort_pred_g.exp()
                    res -= torch.mean(
                        sort_pred_exp_g.cumsum(0).log() - sort_pred_g)
                return res
        else:
            # rank loss
            low, high = self._comparable_pairs(target, group)
            if len(low) == 0:  # 有可能是空的，这时求mean是nan
                return torch.tensor(0.).float()
            low_pred, high_pred = pred[low], pred[high]
            res = self.ce(
                high_pred - low_pred,
                torch.ones(high_pred.size(0),
                           device=pred.device)
            )
            return res

    @staticmethod
    def _comparable_pairs(true_rank, group=None):
        '''
        返回所有可比的样本对的indice

        args:
            true_rank: 真实的每个样本的排序；
            true_batch: 每个样本属于的批次标签；
            group: tensor, shape=(batch,)，用于计算rank loss使用;

        '''
        # 这里接受的rank_batch可能只有rank
        batch_size = len(true_rank)
        indx = torch.arange(batch_size)
        pairs1 = indx.repeat(batch_size)
        pairs2 = indx.repeat_interleave(batch_size, dim=0)
        # 选择第一个小于第二个的元素，因为实际上我们需要的是两两组合
        time_mask = true_rank[pairs1] < true_rank[pairs2]
        pairs1, pairs2 = pairs1[time_mask], pairs2[time_mask]
        # 如果存在group，则需要再排除不在一个group内的pairs
        if group is not None:
            batch_mask = group[pairs1] == group[pairs2]
            pairs1, pairs2 = pairs1[batch_mask], pairs2[batch_mask]
        return pairs1, pairs2


class ClsOrderLoss(nn.Module):
    def __init__(
        self, cls_leastsquare=False, order_losstype='paired_ce',
        cls_smoothing=0.2
    ):
        super(ClsOrderLoss, self).__init__()
        self.cls_loss = ClswithLabelSmoothLoss(
            smoothing=cls_smoothing, leastsquare=cls_leastsquare)
        self.order_loss = OrderLoss(loss_type=order_losstype)

    def forward(
        self, cls_pred=None, cls_target=None, order_pred=None,
        order_target=None, order_group=None, cls_weight=1.0,
        order_weight=1.0
    ):
        '''
        args:
            cls_pred: 分类的预测，batch x num_cls；
            cls_target: 分类的标签，batch, [1, 2, 3, 2];
            order_pred: 排序的预测，batch;
            order_target: 排序的标签, batch；
            order_group: OrderLoss的group参数；
        '''
        all_loss = 0.
        warning_indice = 0
        if cls_pred is not None and cls_target is not None:
            cls_loss_output =  self.cls_loss(cls_pred, cls_target)
            all_loss += cls_weight * cls_loss_output
        else:
            warning_indice += 1
        if order_pred is not None and order_pred is not None:
            order_loss_output = self.order_loss(
                order_pred, order_target, order_group)
            all_loss += order_weight * order_loss_output
        else:
            warning_indice += 1
        assert warning_indice < 2
        return all_loss


def test():
    cls_pred = torch.randn(32, 4)
    cls_target = torch.randint(4, size=(32,))
    order_pred = torch.randn(32, 1)
    order_target = torch.arange(32).permute(0)
    raise NotImplementedError


if __name__ == '__main__':
    test()
