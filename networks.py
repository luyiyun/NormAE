import torch
import torch.nn as nn


''' 网络结构部分 '''


class SimpleCoder(nn.Module):
    def __init__(
        self, units, dropout=None, norm=None, lrelu=True, last_act=None,
        spectral_norm=False, return_hidden=True
    ):
        '''
        构建多层全连接

        args:
            units: list，表示每一次的节点数，注意，第一个元素是输入层，最后一个
                元素是输出层，其他是隐藏层；
            dropout: None or float or list of float，如果是float需要在0-1之间，
                表示是否在每个linear后加dropout，如果是None则不加，float这
                表示dropout的rate， 每个linear后的dropout都是一样的，如果是list
                则表示每一个隐层中的dropout rate,注意到，最后一个hidden layer和
                output layer间不会加入dropout；
            norm: None or nn.BatchNorm1d，在每一层linear后，relu前加入BN，如果
                是None则没有，最后的hidden layer和output layer间不会加入；
            lrelu: Boolean，True则每一层激活函数为leaky relu, False则使用relu，
                这不会决定output的激活函数;
            last_act: None or activitiy modules, 如果是None则output没有激活函数，
                不然则使用这个module作为output layer的激活函数；
            spectral_norm: Boolean，如果是True，则每个linear module是被谱归一化
                的，False则没有；
            return_hidden: Boolean，如果是True则forward返回每个hidden layers和
                output layer的输出，如果是False则只返回output layer的输出；
        '''
        super(SimpleCoder, self).__init__()
        self.return_hidden = return_hidden
        self.layers = nn.ModuleList()
        for i, (u1, u2) in enumerate(zip(units[:-1], units[1:])):
            one_layer = []
            linear_layer = nn.Linear(u1, u2)
            if spectral_norm:
                linear_layer = nn.utils.spectral_norm(linear_layer)
            one_layer.append(linear_layer)
            if i < (len(units) - 2):  # 因为units是包括了输入层的
                one_layer.append(nn.LeakyReLU() if lrelu else nn.ReLU())
                if norm is not None:
                    one_layer.append(norm(u2))
                # dropout可以是None(没有dropout)，也可以是float(除了最后一
                # 层，其他层都加这个dropout)，也可以是list(表示第几层指定的
                # dropout是多少，None表示这一层不加)
                if isinstance(dropout, float):
                    one_layer.append(nn.Dropout(dropout))
                elif isinstance(dropout, (list, tuple)):
                    dropout_i = dropout[i]
                    one_layer.append(nn.Dropout(dropout_i))
            else:
                if last_act is not None:
                    one_layer.append(last_act)
            one_layer = nn.Sequential(*one_layer)
            self.layers.append(one_layer)

    def forward(self, x):
        layers_out = []
        for layer in self.layers:
            x = layer(x)
            layers_out.append(x)
        if self.return_hidden:
            return layers_out
        return layers_out[-1]


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
    def __init__(self, leastsquare=False, group=None):
        '''
        discriminator的关于injection.order的部分，可以选择使用rank loss和最小
        二乘回归，如果是使用最小二乘回归，则直接将injection.order当做一个连续
        的数值进行预测，如果是rank loss则将pred看做是输出的秩次，其只会保证秩次
        的顺序。
        如果使用的rankloss，并且提供了group，则计算rank loss的时候只会对处于
        一个group内的样本进行比较，即认为不在一个group的样本间没有可比性，不会
        参与loss的计算，不会提供信息。

        args:
            leastsquare: Boolean，是否使用mse来代替rank loss；
            group: tensor, shape=(batch,)，用于计算rank loss使用;
        '''
        super(OrderLoss, self).__init__()
        self.leastsquare=leastsquare
        self.group = group
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        '''

        args:
            pred: 是预测的order部分，batch x 1;
            target: size=(batch,)，储存injection.order信息;
        '''
        pred = pred.squeeze()
        target = target.float()
        if self.leastsquare:
            return self.mse(pred, target)
        else:
            # rank loss
            low, high = self._comparable_pairs(target)
            low_pred, high_pred = pred[low], pred[high]
            diff = (1 - high_pred + low_pred).clamp(min=0)
            diff = diff ** 2
            return diff.mean()

    def _comparable_pairs(self, true_rank):
        '''
        返回所有可比的样本对的indice

        args:
            true_rank: 真实的每个样本的排序；
            true_batch: 每个样本属于的批次标签；

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
        if self.group is not None:
            batch_mask = self.group[pairs1] == self.group[pairs2]
            pairs1, pairs2 = pairs1[batch_mask], pairs2[batch_mask]
        return pairs1, pairs2


class ClsOrderLoss(nn.Module):
    def __init__(
        self, cls_weight=1.0, order_weight=1.0, cls_leastsquare=False,
        order_leastsquare=False, cls_smoothing=0.2, order_group=None
    ):
        '''
        这个是将上面两个loss结合在一起的loss module，便于管理和使用。

        args:
            cls_weight: 分类部分所占的weight，如果是0.0则没有分类；
            order_weight: 排序部分所占的weight，如果是0.0则没有排序；
            cls_leastsquare: ClswithLabelSmoothLoss的leastsquare参数；
            order_leastsquare: OrderLoss的leastsquare参数；
            cls_smoothing: ClswithLabelSmoothLoss的smoothin参数；
            order_group: OrderLoss的group参数；
        '''
        assert cls_weight > 0.0 or order_weight > 0.0
        super(ClsOrderLoss, self).__init__()
        self.cls_weight, self.order_weight = cls_weight, order_weight
        self.cls_loss = ClswithLabelSmoothLoss(
            smoothing=cls_smoothing, leastsquare=cls_leastsquare)
        self.order_loss = OrderLoss(
            leastsquare=order_leastsquare, group=order_group)

    def forward(
        self, cls_pred=None, cls_target=None, order_pred=None,
        order_target=None
    ):
        '''
        args:
            cls_pred: 分类的预测，batch x num_cls；
            cls_target: 分类的标签，batch, [1, 2, 3, 2];
            order_pred: 排序的预测，batch;
            order_target: 排序的标签, batch；
        '''
        all_loss = 0.
        warning_indice = 0
        if cls_pred is not None and cls_target is not None:
            cls_loss_output = self.cls_weight * \
                self.cls_loss(cls_pred, cls_target)
            all_loss += cls_loss_output
        else:
            warning_indice += 1
        if order_pred is not None and order_pred is not None:
            order_loss_output = self.order_weight * \
                self.order_loss(order_pred, order_target)
            all_loss += order_loss_output
        else:
            warning_indice += 1
        assert warning_indice < 2
        return all_loss



def test():
    cls_pred = torch.randn(32, 4)
    cls_target = torch.randint(4, size=(32,))
    order_pred = torch.randn(32, 1)
    order_target = torch.arange(32).permute(0)

    criterion = ClsOrderLoss(
        1.0, 1.0, True, True, 0.2, cls_target
    )
    print(criterion(cls_pred, cls_target, order_pred, order_target))

    criterion = ClsOrderLoss(
        1.0, 1.0, True, True, 0.2, cls_target
    )
    print(criterion(None, None, order_pred, order_target))

    criterion = ClsOrderLoss(
        1.0, 1.0, False, False, 0.2, cls_target
    )
    print(criterion(cls_pred, cls_target, order_pred, order_target))

    criterion = ClsOrderLoss(
        1.0, 1.0, False, False, 0.0, None
    )
    print(criterion(cls_pred, cls_target, order_pred, order_target))

    criterion = ClsOrderLoss(
        1.0, 1.0, True, True, 0.2, cls_target
    )
    print(criterion(None, None, None, None))


if __name__ == '__main__':
    test()
