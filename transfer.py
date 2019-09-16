import os

import pandas as pd
import sklearn.preprocessing as skp


class MultiCompose:
    def __init__(self, *func):
        self.funcs = func

    def __call__(self, *args):
        for f in self.funcs:
            args = f(*args)
        return args


class LabelMapper:
    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, x, y):
        y = y.applymap(lambda x: self.mapper[x]).astype('int64')
        return x, y


class MaskFilterCol:
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, x, y):
        return x.loc[:, self.mask], y


class ZeroFilterCol:
    def __init__(self, zero_frac=0.8):
        self.zero_frac = zero_frac

    def __call__(self, x, y):
        mask = (x == 0).mean(axis=0) < self.zero_frac
        return x.loc[:, mask], y


class MeanFilterCol:
    def __init__(self, mean_thre=1):
        self.mean_thre = mean_thre

    def __call__(self, x, y):
        mask = x.mean(axis=0) > self.mean_thre
        return x.loc[:, mask], y


class StdFilterCol:
    def __init__(self, std_thre=0.5):
        self.std_thre = std_thre

    def __call__(self, x, y):
        mask = x.std(axis=0) > self.std_thre
        return x.loc[:, mask], y


class TimeNorm:
    def __init__(self, scale=(0, 1)):
        self.scale = scale

    def __call__(self, x, y):
        a, b = self.scale
        ymax, ymin = y.max(), y.min()
        yy = (y - ymin) * (b - a) / (ymax - ymin) + a
        return x, yy


class Normalization:
    Scalers = {
        'standard': skp.StandardScaler,
        'minmax': skp.MinMaxScaler,
        'maxabs': skp.MaxAbsScaler,
        'robust': skp.RobustScaler
    }

    def __init__(self, ty='standard', **kwargs):
        self.ty = ty
        self.scaler = __class__.Scalers[ty](**kwargs)
        self.fit_ind = False

    def __call__(self, x, y):
        xindex, xcolumns = x.index, x.columns  # scaler的结果是ndarray，但需要df
        if self.fit_ind:
            x = self.scaler.transform(x.values)
            return pd.DataFrame(x, index=xindex, columns=xcolumns), y
        else:
            x = self.scaler.fit_transform(x.values)
            self.fit_ind = True
            return pd.DataFrame(x, index=xindex, columns=xcolumns), y

    def reset(self):
        if self.fit_ind:
            self.scaler = __class__.Scalers[self.ty]
            self.fit_ind = False
