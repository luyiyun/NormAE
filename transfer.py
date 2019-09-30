import pandas as pd
import sklearn.preprocessing as skp


class Normalization:
    Scalers = {
        'standard': skp.StandardScaler,
        'minmax': skp.MinMaxScaler,
        'maxabs': skp.MaxAbsScaler,
        'robust': skp.RobustScaler
    }

    def __init__(self, ty='standard', dim='columns', **kwargs):
        self.ty = ty
        self.scaler = __class__.Scalers[ty](**kwargs)
        self.fit_ind = False
        self.dim = dim

    def __call__(self, x, y):
        xindex, xcolumns = x.index, x.columns  # scaler的结果是ndarray，但需要df
        if self.dim == 'rows':
            values = x.values.T
        else:
            values = x.values
        if self.fit_ind:
            x = self.scaler.transform(values)
        else:
            x = self.scaler.fit_transform(values)
            self.fit_ind = True
        if self.dim == 'row':
            return pd.DataFrame(x.T, index=xindex, columns=xcolumns), y
        else:
            return pd.DataFrame(x, index=xindex, columns=xcolumns), y


    def reset(self):
        if self.fit_ind:
            self.scaler = __class__.Scalers[self.ty]
            self.fit_ind = False

    def inverse_transform(self, x, y):
        xindex, xcolumns = x.index, x.columns  # scaler的结果是ndarray，但需要df
        res = self.scaler.inverse_transform(x.values)
        return pd.DataFrame(res, index=xindex, columns=xcolumns), y
