import pandas as pd
import sklearn.preprocessing as skp


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
