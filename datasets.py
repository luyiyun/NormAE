import pandas as pd
import torch.utils.data as data


class BaseData(data.Dataset):
    ''' 一个Dataset基类，可以同时处理分子和图像的数据 '''
    def __init__(self, X_df, Y_df, pre_transfer=None):
        '''
        X_df: 每一行是一个sample，一列是一个代谢物；
        Y_df：一共有4列，分别是injection order(注样顺序)、batch(批次标签)、
            group(生物学标签)和class(sample和qc的指示，其中0表示qc，
            1表示sample)，如果有些数据不提供类似的标签，则统一使用-1来表示(比如
            qc样本中没有group，有些样本不提供order或batch)
        pre_transfer：对X和Y进行的变化
        '''
        super(BaseData, self).__init__()
        self.X_df = X_df
        self.Y_df = Y_df
        self.pre_transfer = pre_transfer
        if self.pre_transfer is not None:
            self.X_df, self.Y_df = self.pre_transfer(
                self.X_df, self.Y_df)

    def __len__(self):
        return len(self.X_df)

    def __getitem__(self, indx):
        x = self.X_df.values[indx]
        y = self.Y_df.values[indx]
        return x, y

    def transform(self, trans):
        self.X_df, self.Y_df = trans(self.X_df, self.Y_df)
        return self

    @property
    def num_features(self):
        return self.X_df.shape[1]


def get_metabolic_data(x_file, y_file, pre_transfer=None, sub_qc_split=True):
    '''
    x_file：储存代谢物丰度值信息的文件路径；
    y_file: 储存injection.order、class、batch、group信息的文件路径；
    '''
    # 获取标签
    y_df = pd.read_csv(y_file, index_col='sample.name')
    y_df = y_df.dropna()
    y_num = y_df.shape[-1]
    # 获取代谢数据
    meta_df = pd.read_csv(
        x_file, index_col='name').drop(['mz', 'rt'], axis=1)
    meta_df = meta_df.T.rename_axis(index='sample', columns='meta')
    # 进行merge
    all_df = y_df.merge(
        meta_df, how='inner', left_index=True, right_index=True)

    # 28法则去除全部样本中0数量较多的变量
    meta_df, y_df = all_df.iloc[:, y_num:], all_df.iloc[:, :y_num]
    mask1 = (meta_df == 0).mean(axis=0) < 0.2
    meta_df = meta_df.loc[:, mask1]
    # 28法则去除QC样本中0数量较多的变量
    qc_mask = y_df['class'] == 'QC'
    qc_meta_df = meta_df.loc[qc_mask, :]
    mask2 = (qc_meta_df == 0).mean(axis=0) < 0.2
    meta_df = meta_df.loc[:, mask2]
    # 对于每个变量，使用其除了0以外的最小值的1/2来填补其0值
    def impute_zero(x):
        zero_mask = x == 0
        if zero_mask.any():
            new_x = x.copy()
            impute_value = x.loc[~zero_mask].min()
            new_x[zero_mask] = impute_value / 2
            return new_x
        return x
    meta_df = meta_df.apply(impute_zero, axis=0)

    # y包括injection order和batch，还有group，用于重建之后的评价
    y_df = y_df.loc[:, ['injection.order', 'batch', 'group', 'class']]
    # batch是从1开始的，但pytorch使用的标签是从0开始
    y_df.loc[:, 'batch'] -= 1
    # 为了能够输出到torch的系统中，对group进行一下改进, -1表示qc
    y_df['group'].replace('QC', '-1', inplace=True)
    y_df['group'] = y_df['group'].astype('int')
    # class也进行修改
    y_df['class'].replace({'Subject': 1, 'QC': 0}, inplace=True)

    if pre_transfer is not None:
        meta_df, y_df = pre_transfer(meta_df, y_df)
    if sub_qc_split:
        qc_index = y_df['class'] == 0
        return BaseData(meta_df[~qc_index], y_df[~qc_index]), \
            BaseData(meta_df[qc_index], y_df[qc_index])
    return BaseData(meta_df, y_df)


def get_demo_data(subject_file, qc_file, pre_transfer=None, sub_qc_split=True):
    '''
    subject_file: 储存有subject信息，第一行是注样顺序；
    metabolic_file：QC样本信息，第一行是注样顺序；
    '''
    sub_df, qc_df = pd.read_csv(subject_file), pd.read_csv(qc_file)
    sub_df, qc_df = sub_df.T, qc_df.T
    sub_df_Y, sub_df_X = sub_df.iloc[:, [0]], sub_df.iloc[:, 1:]
    qc_df_Y, qc_df_X = qc_df.iloc[:, [0]], qc_df.iloc[:, 1:]
    # 将注样顺序变量的变量名写上，并将另外几个y也补充上，但因为没有，所以用-1
    #   来填补
    sub_df_Y.columns = ['injection.order']
    qc_df_Y.columns = ['injection.order']
    sub_df_Y, qc_df_Y = impute_y_df(sub_df_Y), impute_y_df(qc_df_Y)
    sub_df_Y['class'] = 1
    qc_df_Y['class'] = 0

    # 如果不需要进行transfer而且需要分开输出
    if pre_transfer is None and sub_qc_split:
        return BaseData(sub_df_X, sub_df_Y), BaseData(qc_df_X, qc_df_Y)
    # 则剩下的情况都需要先把sub和qc并在一起
    all_df_X = pd.concat([sub_df_X, qc_df_X])
    all_df_Y = pd.concat([sub_df_Y, qc_df_Y])
    # 如果transfer还是None，说明sub_qc_split是False
    if pre_transfer is None:
        return BaseData(all_df_X, all_df_Y)
    # 剩下两种情况都需要进行transfer
    all_df_X, all_df_Y = pre_transfer(all_df_X, all_df_Y)
    # 如果需要分开
    if sub_qc_split:
        qc_index = all_df_Y['class'] == 0
        return BaseData(all_df_X[~qc_index], all_df_Y[~qc_index]), \
            BaseData(all_df_X[qc_index], all_df_Y[qc_index])
    # 最后剩下的情况是既不需要分开，还需要transferの情况
    return BaseData(all_df_X, all_df_Y)


def impute_y_df(part_df):
    '''
    part_df是一个有以下4列其中1或多列的df，此函数的作用是用-1将其补充至这4列,
    并将其顺序整理成如此
    '''
    y_names = ['injection.order', 'batch', 'group', 'class']
    y_set = set(y_names)
    new_df = part_df.copy()

    exits_set = set(part_df.columns)
    absence_set = y_set.difference(exits_set)
    for n in absence_set:
        new_df[n] = -1
    new_df = new_df.loc[:, y_names]
    return new_df


def test():
    import sys
    import time

    # sample_file = "./DATA/metabolic/sample.information.T3.csv"
    # meta_file = "./DATA/metabolic/data_T3原始数据.csv"
    # t1 = time.perf_counter()
    # subject_dat, qc_dat = get_metabolic_data(meta_file, sample_file)
    # t2 = time.perf_counter()
    # print(t2 - t1)
    # print('')

    # print(subject_dat.X_df.head())
    # print(subject_dat.Y_df.head())
    # print(qc_dat.X_df.head())
    # print(qc_dat.Y_df.head())

    # print(subject_dat[0])
    # print(qc_dat[0])

    # print(len(subject_dat))
    # print(subject_dat.num_features)
    # print(len(qc_dat))
    # print(qc_dat.num_features)

    sample_file = './DATA/Demo/sample.csv'
    qc_file = './DATA/Demo/qc.csv'
    sub_dat, qc_dat = get_demo_data(sample_file, qc_file, None)
    print(len(sub_dat), len(qc_dat))
    print(sub_dat.X_df.head())
    print(sub_dat.Y_df.head())
    print(qc_dat.X_df.head())
    print(qc_dat.Y_df.head())


if __name__ == "__main__":
    test()
