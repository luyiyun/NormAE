import os

import pandas as pd
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


class BaseData(data.Dataset):
    ''' 一个Dataset基类，可以同时处理分子和图像的数据 '''
    def __init__(
        self, X_df, Y_df, meta_df=None, X_loader=None, pre_transfer=None,
        transfer=None,
    ):
        super(BaseData, self).__init__()
        self.X_df = X_df
        self.Y_df = Y_df
        self.meta_df = meta_df
        self.X_loader = X_loader
        self.transfer = transfer
        self.pre_transfer = pre_transfer
        if self.pre_transfer is None:
            self.X_df_trans = self.X_df
            self.Y_df_trans = self.Y_df
        else:
            self.X_df_trans, self.Y_df_trans = self.pre_transfer(
                self.X_df, self.Y_df
            )

    def __len__(self):
        return len(self.X_df_trans)

    def __getitem__(self, indx):
        x = self.X_df_trans.values[indx]
        y = self.Y_df_trans.values[indx]
        if self.X_loader is not None:
            x = self.X_loader(x)
        if self.transfer is not None:
            x = self.transfer(x)
        res = [x, y]
        if self.meta_df is not None:
            meta = self.meta_df.values[indx]
            res.append(meta)
        return tuple(res)

    def split(
        self, test_size, valid_size=None, shuffle=True,
        random_seed=1234, train_kwargs={}, test_kwargs={}, valid_kwargs={}
    ):
        '''
        将此Data分成2或3个Data，用于train, valid, test数据集分割
        train_kwargs, test_kwargs, valid_kwargs: 分别对于不同数据集可能有
            不同的处理，这里使用这3个参数来完成，其是一个dict，可以写入的参数有
            transfer或pre_transfer等，实际上这个的内容由继承的子类的__init__函数
            需要什么参数来决定，在这里实现是因为这个split逻辑上对大多数数据集都是
            适用的，所以直接在BaseData上实现，但为了使得到的对象是子类的对象，所以
            设计了这个。
        '''
        # 先将test数据集分出来
        stra_y = self.Y_df_trans.iloc[:, 0].values
        split_dfs = [self.X_df_trans, self.Y_df_trans]
        if self.meta_df is not None:
            split_dfs.append(self.meta_df)
        split_res = train_test_split(
            *split_dfs, test_size=test_size,
            random_state=random_seed, shuffle=shuffle, stratify=stra_y
        )
        X_df_train, X_df_test, Y_df_train, Y_df_test = split_res[:4]
        if self.meta_df is not None:
            meta_df_train, meta_df_test = split_res[4:]
        else:
            meta_df_train, meta_df_test = None, None
        # 如果valid_size不是None，则在从中分出验证集
        if valid_size is not None:
            stra_y2 = Y_df_train.iloc[:, 0].values
            split_dfs = [X_df_train, Y_df_train]
            if self.meta_df is not None:
                split_dfs.append(meta_df_train)
            split_res = train_test_split(
                *split_dfs, test_size=valid_size / (1 - test_size),
                random_state=random_seed, shuffle=shuffle, stratify=stra_y2
            )
            X_df_train, X_df_valid, Y_df_train, Y_df_valid = split_res[:4]
            if self.meta_df is not None:
                meta_df_train, meta_df_valid = split_res[4:]
            else:
                meta_df_train, meta_df_valid = None, None
        class_init = self.__class__  # 这样使得创建的是子类的对象
        res = [
            class_init(
                X_df_train, Y_df_train, meta_df_train, **train_kwargs
            ),
            class_init(
                X_df_test, Y_df_test, meta_df_test, **test_kwargs
            )
        ]
        if valid_size is not None:
            res.insert(
                1,
                class_init(
                    X_df_valid, Y_df_valid, meta_df_valid, **valid_kwargs
                )
            )
        return tuple(res)

    def split_cv(
        self, cv, test_size, valid_size=None, random_seed=None,
        train_kwargs={}, test_kwargs={}, valid_kwargs={}
    ):
        stra_y = self.Y_df_trans.iloc[:, 0].values
        sss = StratifiedShuffleSplit(
            cv, test_size=test_size, random_state=random_seed)
        class_init = self.__class__
        for train_index, test_index in sss.split(
            self.X_df_trans.values, stra_y
        ):
            X_df_train = self.X_df_trans.iloc[train_index, :]
            X_df_test = self.X_df_trans.iloc[test_index, :]
            Y_df_train = self.Y_df_trans.iloc[train_index, :]
            Y_df_test = self.Y_df_trans.iloc[test_index, :]
            if self.meta_df is not None:
                meta_df_train = self.meta_df[train_index, :]
                meta_df_test = self.meta_df[test_index, :]
            else:
                meta_df_train, meta_df_test = None, None
            if valid_size is not None:
                stra_y2 = Y_df_train.iloc[:, 0].values
                split_dfs = [X_df_train, Y_df_train]
                if self.meta_df is not None:
                    split_dfs.append(meta_df_train)
                split_res = train_test_split(
                    *split_dfs, test_size=valid_size / (1 - test_size),
                    shuffle=True, stratify=stra_y2, random_state=random_seed
                )
                X_df_train, X_df_valid, Y_df_train, Y_df_valid = split_res[:4]
                if self.meta_df is not None:
                    meta_df_train, meta_df_valid = split_res[4:]
                else:
                    meta_df_train, meta_df_valid = None, None
            res = [
                class_init(
                    X_df_train, Y_df_train, meta_df_train, **train_kwargs
                ),
                class_init(
                    X_df_test, Y_df_test, meta_df_test, **test_kwargs
                )
            ]
            if valid_size is not None:
                res.insert(
                    1,
                    class_init(
                        X_df_valid, Y_df_valid, meta_df_valid, **valid_kwargs
                    )
                )
            yield tuple(res)


class MetaBatchEffect(BaseData):
    def __init__(self, X, Y, meta_df=None, pre_transfer=None):
        super(MetaBatchEffect, self).__init__(X, Y, meta_df, None, pre_transfer)

    @staticmethod
    def from_csv(sample_file, metabolic_file, y_names, pre_transfer=None):
        '''
        cli_file：临床数据文件；
        rna_file：RNAseq数据文件；
        y_names：使用的y的name，可以是多个，也可以是dict，则其key是原始变量名，
            value是更改后的变量名；
        kwargs：用于传递至RnaData的实例化方法的其他参数;
        '''
        # 获取标签
        sample_index = pd.read_csv(sample_file, index_col='sample.name')
        if isinstance(y_names, dict):
            y = sample_index[list(y_names.keys())].dropna().rename(
                columns=y_names)
        else:
            if not isinstance(y_names, (list, tuple)):
                y_names = [y_names]
            y = sample_index[y_names].dropna()
        y_num = y.shape[-1]

        # 获取代谢数据
        meta_df = pd.read_csv(
            metabolic_file, index_col='name').drop(['mz', 'rt'], axis=1)
        meta_df = meta_df.T.rename_axis(index='sample', columns='meta')
        # 进行merge
        all_df = y.merge(
            meta_df, how='inner', left_index=True, right_index=True)

        return MetaBatchEffect(
            all_df.iloc[:, y_num:], all_df.iloc[:, :y_num],
            pre_transfer=pre_transfer
        )

    @property
    def num_features(self):
        return self.X_df_trans.shape[1]

    def split_qc(self):
        qc_index = [i.startswith('QC') for i in self.X_df_trans.index]
        subject_index = [not i for i in qc_index]

        return (
            MetaBatchEffect(
                self.X_df_trans[subject_index], self.Y_df_trans[subject_index]
            ),
            MetaBatchEffect(
                self.X_df_trans[qc_index], self.Y_df_trans[qc_index]
            ),
        )


def test():
    import sys
    import time

    sample_file = "./DATA/metabolic/sample.information.T3.csv"
    meta_file = "./DATA/metabolic/data_T3原始数据.csv"
    t1 = time.perf_counter()
    data = MetaBatchEffect.from_csv(
        sample_file, meta_file, 'injection.order'
    )
    t2 = time.perf_counter()
    print(t2 - t1)
    subject_dat, qc_dat = data.split_qc()
    print(subject_dat.X_df_trans.head())
    print(subject_dat.Y_df_trans.head())
    print(qc_dat.X_df_trans.head())
    print(qc_dat.Y_df_trans.head())

    print(subject_dat[0])
    print(qc_dat[0])

    print(len(subject_dat))
    print(subject_dat.num_features)
    print(len(qc_dat))
    print(qc_dat.num_features)


if __name__ == "__main__":
    test()