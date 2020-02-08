import numpy as np
import pandas as pd
import torch.utils.data as data
from sklearn.model_selection import train_test_split


class BaseData(data.Dataset):
    ''' Base Data Class '''
    def __init__(self, X_df, Y_df, pre_transfer=None):
        '''
        X_df: samples x peakes, dataframe;
        Y_df：samples x 4, the colnames are injection.order, batch, group and
        class, group is the representation for CRC(1) and CE(0), class is the
        representation for Subject(1) and QCs(0), -1 represeents None.
        '''
        super(BaseData, self).__init__()
        self.x_df, self.y_df = X_df, Y_df
        self.pre_transfer = pre_transfer
        if self.pre_transfer is not None:
            self.x_df, self.y_df = self.pre_transfer(self.x_df, self.y_df)

    def __len__(self):
        return len(self.x_df)

    def __getitem__(self, indx):
        sample_x, sample_y = self.x_df.values[indx], self.y_df.values[indx]
        return sample_x, sample_y

    def transform(self, trans):
        ''' transform X and Y '''
        self.x_df, self.x_df = trans(self.x_df, self.x_df)
        return self

    @property
    def num_features(self):
        ''' the number of peaks '''
        return self.x_df.shape[1]

    @property
    def num_batch_labels(self):
        ''' the number of batches '''
        return len(self.y_df['batch'].unique())


class ConcatData(BaseData):
    ''' concatenate two BaseData objects '''
    def __init__(self, *datas):
        x_dfs = pd.concat([d.x_df for d in datas], axis=0)
        y_dfs = pd.concat([d.y_df for d in datas], axis=0)
        super(ConcatData, self).__init__(x_dfs, y_dfs, None)


def get_metabolic_data(
    x_file, y_file, pre_transfer=None, sub_qc_split=True, use_log=False,
    use_batch=None, use_samples_size=None, random_seed=None
):
    '''
    Read metabolic data file and get dataframes
    metabolic data (x_file) example:
        name,mz,rt,QC1,A1,A2,A3,QC2,A4\n
        M64T32,64,32,1000,2000,3000,4000,5000,6000\n
        M65T33,65,33,10000,20000,30000,40000,50000,60000\n
        ...
    sample information data (y_file) example:
        sample.name,injection.order,batch,group,class\n
        QC1,1,1,QC,QC\n
        A1,2,1,0,Subject\n
        A2,3,1,1,Subject\n
        A3,4,1,1,Subject\n
        QC2,5,2,QC,QC\n
        A4,6,2,0,Subject\n
        A5,7,2,1,Subject\n
        A6,8,2,1,Subject\n
        ...
    '''
    # read y_file
    y_df = pd.read_csv(y_file, index_col='sample.name')
    y_df = y_df.dropna()
    y_num = y_df.shape[-1]
    # read x_file
    meta_df = pd.read_csv(x_file, index_col='name').drop(['mz', 'rt'], axis=1)
    meta_df = meta_df.T.rename_axis(index='sample', columns='meta')
    # merge
    all_df = y_df.merge(meta_df,
                        how='inner',
                        left_index=True,
                        right_index=True)

    # remove peaks that has most zero values in all samples
    meta_df, y_df = all_df.iloc[:, y_num:], all_df.iloc[:, :y_num]
    mask1 = (meta_df == 0).mean(axis=0) < 0.2
    meta_df = meta_df.loc[:, mask1]
    # remove peaks that has most zero values in QCs
    qc_mask = y_df['class'] == 'QC'
    qc_meta_df = meta_df.loc[qc_mask, :]
    mask2 = (qc_meta_df == 0).mean(axis=0) < 0.2
    meta_df = meta_df.loc[:, mask2]

    # for each peak, impute the zero values with the half of minimum values
    def impute_zero(peak):
        zero_mask = peak == 0
        if zero_mask.any():
            new_x = peak.copy()
            impute_value = peak.loc[~zero_mask].min()
            new_x[zero_mask] = impute_value / 2
            return new_x
        return peak

    meta_df = meta_df.apply(impute_zero, axis=0)

    # extract the useful information from y_file
    y_df = y_df.loc[:, ['injection.order', 'batch', 'group', 'class']]
    # batch labels are transform to beginning from zero
    y_df.loc[:, 'batch'] -= 1
    # digitize group
    y_df['group'].replace('QC', '-1', inplace=True)
    y_df['group'] = y_df['group'].astype('int')
    # digitize class
    y_df['class'].replace({'Subject': 1, 'QC': 0}, inplace=True)
    # inverse injection.order
    # y_df['injection.order'] = y_df['injection.order'].max(
    # ) - y_df['injection.order']

    if use_batch is not None:
        bool_ind = (y_df.loc[:, "batch"] < use_batch).values
        meta_df, y_df = meta_df.loc[bool_ind, :], y_df.loc[bool_ind, :]
    if use_samples_size is not None:
        meta_df, _, y_df, _ = train_test_split(
            meta_df, y_df, train_size=use_samples_size,
            stratify=y_df.loc[:, "batch"].values,
            random_state=random_seed
        )
    if use_log:
        meta_df = meta_df.applymap(np.log)
    if pre_transfer is not None:
        meta_df, y_df = pre_transfer(meta_df, y_df)
    if sub_qc_split:
        qc_index = y_df['class'] == 0
        return BaseData(meta_df[~qc_index], y_df[~qc_index]), \
            BaseData(meta_df[qc_index], y_df[qc_index])
    return BaseData(meta_df, y_df)


if __name__ == "__main__":
    # for testing
    meta_file = "./DATA/Amide/meta.csv"
    sample_file = "./DATA/Amide/sample.information.csv"
    subject_dat, qc_dat = get_metabolic_data(meta_file, sample_file)
    print('')

    print(subject_dat.x_df.head())
    print(subject_dat.x_df.head())
    print(qc_dat.x_df.head())
    print(qc_dat.x_df.head())

    print(subject_dat[0])
    print(qc_dat[0])

    print(len(subject_dat))
    print(subject_dat.num_features)
    print(len(qc_dat))
    print(qc_dat.num_features)
    print(qc_dat.num_batch_labels)
