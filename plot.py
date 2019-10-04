import os

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from visual import pca_for_dict, pca_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('save', help="储存结果的文件夹")
    parser.add_argument(
        '--ica', action='store_true', help='是否使用ica处理后的数据')
    parser.add_argument(
        '--to', default='evaluation_ml_res',
        help='保存评价结果的json文件名，默认是evaluation_ml_res')
    parser.add_argument('--rand_seed', default=1234, type=int)
    args = parser.parse_args()
    print(args)
    print('')

    task_path = args.save
    # ----- 读取数据集 -----
    data_names = ['original_x', 'recons_no_batch', 'recons_all', 'ys']
    all_res = {}
    for dn in data_names:
        if dn == 'recons_no_batch' and args.ica:
            file_name = dn + '_ica'
        else:
            file_name = dn
        all_res[dn] = pd.read_csv(
            os.path.join(task_path, 'all_res_%s.csv' % file_name), index_col=0)

    # ----- PCA -----
    sub_pca, qc_pca = pca_for_dict(all_res)
    pca_plot(sub_pca, qc_pca)
    plt.show()


if __name__ == '__main__':
    main()
