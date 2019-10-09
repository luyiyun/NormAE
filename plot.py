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
        '--pca', action='store_true', help="是否绘制PCA图"
    )
    parser.add_argument('--meta_scatter', nargs='*',
                        #  default=['M221T336', 'M122T352',])
                        default=['M124T609', 'M317T294', 'M538T586'])
    parser.add_argument('--no_plot', action='store_false')
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
    if args.pca:
        sub_pca, qc_pca = pca_for_dict(all_res)
        pca_plot(sub_pca, qc_pca)
        if args.no_plot:
            plt.show()

    # ----- single metabolite scatter -----
    meta_len = len(args.meta_scatter)
    if meta_len > 0:
        fig, axes = plt.subplots(
            ncols=2, nrows=meta_len, figsize=(20, 5*meta_len), squeeze=False)
        for i in range(meta_len):
            meta_df = pd.concat([
                all_res['original_x'][[args.meta_scatter[i]]],
                all_res['ys'][['injection.order', 'class']],
            ], axis=1)
            meta_df.plot.scatter(
                x="injection.order", y=args.meta_scatter[i], c="class",
                ax=axes[i, 0], cmap=plt.get_cmap('jet'), colorbar=False,
                title='Original'
            )
            meta_df = pd.concat([
                all_res['recons_no_batch'][[args.meta_scatter[i]]],
                all_res['ys'][['injection.order', 'class']],
            ], axis=1)
            meta_df.plot.scatter(
                x="injection.order", y=args.meta_scatter[i], c="class",
                ax=axes[i, 1], cmap=plt.get_cmap('jet'), colorbar=False,
                title='BEAE'
            )
        if args.no_plot:
            plt.show()


if __name__ == '__main__':
    main()
