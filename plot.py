import os

import matplotlib.pyplot as plt
import argparse
import numpy as np
from sklearn.decomposition import PCA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('save')
    parser.add_argument('--ica', action='store_true')
    args = parser.parse_args()

    task_path = args.save
    # ----- 读取数据集 -----
    data_names = ['original_x', 'recons_no_batch']
    qc_res = {}
    for dn in data_names:
        if dn == 'recons_no_batch' and args.ica:
            file_name = dn + '_ica'
        else:
            file_name = dn
        qc_res[dn] = np.loadtxt(
            os.path.join(task_path, 'qc_res_%s.txt' % file_name))

    # ----- PCA -----
    pca = PCA(2)
    ori_x_pca = pca.fit_transform(qc_res['original_x'])
    nobe_x_pca = pca.fit_transform(qc_res['recons_no_batch'])

    # ----- plot -----
    _, ax = plt.subplots()
    ax.scatter(ori_x_pca[:, 0], ori_x_pca[:, 1], c='r', label='Original')
    ax.scatter(nobe_x_pca[:, 0], nobe_x_pca[:, 1], c='b', label='No Batch')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
