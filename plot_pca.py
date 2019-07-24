import os

import argparse
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dir', help="想要预测的模型，指的是训练完保存的文件夹"
    )
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.dir, 'result_data.csv'), index_col=0)
    df_ori = df[df['index'] == 'ori'].drop('index', axis=1)
    df_res = df[df['index'] == 'res'].drop('index', axis=1)

    qc_index = [i.startswith('QC') for i in df_ori.index]
    sub_index = [not i for i in qc_index]

    pca = PCA(2)
    pca_ori = pca.fit_transform(df_ori.values)
    pca_res = pca.fit_transform(df_res.values)

    _, axes = plt.subplots(ncols=2)
    for i, (label, dat) in enumerate(
        zip(['origin', 'handled'], [pca_ori, pca_res])
    ):
        ax = axes[i]
        ax.scatter(
            dat[sub_index][:, 0], dat[sub_index][:, 1], color='gray',
            label='subject'
        )
        ax.scatter(
            dat[qc_index][:, 0], dat[qc_index][:, 1], color='r', label='QC'
        )
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.legend()
        ax.set_title(label)

    plt.show()


if __name__ == '__main__':
    main()
