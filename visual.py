import copy

import numpy as np
import visdom
import torch
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use('Pdf')
import matplotlib.pyplot as plt


class VisObj:
    def __init__(self, pca_plot=True):
        self.epoch_idx = {}
        self.batch_losses = {}
        self.epoch_losses = {}
        self.vis = visdom.Visdom()
        self.pca_plot = pca_plot
        if pca_plot:
            self.pca = PCA(2)

    def add_batch_loss(self, reset=True, **loss_dict):
        if reset:
            self.batch_idx = 0
            update = None
        else:
            self.batch_idx += 1
            update = 'append'
        ks, vs = self._dict2tensor(**loss_dict)
        self.vis.line(
            X=torch.tensor([self.batch_idx]),
            Y=vs, update=update, win='batch_losses',
            opts={'title': "Batch Loss", 'legend': ks}
        )

    def add_epoch_loss(self, winname='epoch_losses', **loss_dict):
        if winname not in self.epoch_idx.keys():
            update = None
            self.epoch_idx[winname] = 1
        else:
            update = 'append'
            self.epoch_idx[winname] += 1
        ks, vs = self._dict2tensor(**loss_dict)
        self.vis.line(
            X=torch.tensor([self.epoch_idx[winname]]),
            Y=vs, update=update, win=winname,
            opts={'title': winname, 'legend': ks}
        )

    def plot_pca(self, **datas):
        assert self.pca_plot
        self.pca.fit_transform()


    @staticmethod
    def _dict2tensor(**loss_dict):
        ks = []
        vs = []
        for k, v in loss_dict.items():
            ks.append(k)
            vs.append(v)
        vs = torch.tensor(vs).squeeze().unsqueeze(0)
        return ks, vs


def pca_plot(
    x_ori_plot_pca, x_recons_nobe_pca, x_recons_be_pca, ys, subject_num
):
    qc_num = len(x_ori_plot_pca) - subject_num
    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))

    # Subject points of Original datas, reconstructed datas containing batch
    #   effects or no batch effects
    ax = axs[0, 0]
    ax.scatter(
        x_ori_plot_pca[:subject_num, 0], x_ori_plot_pca[:subject_num, 1],
        c='r', label='Original X'
    )
    ax.scatter(
        x_recons_be_pca[:subject_num, 0], x_recons_be_pca[:subject_num, 1],
        c='b', label='Reconstructed X with BE'
    )
    ax.scatter(
        x_recons_nobe_pca[:subject_num, 0], x_recons_nobe_pca[:subject_num, 1],
        c='g', label='Reconstructed X without BE'
    )
    ax.set_title('Subject points')
    ax.legend()

    # QC points of Original datas, reconstructed datas containing batch
    #   effects or no batch effects
    ax = axs[0, 1]
    ax.scatter(
        x_ori_plot_pca[subject_num:, 0], x_ori_plot_pca[subject_num:, 1],
        c='r', label='Original X'
    )
    ax.scatter(
        x_recons_be_pca[subject_num:, 0], x_recons_be_pca[subject_num:, 1],
        c='b', label='Reconstructed X with BE'
    )
    ax.scatter(
        x_recons_nobe_pca[subject_num:, 0], x_recons_nobe_pca[subject_num:, 1],
        c='g', label='Reconstructed X without BE'
    )
    ax.set_title('QC points')
    ax.legend()

    # reconstructed datas without batch effects of Subject points and QC points
    # Batch Label
    if ys.shape[1] > 1:
        ax = axs[1, 0]
        ax.scatter(
            x_recons_nobe_pca[:subject_num, 0],
            x_recons_nobe_pca[:subject_num, 1],
            c=ys[:subject_num, 1], marker='o', label='Subject', cmap='jet'
        )
        scatter = ax.scatter(
            x_recons_nobe_pca[subject_num:, 0],
            x_recons_nobe_pca[subject_num:, 1],
            c=ys[subject_num:, 1], marker='^', label='QC', cmap='jet'
        )
        ax.set_title('Subject vs QC without BE')
        legend1 = ax.legend(
            *scatter.legend_elements(), loc="lower left", title='Batch')
        ax.add_artist(legend1)
        ax.legend()

    # reconstructed datas without batch effects of Subject points and QC points
    # injection order
    ax = axs[1, 1]
    ax.scatter(
        x_recons_nobe_pca[:subject_num, 0], x_recons_nobe_pca[:subject_num, 1],
        s=ys[:subject_num, 0], c='r', label='Subject', alpha=0.5
    )
    scatter = ax.scatter(
        x_recons_nobe_pca[subject_num:, 0], x_recons_nobe_pca[subject_num:, 1],
        s=ys[subject_num:, 0], c='b', label='QC', alpha=0.5
    )
    ax.set_title('Subject vs QC without BE')
    handles, labels = scatter.legend_elements(prop='sizes', alpha=0.6, num=5)
    legend2 = ax.legend(
        handles, labels, loc='lower left', title="Injection Order")
    ax.add_artist(legend2)
    ax.legend()



def test():
    a = np.random.randn(100, 2)
    b = np.random.randn(100, 2)
    c = np.random.randn(100, 2)
    ys = np.stack(
        [np.arange(100), np.random.randint(1, 5, size=(100,))], axis=1)
    subject_num = 50
    pca_plot(a, b, c, ys, subject_num)
    vis = visdom.Visdom()
    vis.matplot(plt, win='test2', opts={'title': 'test2'})
    plt.close()


if __name__ == '__main__':
    test()
