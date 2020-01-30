import numpy as np
import visdom
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class VisObj:
    def __init__(self, port=8097, env='main'):
        self.epoch_idx = {}
        self.batch_losses = {}
        self.epoch_losses = {}
        self.vis = visdom.Visdom(port=port, env=env)
        self.pca_plot = pca_plot

    def add_epoch_loss(self, winname='epoch_losses', **loss_dict):
        if winname not in self.epoch_idx.keys():
            update = None
            self.epoch_idx[winname] = 1
        else:
            update = 'append'
            self.epoch_idx[winname] += 1
        ks, vs = self._dict2array(**loss_dict)
        self.vis.line(
            X=np.array([self.epoch_idx[winname]]),
            Y=vs, update=update, win=winname,
            opts={'title': winname, 'legend': ks}
        )

    @staticmethod
    def _dict2array(**loss_dict):
        ks = []
        vs = []
        for k, v in loss_dict.items():
            ks.append(k)
            vs.append(v)
        vs = np.expand_dims(np.array(vs).squeeze(), 0)
        return ks, vs


def pca_for_dict(all_dict, n_components=2, sub_qc_split=True):
    # results are dataframes
    ss = StandardScaler()
    pca = PCA(n_components)
    pca_dict = {}
    for k, v in all_dict.items():
        if k in ['Rec_nobe', 'Rec', 'Ori']:
            temp = ss.fit_transform(v.values)
            pca_dict[k] = pca.fit_transform(temp)
        elif k == 'Ys':
            pca_dict[k] = v.values
    if sub_qc_split:
        qc_index = pca_dict['Ys'][:, -1] == 0
        sub_pca_dict = {k: v[~qc_index, :] for k, v in pca_dict.items()}
        qc_pca_dict = {k: v[qc_index, :] for k, v in pca_dict.items()}
        return sub_pca_dict, qc_pca_dict
    return pca_dict


def pca_plot(subject_pca, qc_pca):
    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))

    # Subject points of Original datas, reconstructed datas containing batch
    #   effects or no batch effects
    ax = axs[0, 0]
    ax.scatter(
        subject_pca['Ori'][:, 0], subject_pca['Ori'][:, 1],
        c='r', label='Original X', alpha=0.5
    )
    ax.scatter(
        subject_pca['Rec'][:, 0], subject_pca['Rec'][:, 1],
        c='b', label='Reconstructed X with BE', alpha=0.5
    )
    ax.scatter(
        subject_pca['Rec_nobe'][:, 0], subject_pca['Rec_nobe'][:, 1],
        c='g', label='Reconstructed X without BE', alpha=0.5
    )
    ax.set_title('Subject points')
    ax.legend()

    # QC points of Original datas, reconstructed datas containing batch
    #   effects or no batch effects
    ax = axs[0, 1]
    ax.scatter(
        qc_pca['Ori'][:, 0], qc_pca['Ori'][:, 1],
        c='r', label='Original X', alpha=0.5
    )
    ax.scatter(
        qc_pca['Rec'][:, 0], qc_pca['Rec'][:, 1],
        c='b', label='Reconstructed X with BE', alpha=0.5
    )
    ax.scatter(
        qc_pca['Rec_nobe'][:, 0], qc_pca['Rec_nobe'][:, 1],
        c='g', label='Reconstructed X without BE', alpha=0.5
    )
    ax.set_title('QC points')
    ax.legend()

    # reconstructed datas without batch effects of Subject points and QC points
    # Batch Label
    plot_index = np.any(subject_pca['Ys'][:, 1] != -1)
    if plot_index:
        ax = axs[1, 0]
        ax.scatter(
            subject_pca['Rec_nobe'][:, 0], subject_pca['Rec_nobe'][:, 1],
            c=subject_pca['Ys'][:, 1], label='Subject', alpha=0.1
        )
        scatter = ax.scatter(
            qc_pca['Rec_nobe'][:, 0], qc_pca['Rec_nobe'][:, 1],
            c=qc_pca['Ys'][:, 1], label='QC', alpha=1.0
        )
        ax.set_title('Subject vs QC without BE under batch group')
        legend1 = ax.legend(
            *scatter.legend_elements(), loc="lower left", title='Batch')
        ax.add_artist(legend1)
        ax.legend()

    # reconstructed datas without batch effects of Subject points and QC points
    # injection order
    plot_index = np.any(subject_pca['Ys'][:, 2] != -1)
    if plot_index:
        ax = axs[1, 1]
        ax.scatter(
            qc_pca['Rec_nobe'][:, 0], qc_pca['Rec_nobe'][:, 1],
            c='r', label='QC', alpha=0.5
        )
        scatter = ax.scatter(
            subject_pca['Rec_nobe'][:, 0], subject_pca['Rec_nobe'][:, 1],
            c=subject_pca['Ys'][:, 2], label='Subject', alpha=0.5
        )
        ax.set_title('Subject vs QC without BE under predicted class')
        handles, labels = scatter.legend_elements()
        legend2 = ax.legend(
            *scatter.legend_elements(), loc="lower left", title='label')
        ax.add_artist(legend2)
        ax.legend()
