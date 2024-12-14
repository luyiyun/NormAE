import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from normae import NormAE
from sklearn.decomposition import PCA


def get_data(log_transform: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_df = pd.read_csv("./data/OriData/Amide/meta.csv", index_col=0)
    # mz_rt_df = meta_df[["mz", "rt"]]
    meta_df = meta_df.drop(columns=["mz", "rt"]).T

    info_df = pd.read_csv(
        "./data/OriData/Amide/sample.information.csv", index_col=0
    )

    indice_inter = info_df.index.intersection(meta_df.index)
    meta_df = meta_df.loc[indice_inter]
    info_df = info_df.loc[indice_inter]

    # remove peaks that has most zero values in all samples
    mask1 = (meta_df == 0).mean(axis=0) < 0.2
    meta_df = meta_df.loc[:, mask1]
    # remove peaks that has most zero values in QCs
    qc_mask = info_df["class"] == "QC"
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
    info_df = info_df.loc[:, ["injection.order", "batch", "group", "class"]]
    # batch labels are transform to beginning from zero
    info_df.loc[:, "batch"] -= 1
    # digitize group
    info_df["group"] = info_df["group"].replace("QC", -1).astype("int")
    # digitize class
    # info_df.replace({"class": {"Subject": 1, "QC": 0}}, inplace=True)

    if log_transform:
        meta_df = meta_df.map(np.log)

    print(meta_df.shape, info_df.shape)

    return meta_df, info_df


def main():
    meta_df, info_df = get_data(log_transform=True)
    meta_df_subject = meta_df.loc[info_df["class"] == "Subject", :]
    info_df_subject = info_df.loc[info_df["class"] == "Subject", :]
    meta_df_qc = meta_df.loc[info_df["class"] == "QC", :]
    info_df_qc = info_df.loc[info_df["class"] == "QC", :]

    normae = NormAE(
        enc_hiddens=(300, 300),
        dec_hiddens=(300, 300),
        n_latents=100,
        disc_batch_hiddens=(250,),
        disc_order_hiddens=(250,),
        # lr_rec=0.01,
        # lr_disc_batch=0.01,
        # lr_disc_order=0.01,
        n_epochs_rec_pretrain=40,
        n_epochs_disc_pretrain=10,
        n_epochs_iter_train=100,
        early_stop=False,
        batch_size=128,
    )
    normae.fit(
        meta_df_subject.values,
        info_df_subject["batch"].values,
        info_df_subject["injection.order"].values,
        X_qc=meta_df_qc.values,
        y_qc=info_df_qc["batch"].values,
        z_qc=info_df_qc["injection.order"].values,
    )
    normae.plot_history("./normae_history.png")

    X_clean = normae.transform(meta_df.values)
    X_clean_pca = PCA(n_components=2).fit_transform(X_clean)
    batches = info_df["batch"].unique()
    palette = sns.color_palette()
    fig, ax = plt.subplots(figsize=(8, 8))
    for ci in ["Subject", "QC"]:
        for i, bi in enumerate(batches):
            mask = (info_df["class"] == ci) & (info_df["batch"] == bi)
            x, y = X_clean_pca[mask, 0], X_clean_pca[mask, 1]
            ax.plot(
                x,
                y,
                "o",
                color=palette[i],
                label=f"batch={bi}",
                alpha=0.2 if ci == "Subject" else 1,
            )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles[batches.shape[0] :],
        labels=labels[batches.shape[0] :],
        loc="best",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.savefig("./normae_pca.png")


if __name__ == "__main__":
    main()
