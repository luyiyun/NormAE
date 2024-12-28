import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_pca(
    X_clean: np.ndarray,
    qc: np.ndarray | None = None,
    batch: np.ndarray | None = None,
    label: np.ndarray | None = None,
    injection: np.ndarray | None = None,
) -> tuple[Figure, Axes]:
    X_clean_pca = PCA(n_components=2).fit_transform(X_clean)
    n_subfigs = (
        (qc is not None)
        + (batch is not None)
        + (label is not None)
        + (injection is not None)
    )
    nrows = 2 if n_subfigs > 3 else 1
    ncols = n_subfigs if n_subfigs <= 3 else n_subfigs // 2

    fig, axs = plt.subplots(figsize=(ncols * 4, nrows * 4), ncols=ncols, nrows=nrows)
    axs = axs.flatten()

    i = 0
    for c, cname in [
        (qc, "QC"),
        (batch, "Batch"),
        (label, "Label"),
        (injection, "Injection"),
    ]:
        if c is None:
            continue
        ax = axs[i]
        cu = np.unique(c)
        if cname != "Injection":
            for j, ci in enumerate(cu):
                mask = c == ci
                x, y = X_clean_pca[mask, 0], X_clean_pca[mask, 1]
                ax.plot(
                    x,
                    y,
                    ".",
                    color=sns.color_palette()[j],
                    label=f"{ci}",
                    markersize=3,
                )
            ax.legend(loc="best")
        else:
            sc = ax.scatter(
                X_clean_pca[:, 0],
                X_clean_pca[:, 1],
                c=injection,
                cmap="Blues",
                s=3,
            )

            fig.colorbar(sc, ax=ax)
        ax.set_title(f"{cname} distribution")

        i += 1

    fig.supxlabel("PC1")
    fig.supylabel("PC2")

    return fig, axs
