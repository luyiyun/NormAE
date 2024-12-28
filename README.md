> Now NormAE is a package.

# NormAE (Normalization Autoencoder)

It's a novel batch effects removal method based on deep autoencoder and adversarial leanring for metabolomics data. Additional classifier and ranker are trained to provide adversarial regularizations during training AE model, and latent representations are extracted by the encoder and then decoder reconstructs data without batch effects. The schematic diagram of NormAE is shown as follow.

![normAE](/imgs/graphics.png)

The NormAE method was tested in two real metabolomics datasets. We show the results of Amide dataset as follow.

![amide](/imgs/figure2.png)

<div align=center>
The results of Amide dataset. There are PCA score plots (A), heatmaps of the PCCs (B), intensity of peak M235T294 changing with injection order (C), the cumulative RSD curve of QCs (D), the number of differential peaks (E), average AUC values using same number of peaks (F), and AUC values using selected peaks based on features selection pipeline (G) before and after applying each batch effects removal method. Four colors circles refer to different batches. The solid and open circles refer to QCs and subject samples, respectively.
</div>

---

Paper: **NormAE: A Novel Deep Adversarial Learning Model to Remove Batch Effects in Liquid Chromatography Mass Spectrometry-Based Metabolomics Data**

---

## Table of Contents

- Detail informations
- Installation
- How to use
- Contact

### Detail informations

#### Running time

If using GPU (GTX 1080Ti), the training time on the Amide dataset (729 samples, 8113 peaks) is 58 minutes. If using CPU (1 core, Intel i7-8700k), the trainig time is 102 minutes. Correspondingly, the time taken by QC-RLSC using CPU is 125 minutes.

#### Hardware requirements

The CPU we used is Intel(R) Core(TM) i7-8700k CPU @ 3.70GHz. The memory occupied by the program is less than 1.1G. As shown above, the training efficiency will be improved by 50%-100% if using GPU.

#### Recommended sample size

We carried out experiments to explore the influence of sample sizes. We used Amide dataset and reduced the sample size to 80% (583), 60% (437), 40% (291), 20% (145), 10% (72), and 5% (36) of original sample size. The PCA score plot is shown below:

![samples](/imgs/pca_samples.png)

The figure above shows that NormAE is available for data with more than 150 samles. For data whose sample size is smaller than 150, QCs didn't cluster together in PCA score plot.

#### Number of QCs

NormAE dosen't need QCs. It removes batch effects throught batch labels and injection orders. But having dozens of QCs will help users to evaluate the model and optimize the hyperparameters. Our recommendation is more than 10 QCs.

#### Imput format

NormAE has no requirements for the input format. In the article, we used peak area without any transformation. We also performed experiments for logarithm transformed data. The PCA score plot of Amide dataset is show below:

![log](/imgs/pca_log.png)

The figure above shows that NormAE performed well for data after logarithm thansformation. It proves that NormAE is robust for the data format.

### Installation

The code is written in Python 3.10 and can be installed using the following command:

```bash
pip install git+https://github.com/luyiyun/NormAE.git
```

### How to use

#### Using NormAE like a normal python package

NormAE can be used like a normal python package. You can import the package and instantiate the `NormAE` and run the `fit` and `transform` methods to remove batch effects, like scikit-learn.

```python
from normae import NormAE

normae = NormAE()
normae.fit(X, y=batch_labels, z=injection_orders, X_qc=X_qc, y_qc=batch_labels_qc, z_qc=injection_orders_qc)
X_nobe = normae.transform(X)
```

#### Using NormAE in command line

After installing NormAE, you can use the command line to run the `normae` command. The command line interface provides a simple way to run the NormAE model.

1. The metabolomics data should be in the format of CSV file like `./example/example_x.csv`. The batch label, injection_orders and QC/non-QC labels should be in the format of CSV like `./example/example_y.csv`.

2. Run the following command to train the NormAE model:

  ```bash
  normae --meta_csv ./example/example_x.csv --sample_csv ./example/example_sample_info.csv --output_dir ./example/
  ```

3. The results will be saved in the output directory.

You can run `normae --help` to see the full list of options.

### Contact

For more information please contact Zhiwei Rong (18845728185@163.com)
