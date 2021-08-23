> There is a `visdom`-related bug that I will fix in the future.

NormAE (Normalization Autoencoder)
=============================
It's a novel batch effects removal method based on deep autoencoder and adversarial leanring for metabolomics data. Additional classifier and ranker are trained to provide adversarial regularizations during training AE model, and latent representations are extracted by the encoder and then decoder reconstructs data without batch effects. The schematic diagram of NormAE is shown as follow.

![normAE](/imgs/graphics.png)

The NormAE method was tested in two real metabolomics datasets. We show the results of Amide dataset as follow.

![amide](/imgs/figure2.png)

<div align=center>
The results of Amide dataset. There are PCA score plots (A), heatmaps of the PCCs (B), intensity of peak M235T294 changing with injection order (C), the cumulative RSD curve of QCs (D), the number of differential peaks (E), average AUC values using same number of peaks (F), and AUC values using selected peaks based on features selection pipeline (G) before and after applying each batch effects removal method. Four colors circles refer to different batches. The solid and open circles refer to QCs and subject samples, respectively.
</div>

***

Paper: **NormAE: A Novel Deep Adversarial Learning Model to Remove Batch Effects in Liquid Chromatography Mass Spectrometry-Based Metabolomics Data**

***

## Table of Contents
* Detail informations
* Requirements
* How to use
* Contact

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

### Requirements

- [python == 3.6.8](https://www.python.org)
- [pytorch == 1.2.0](https://pytorch.org)
- [numpy == 1.17.3](https://numpy.org)
- [pandas == 0.25.3](https://pandas.pydata.org)
- [scipy == 1.3.1](https://www.scipy.org)
- [sklearn == 0.21.3](https://scikit-learn.org)
- [matplotlib == 3.1.1](https://matplotlib.org)
- [visdom == 0.1.8.8](https://www.github.com/facebookresearch/visdom)
- [argparse == 1.1](https://docs.python.org/3/library/argparse.html)
- [tqdm == 4.32.2](https://tqdm.github.io)

### How to use

#### Data preparation

metabolomics_data:

```
name,mz,rt,QC1,A1,A2,A3,QC2,A4\n
M64T32,64,32,1000,2000,3000,4000,5000,6000\n
M65T33,65,33,10000,20000,30000,40000,50000,60000\n
...
```

batch_information:

```
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
```

#### Training

if you need the visualization of visdom, you can open visdom server firstly.

```bash
visdom --port 8097
```
Then you can perform the script to training and save trained model in `/path/to/save_dir`.

```bash
python main.py --task train --meta_data /path/to/metabolomics_data --sample_data /path/to/batch_information --save /path/to/save_dir
```

In `/path/to/save_dir`, there will be some files:

- models.pth ==> saved model
- train.csv ==> recorded losses for all training epochs
- config.json ==> saved configuration
- early_stop_info.json ==> saved training times and other

#### Remove batch effects

Finally, you can perform the script with "remove mode" to remove batch effects using saved model. The `path/to/save_dir/Rec_nobe.csv` is the data with out batch effects.

```
python main.py --task remove --meta_data /path/to/metabolomics_data --sample_data /path/to/batch_information --save /path/to/save_dir --load path/to/saved_model.pth
```

There are some new files that will be added in `/path/to/saved_dir`:

- `Ori.csv`, `Ys.csv` ==> original dataset
- `Rec.csv` ==> the reconstruction of original data with batch effects
- `Codes.csv` ==> the values of bottle neck layers
- `Rec_nobe.csv` ==> the reconstruction of original data without batch effects

#### Other

There are many other parameters to control the configuration of NormAE. Using `python main.py --help` will get the help for them.

```
usage: main.py [-h] [--task TASK] [--meta_data META_DATA]
               [--sample_data SAMPLE_DATA] [-td TRAIN_DATA] [-s SAVE]
               [--ae_encoder_units AE_ENCODER_UNITS [AE_ENCODER_UNITS ...]]
               [--ae_decoder_units AE_DECODER_UNITS [AE_DECODER_UNITS ...]]
               [--disc_b_units DISC_B_UNITS [DISC_B_UNITS ...]]
               [--disc_o_units DISC_O_UNITS [DISC_O_UNITS ...]]
               [--bottle_num BOTTLE_NUM]
               [--dropouts DROPOUTS DROPOUTS DROPOUTS DROPOUTS]
               [--lambda_b LAMBDA_B] [--lambda_o LAMBDA_O] [--lr_rec LR_REC]
               [--lr_disc_b LR_DISC_B] [--lr_disc_o LR_DISC_O]
               [-e EPOCH EPOCH EPOCH]
               [--use_batch_for_order USE_BATCH_FOR_ORDER] [-bs BATCH_SIZE]
               [--load LOAD] [--visdom_env VISDOM_ENV]
               [--visdom_port VISDOM_PORT] [-nw NUM_WORKERS] [--use_log]
               [--use_batch USE_BATCH] [--sample_size SAMPLE_SIZE]
               [--random_seed RANDOM_SEED] [--device {None,CPU,GPU}]

optional arguments:
  -h, --help            show this help message and exit
  --task TASK           task, train model (train, default) or remove batch
                        effects (remove)
  --meta_data META_DATA
                        the path of metabolomics data
  --sample_data SAMPLE_DATA
                        the path of sample information
  -td TRAIN_DATA, --train_data TRAIN_DATA
                        the training data, subject or all (default)
  -s SAVE, --save SAVE  the path to save results, default ./save
  --ae_encoder_units AE_ENCODER_UNITS [AE_ENCODER_UNITS ...]
                        the hidden units of encoder, default 1000, 1000
  --ae_decoder_units AE_DECODER_UNITS [AE_DECODER_UNITS ...]
                        the hidden units of decoder, default 1000, 1000
  --disc_b_units DISC_B_UNITS [DISC_B_UNITS ...]
                        the hidden units of disc_b, default 250, 250
  --disc_o_units DISC_O_UNITS [DISC_O_UNITS ...]
                        the hidden units of disc_b, default 250, 250
  --bottle_num BOTTLE_NUM
                        the number of bottle neck units, default 500
  --dropouts DROPOUTS DROPOUTS DROPOUTS DROPOUTS
                        the dropout rates of encoder, decoder, disc_b,
                        disc_o,default 0.3, 0.1, 0.3, 0.3
  --lambda_b LAMBDA_B   the weight of adversarial loss for batch labels,
                        default 1
  --lambda_o LAMBDA_O   the weight of adversarial loss for injection order,
                        default 1
  --lr_rec LR_REC       the learning rate of AE training, default 0.0002
  --lr_disc_b LR_DISC_B
                        the leanring rate of disc_b training, default 0.005
  --lr_disc_o LR_DISC_O
                        the leanring rate of disc_o training, default 0.0005
  -e EPOCH EPOCH EPOCH, --epoch EPOCH EPOCH EPOCH
                        ae pretrain, disc pretrain, iteration train
                        epochs，default (1000, 10, 700)
  --use_batch_for_order USE_BATCH_FOR_ORDER
                        if compute rank loss with batch ?, default True
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size，default 64
  --load LOAD           load trained models, default None
  --visdom_env VISDOM_ENV
                        if use visdom, it is the env name,default main
  --visdom_port VISDOM_PORT
                        if use visdom, it is the port, default 8097
  -nw NUM_WORKERS, --num_workers NUM_WORKERS
                        the number of multi cores, default 12
  --use_log             use logrithm?
  --use_batch USE_BATCH
                        use part of batches? default None
  --sample_size SAMPLE_SIZE
                        use size of part of samples? default None
  --random_seed RANDOM_SEED
                        random seed, default 1234.
  --device {None,CPU,GPU}
                        device
```

### Contact

For more information please contact Zhiwei Rong (18845728185@163.com)
