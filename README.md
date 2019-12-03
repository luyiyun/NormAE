NormAE (Normalization Autoencoder)
=============================
It's a novel batch effects removal method based on deep autoencoder and adversarial leanring for metabolomics data.

***

Paper: NormAE: A Novel Deep Adversarial Learning Model to Remove Batch Effects in Liquid Chromatography Mass Spectrometry-Based Metabolomics Data

***

## Table of Contents
* Requirements
* How to use
* Contact

### Requirements

- [python >= 3.6.8](https://www.python.org)
- [pytorch >= 1.2.0](https://pytorch.org)
- [numpy >= 1.17.3](https://numpy.org)
- [pandas >= 0.25.3](https://pandas.pydata.org)
- [scipy >= 1.3.1](https://www.scipy.org)
- [sklearn >= 0.21.3](https://scikit-learn.org)
- [matplotlib >= 3.1.1](https://matplotlib.org)
- [visdom >= 0.1.8.8](https://www.github.com/facebookresearch/visdom)
- [argparse >= 1.1](https://docs.python.org/3/library/argparse.html)
- [tqdm >= 4.32.2](https://tqdm.github.io)

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

```
python main.py --task train --meta_data /path/to/metabolomics_data --sample_data /path/to/batch_information --save /path/to/save_dir
```

#### Remove batch effects

```
python main.py --task remove --meta_data /path/to/metabolomics_data --sample_data /path/to/batch_information --save /path/to/save_dir --load path/to/saved_model.pth
```

### Contact

For more information please contact Zhiwei Rong (18845728185@163.com)
