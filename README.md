# Sliced Iterative Generator

This repository is the official implementation of Sliced Iterative Generator. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the models in the paper on different datasets, run the following commands:

```train
python SIGtrain_MNIST_hierarchy.py
python SIGtrain_FashionMNIST_hierarchy.py
python SIGtrain_CIFAR10_hierarchy.py
python SIGtrain_CelebA_hierarchy.py
python SIGtrain_FashionMNIST.py
python SIGtrain_CelebA_AE64.py
```

## Evaluation

All the evaluation codes are in SIG.ipynb


## Pre-trained Models

You can download pretrained models [here](https://drive.google.com/mymodel.pth)


## Results

Our model achieves the following performance on :

### FID score

|       MNIST       |      Fashion      |      CIFAR10      |  CelebA (SIG+AE)  |
| ----------------- | ----------------- | ----------------- | ----------------- |
|        5.5        |       16.0        |       91.7        |       48.1        |

### OoD detection (AUROC on models trained on FashionMNIST)

|       MNIST       |      OMNIGLOT     |    FMNIST-hflip   |   FMNIST-vflip    |
| ----------------- | ----------------- | ----------------- | ----------------- |
|       0.977       |       0.990       |       0.636       |       0.815       |

The codes to reproduce these results are in SIG.ipynb
 
