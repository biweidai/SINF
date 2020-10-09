# Sliced Iterative Generator (SIG) & Gaussianizing Iterative Slicing (GIS)

This repository is the official implementation of [Sliced Iterative Generator](https://arxiv.org/abs/2007.00674). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train SIG and GIS, run the following commands:

```train
python SIG.py --dataset DATASET --seed SEED --save SAVING_ADDRESS   
python GIS.py --dataset DATASET --seed SEED --save SAVING_ADDRESS
```

## Evaluation

All the evaluation codes of SIG are in SIG.ipynb.
The FID score of SIG samples can also be evaluated on the fly with the "--evaluateFID" argument. 


## Results

SIG achieves the following performance on :

### FID score

|       MNIST       |      Fashion      |      CIFAR10      |  CelebA (SIG+AE)  |
| ----------------- | ----------------- | ----------------- | ----------------- |
|        5.5        |       16.0        |   71.6 (updated)  |       48.1        |

### OoD detection (AUROC on models trained on FashionMNIST)

|       MNIST       |      OMNIGLOT     |    FMNIST-hflip   |   FMNIST-vflip    |
| ----------------- | ----------------- | ----------------- | ----------------- |
|       0.977       |       0.990       |       0.636       |       0.815       |

The codes to reproduce these results are in SIG.ipynb
 
