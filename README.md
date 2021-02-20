# Sliced Iterative Normalizing Flows

This repository is the official implementation of [Sliced Iterative Normalizing Flows](https://arxiv.org/abs/2007.00674). 

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

The FID score of SIG samples can also be evaluated on the fly with the "--evaluateFID" argument. 


## Results

SIG achieves the following performance on :

### FID score

|       MNIST       |      Fashion      |      CIFAR10      |       CelebA      |
| ----------------- | ----------------- | ----------------- | ----------------- |
|        4.5        |       13.7        |       66.5        |       37.3        |

### OoD detection (AUROC on models trained on FashionMNIST)

|       MNIST       |      OMNIGLOT     |    FMNIST-hflip   |   FMNIST-vflip    |
| ----------------- | ----------------- | ----------------- | ----------------- |
|       0.990       |       0.993       |       0.631       |       0.821       |

 
