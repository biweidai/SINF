import torch
import numpy as np
import torchvision
from torchvision import transforms as transforms
import os
import imageio
from PIL import Image

def load_data_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train_data.data, test_data.data


def load_data_fmnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    return train_data.data, test_data.data


def load_data_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return train_data.data, test_data.data


def load_data_celeba(flag='training', side_length=None, num=None):
    data_dir = "./data"
    dir_path = os.path.join(data_dir, 'img_align_celeba')
    filelist = [filename for filename in os.listdir(dir_path) if filename.endswith('jpg')]
    #ID = [int(filename[:6]) for filename in filelist]
    #index = numpy.argsort(ID)
    #filelist = np.array(filelist)[index]
    #filelist = [filename for filename in filelist]
    
    assert len(filelist) == 202599
    if flag == 'training':
        start_idx, end_idx = 0, 162770
    elif flag == 'val':
        start_idx, end_idx = 162770, 182637
    else:
        start_idx, end_idx = 182637, 202599

    imgs = []
    for i in range(start_idx, end_idx):
        img = np.array(imageio.imread(dir_path + os.sep + filelist[i]))
        img = img[45:173,25:153]
        if side_length is not None:
            img = Image.fromarray(img)
            img = np.asarray(img.resize([side_length, side_length]))
        new_side_length = np.shape(img)[1]
        img = np.reshape(img, [1, new_side_length, new_side_length, 3])
        imgs.append(img)
        if num is not None and len(imgs) >= num:
            break
        if len(imgs) % 5000 == 0:
            print('Processing {} images...'.format(len(imgs)))
    imgs = np.concatenate(imgs, 0)

    return imgs.astype(np.uint8) 
