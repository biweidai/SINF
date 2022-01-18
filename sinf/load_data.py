import torch
import numpy as np
import torchvision
from torchvision import transforms as transforms_
import os
#import imageio
#from PIL import Image
import pandas as pd
from os.path import join
from collections import Counter
import h5py

def load_data_mnist():
    transform = transforms_.Compose([
        transforms_.ToTensor(),
        transforms_.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.MNIST(root="/global/scratch/biwei/data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root="/global/scratch/biwei/data", train=False, download=True, transform=transform)
    return train_data.data, test_data.data



def load_data_fmnist():
    transform = transforms_.Compose([
        transforms_.ToTensor(),
        transforms_.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.FashionMNIST(root="/global/scratch/biwei/data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.FashionMNIST(root="/global/scratch/biwei/data", train=False, download=True, transform=transform)
    return train_data.data, test_data.data



def load_data_cifar10():
    transform = transforms_.Compose([
        transforms_.ToTensor(),
        transforms_.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.CIFAR10(root="/global/scratch/biwei/data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root="/global/scratch/biwei/data", train=False, download=True, transform=transform)
    return train_data.data, test_data.data



def load_data_celeba(flag='training'):
    assert flag in ['training', 'validation', 'test']
    if flag == 'training':
        return np.load('/global/scratch/biwei/data/CelebA/CelebA_train.npy')
    elif flag == 'test':
        return np.load('/global/scratch/biwei/data/CelebA/CelebA_test.npy') 

'''
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
'''


def load_data_power():
    def load_data():
        return np.load('/global/scratch/biwei/data/power/data.npy')

    def load_data_split_with_noise():
    
        rng = np.random.RandomState(42)
    
        data = load_data()
        rng.shuffle(data)
        N = data.shape[0]
    
        data = np.delete(data, 3, axis=1)
        data = np.delete(data, 1, axis=1)
        ############################
        # Add noise
        ############################
        # global_intensity_noise = 0.1*rng.rand(N, 1)
        voltage_noise = 0.01*rng.rand(N, 1)
        # grp_noise = 0.001*rng.rand(N, 1)
        gap_noise = 0.001*rng.rand(N, 1)
        sm_noise = rng.rand(N, 3)
        time_noise = np.zeros((N, 1))
        # noise = np.hstack((gap_noise, grp_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
        # noise = np.hstack((gap_noise, grp_noise, voltage_noise, sm_noise, time_noise))
        noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
        data = data + noise
    
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
    
        return data_train, data_validate, data_test
    
    
    def load_data_normalised():
    
        data_train, data_validate, data_test = load_data_split_with_noise()
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu)/s
        data_validate = (data_validate - mu)/s
        data_test = (data_test - mu)/s
    
        return data_train, data_validate, data_test
    
    return load_data_normalised()



def load_data_gas():
    def load_data(file):
    
        data = pd.read_pickle(file)
        # data = pd.read_pickle(file).sample(frac=0.25)
        # data.to_pickle(file)
        data.drop("Meth", axis=1, inplace=True)
        data.drop("Eth", axis=1, inplace=True)
        data.drop("Time", axis=1, inplace=True)
        return data
    
    
    def get_correlation_numbers(data):
        C = data.corr()
        A = C > 0.98
        B = A.to_numpy().sum(axis=1)
        return B
    
    
    def load_data_and_clean(file):
    
        data = load_data(file)
        B = get_correlation_numbers(data)
    
        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            col_name = data.columns[col_to_remove]
            data.drop(col_name, axis=1, inplace=True)
            B = get_correlation_numbers(data)
        # print(data.corr())
        data = (data-data.mean())/data.std()
    
        return data
    
    
    def load_data_and_clean_and_split(file):
    
        data = load_data_and_clean(file).to_numpy()
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data_train = data[0:-N_test]
        N_validate = int(0.1*data_train.shape[0])
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]
    
        return data_train, data_validate, data_test

    return load_data_and_clean_and_split('/global/scratch/biwei/data/gas/ethylene_CO.pickle')    



def load_data_hepmass():
    def load_data(path):
    
        data_train = pd.read_csv(filepath_or_buffer=join(path, "1000_train.csv"), index_col=False)
        data_test = pd.read_csv(filepath_or_buffer=join(path, "1000_test.csv"), index_col=False)
    
        return data_train, data_test
    
    
    def load_data_no_discrete(path):
        """
        Loads the positive class examples from the first 10 percent of the dataset.
        """
        data_train, data_test = load_data(path)
    
        # Gets rid of any background noise examples i.e. class label 0.
        data_train = data_train[data_train[data_train.columns[0]] == 1]
        data_train = data_train.drop(data_train.columns[0], axis=1)
        data_test = data_test[data_test[data_test.columns[0]] == 1]
        data_test = data_test.drop(data_test.columns[0], axis=1)
        # Because the data set is messed up!
        data_test = data_test.drop(data_test.columns[-1], axis=1)
    
        return data_train, data_test
    
    
    def load_data_no_discrete_normalised(path):
    
        data_train, data_test = load_data_no_discrete(path)
        mu = data_train.mean()
        s = data_train.std()
        data_train = (data_train - mu)/s
        data_test = (data_test - mu)/s
    
        return data_train, data_test
    
    
    def load_data_no_discrete_normalised_as_array(path):
    
        data_train, data_test = load_data_no_discrete_normalised(path)
        data_train, data_test = data_train.to_numpy(), data_test.to_numpy()
    
        i = 0
        # Remove any features that have too many re-occurring real values.
        features_to_remove = []
        for feature in data_train.T:
            c = Counter(feature)
            max_count = np.array([v for k, v in sorted(c.items())])[0]
            if max_count > 5:
                features_to_remove.append(i)
            i += 1
        data_train = data_train[:, np.array([i for i in range(data_train.shape[1]) if i not in features_to_remove])]
        data_test = data_test[:, np.array([i for i in range(data_test.shape[1]) if i not in features_to_remove])]
    
        N = data_train.shape[0]
        N_validate = int(N*0.1)
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]
    
        return data_train, data_validate, data_test
    return load_data_no_discrete_normalised_as_array('/global/scratch/biwei/data/hepmass/') 



def load_data_miniboone():
    def load_data(root_path):
        # NOTE: To remember how the pre-processing was done.
        # data = pd.read_csv(root_path, names=[str(x) for x in range(50)], delim_whitespace=True)
        # print data.head()
        # data = data.as_matrix()
        # # Remove some random outliers
        # indices = (data[:, 0] < -100)
        # data = data[~indices]
        #
        # i = 0
        # # Remove any features that have too many re-occuring real values.
        # features_to_remove = []
        # for feature in data.T:
        #     c = Counter(feature)
        #     max_count = np.array([v for k, v in sorted(c.iteritems())])[0]
        #     if max_count > 5:
        #         features_to_remove.append(i)
        #     i += 1
        # data = data[:, np.array([i for i in range(data.shape[1]) if i not in features_to_remove])]
        # np.save("~/data/miniboone/data.npy", data)
    
        data = np.load(root_path)
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
    
        return data_train, data_validate, data_test
    
    def load_data_normalised(root_path):
    
        data_train, data_validate, data_test = load_data(root_path)
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu)/s
        data_validate = (data_validate - mu)/s
        data_test = (data_test - mu)/s
    
        return data_train, data_validate, data_test
    
    return load_data_normalised('/global/scratch/biwei/data/miniboone/data.npy')



def load_data_bsds300():
    f = h5py.File('/global/scratch/biwei/data/BSDS300/BSDS300.hdf5', 'r')
    return f['train'], f['validation'], f['test']
 
