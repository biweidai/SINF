from SINF import *
from load_data import * 
import argparse
import time
from fid_score import evaluate_fid_score
import torch.multiprocessing as mp
import sys
import os

def preprocess(data):
    data = torch.tensor(data).float().reshape(len(data), -1)
    data = (data + torch.rand_like(data)) / 128. - 1
    return data

def toimage(sample, shape):
    sample = (sample+1)*128
    sample[sample<0] = 0
    sample[sample>255] = 255
    sample = sample.cpu().numpy().astype('uint8').reshape(len(sample), *shape)
    return sample


def add_one_layer_inverse(model, data, sample, n_component, nsample_wT, nsample, layer_type='regular', shape=None, kernel=None, shift=None, interp_nbin=200, MSWD_max_iter=200, edge_bins=5, derivclip=1, extrapolate='regression', alpha=(0., 0.), noise_threshold=0, KDE=False, bw_factor_data=1, bw_factor_sample=1, batchsize=None, verbose=True, sample_test=None, pool=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert layer_type in ['regular', 'patch']
    if layer_type == 'patch':
        assert shape is not None
        assert kernel is not None
        assert shift is not None
 
    if verbose:
        tstart = start_timing()

    t = time.time()
    assert isinstance(data, torch.Tensor) and isinstance(sample, torch.Tensor)
    if nsample < len(sample):
        sample = sample[torch.randperm(sample.shape[0], device=sample.device)]
        sample1 = sample[:nsample]
    else:
        sample1 = sample
    if nsample < len(data):
        data = data[torch.randperm(data.shape[0], device=data.device)]
        data1 = data[:nsample]
    else:
        data1 = data

    if layer_type == 'regular':
        layer = SlicedTransport(ndim=model.ndim, n_component=n_component, interp_nbin=interp_nbin).requires_grad_(False).to(device)
    elif layer_type == 'patch':
        layer = PatchSlicedTransport(shape=shape, kernel=kernel, shift=shift, n_component=n_component, interp_nbin=interp_nbin).requires_grad_(False).to(device)

    layer.fit_wT(data1, sample=sample1, ndata_wT=nsample_wT, MSWD_max_iter=MSWD_max_iter, pool=pool, verbose=verbose)

    del data1, sample1

    t = time.time()
    if nsample < len(sample):
        sample1 = sample[-nsample:]
    else:
        sample1 = sample
    if nsample < len(data):
        data1 = data[-nsample:]
    else:
        data1 = data

    success = layer.fit_spline_inverse(data1, sample1, edge_bins=edge_bins, derivclip=derivclip, extrapolate=extrapolate, alpha=alpha, noise_threshold=noise_threshold,
                                       KDE=KDE, bw_factor_data=bw_factor_data, bw_factor_sample=bw_factor_sample, batchsize=batchsize, verbose=verbose)

    del data1, sample1

    t = time.time()

    if success:
        if batchsize is None:
            sample = layer.inverse(sample.to(device))[0].to(sample.device)
            if sample_test is not None:
                sample_test = layer.inverse(sample_test.to(device))[0].to(sample_test.device)
        else:
            sample = transform_batch_layer(layer, sample, batchsize, direction='inverse', pool=pool)[0]
            if sample_test is not None:
                sample_test = transform_batch_layer(layer, sample_test, batchsize, direction='inverse', pool=pool)[0]

        model.add_layer(layer.cpu(), position=0)
    if verbose:
        t = end_timing(tstart)
        print ('Nlayer:', len(model.layer), 'Time:', t, layer_type)
        print ()

    return model, sample, sample_test



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fmnist', 'cifar10', 'celeba'],
                        help='Name of dataset to use.')
    
    parser.add_argument('--evaluateFID', action='store_true',
                        help='Whether to evaluate FID scores between random samples and test data.')
    
    parser.add_argument('--mp', action='store_true',
                        help='Whether to use multiprocessing. Automatically enabled if there are multiple available GPUs. Automatically disabled if there is only one available GPU.')
    
    parser.add_argument('--seed', type=int, default=738,
                        help='Random seed for PyTorch and NumPy.')
    
    parser.add_argument('--save', type=str, default='./',
                        help='Where to save the trained model.')
    
    parser.add_argument('--restore', type=str, help='Path to model to restore.')

    parser.add_argument('--improvesample', type=str, help='Path to load the samples to be improved.')

    parser.add_argument('--improvesampletest', type=str, help='Path to load the test samples to be improved.')
    
    parser.add_argument('--nohierarchy', action='store_true',
                        help='Whether to use hierarchical patch based modeling strategy.')
    
    args = parser.parse_args()
    
    if args.mp:
        mp.set_start_method('spawn', force=True)
        if torch.cuda.is_available():
            nprocess = torch.cuda.device_count()
        else:
            nprocess = mp.cpu_count()  
        pool = mp.Pool(processes=nprocess)
        print('Number of processes:', nprocess)
    else:
        pool = None
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    
    if args.dataset == 'mnist':
        data_train, data_test = load_data_mnist()
        shape = [28,28,1]
        batchsize = 10000 #Avoid running out of memory. Do not affect performance
    elif args.dataset == 'fmnist':
        data_train, data_test = load_data_fmnist()
        shape = [28,28,1]
        batchsize = 10000 
    elif args.dataset == 'cifar10':
        data_train, data_test = load_data_cifar10()
        shape = [32,32,3]
        batchsize = 3000
    elif args.dataset == 'celeba':
        data_train = load_data_celeba(flag='training')
        data_test = load_data_celeba(flag='test')
        shape = [64,64,3]
        batchsize = 5000 
    
    if args.dataset in ['mnist', 'fmnist', 'cifar10']:
        nsample_wT = len(data_train) 
        nsample = 6*len(data_train)
    elif args.dataset == 'celeba':
        nsample_wT = 40000 
        nsample = 100000 
    
    data_train = preprocess(data_train)
    
    if args.evaluateFID:
        FID = []
        data_test = data_test.reshape(-1, *shape)
    else:
        del data_test
    
    verbose = True
    update_iteration = 100
    ndim = shape[0]*shape[1]*shape[2]
    
    t_total = time.time()
    
    #define the model
    if args.restore:
        model = torch.load(args.restore) 
        print('Successfully load in the model. Time:', time.time()-t_total)
    else:
        model = SINF(ndim=ndim).requires_grad_(False)
    
    if args.restore:
        try:
            sample = torch.as_tensor(np.load(args.restore + '_sample_train.npy'))
        except:
            if args.improvesample:
                sample = preprocess(np.load(args.improvesample))
                nsample = len(sample)
                if nsample_wT > nsample:
                    nsample_wT = nsample
            else:
                sample = torch.randn(nsample, ndim)
            t = time.time()
            sample = transform_batch_model(model, sample, batchsize, start=None, end=0)[0]
            print ('Transform samples. time:', time.time()-t, 'iteration:', len(model.layer))
    else:
        if args.improvesample:
            sample = preprocess(np.load(args.improvesample))
            nsample = len(sample)
            if nsample_wT > nsample:
                nsample_wT = nsample
        else:
            sample = torch.randn(nsample, ndim)
        
    if args.evaluateFID:
        if args.restore:
            try:
                sample_test = torch.as_tensor(np.load(args.restore + '_sample_test.npy'))
            except:
                if args.improvesampletest:
                    sample_test = preprocess(np.load(args.improvesampletest))
                else:
                    sample_test = torch.randn(10000, ndim)
                t = time.time()
                sample_test = transform_batch_model(model, sample_test, batchsize, start=None, end=0)[0]
                model = model.cpu()
                sample_test1 = toimage(sample_test, shape)
                FID.append(evaluate_fid_score(sample_test1.astype(np.float32)/255., data_test.astype(np.float32)/255.))
                del sample_test1
                print ('Transform test samples. time:', time.time()-t, 'iteration:', len(model.layer), 'Current FID score:', FID[-1])
        else:
            if args.improvesampletest:
                sample_test = preprocess(np.load(args.improvesampletest))
            else:
                sample_test = torch.randn(10000, ndim)
    
    else:
        sample_test = None
    
    model = model.cpu()
    
    
    nlayer = 0 
    if not args.nohierarchy:
        if args.dataset == 'celeba':
            patch_size = [[64,64,3], 
                          [32,32,3],
                          [16,16,3],
                          [8,8,3],
                          [8,8,1],
                          [7,7,3],
                          [7,7,1],
                          [6,6,3],
                          [6,6,1],
                          [5,5,3],
                          [5,5,1],
                          [4,4,3],
                          [4,4,1],
                          [3,3,3],
                          [3,3,1],
                          [2,2,3]]
            K_factor3 = 2 #n_component = K_factor * patch_size[0]
            K_factor1 = 1 
        elif args.dataset == 'cifar10':
            patch_size = [[32,32,3], 
                          [16,16,3],
                          [8,8,3],
                          [8,8,1],
                          [7,7,3],
                          [7,7,1],
                          [6,6,3],
                          [6,6,1],
                          [5,5,3],
                          [5,5,1],
                          [4,4,3],
                          [4,4,1],
                          [3,3,3],
                          [3,3,1],
                          [2,2,3],
                          [2,2,1]]
            K_factor3 = 2 #n_component = K_factor * patch_size[0]
            K_factor1 = 1 
        elif args.dataset in ['mnist', 'fmnist']:
            patch_size = [[28,28,1], 
                          [14,14,1],
                          [7,7,1],
                          [6,6,1],
                          [5,5,1],
                          [4,4,1],
                          [3,3,1],
                          [2,2,1]]
            K_factor1 = 2 #n_component = K_factor * patch_size[0]

        if args.improvesample:
            args.save = args.save + 'SIG_%s_seed%d_hierarchy_improve' % (args.dataset, args.seed)
        else:
            args.save = args.save + 'SIG_%s_seed%d_hierarchy' % (args.dataset, args.seed)
       
        for patch in patch_size:
            if patch[-1] == 3:
                Niter = 200
                if args.improvesample:
                    Niter = 60  
                n_component = K_factor3 * patch[0]
            elif patch[-1] == 1:
                Niter = 100
                if args.improvesample:
                    Niter = 30  
                n_component = K_factor1 * patch[0]
            for _ in range(Niter):
                nlayer += 1 
                if nlayer <= len(model.layer):
                    continue
                if patch[0] == shape[0] and patch[2] == shape[2]:
                    model, sample, sample_test = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample, layer_type='regular', 
                                                                       batchsize=batchsize, sample_test=sample_test, pool=pool)
                else:
                    shift = torch.randint(shape[0], (2,)).tolist()
                    model, sample, sample_test = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample, layer_type='patch', shape=shape, kernel=patch, 
                                                                       shift=shift, batchsize=batchsize, sample_test=sample_test, pool=pool)
                if len(model.layer) % update_iteration == 0:
                    print()
                    print('Finished %d iterations' % len(model.layer), 'Total Time:', time.time()-t_total)
                    print()
                    #save the model and the data 
                    torch.save(model, args.save)
                    np.save(args.save + '_sample_train.npy', sample.numpy())
                    np.save(args.save + '_sample_test.npy', sample_test.numpy())
                    if args.evaluateFID:
                        sample_test1 = toimage(sample_test, shape)
                        FID.append(evaluate_fid_score(sample_test1.astype(np.float32)/255., data_test.astype(np.float32)/255.))
                        print('Current FID score:', FID[-1])
                        del sample_test1
        
    else:
        args.save = args.save + 'SIG_%s_seed%d' % (args.dataset, args.seed)
        if args.dataset == 'celeba':
            n_component = 128 
            Niter = 2700
            if args.improvesample:
                Niter = 810  
        elif args.dataset == 'cifar10':
            n_component = 64
            Niter = 2500
            if args.improvesample:
                Niter = 750  
        elif args.dataset in ['mnist', 'fmnist']:
            n_component = 56 
            Niter = 800
            if args.improvesample:
                Niter = 240  
        
        for _ in range(Niter):
            nlayer += 1
            if nlayer <= len(model.layer):
                continue
            model, sample, sample_test = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample, layer_type='regular', 
                                                               batchsize=batchsize, sample_test=sample_test)
            if len(model.layer) % update_iteration == 0:
                print()
                print('Finished %d iterations' % len(model.layer), 'Total Time:', time.time()-t_total)
                print()
                torch.save(model, args.save)
                np.save(args.save + '_sample_train.npy', sample.numpy())
                np.save(args.save + '_sample_test.npy', sample_test.numpy())
                if args.evaluateFID:
                    sample_test1 = toimage(sample_test, shape)
                    FID.append(evaluate_fid_score(sample_test1.astype(np.float32)/255., data_test.astype(np.float32)/255.))
                    print('Current FID score:', FID[-1])
                    del sample_test1
    
    torch.save(model, args.save)
    np.save(args.save + '_sample_train.npy', sample.numpy())
    np.save(args.save + '_sample_test.npy', sample_test.numpy())
    if pool:
        pool.close()
        pool.join()
    
    sys.exit(0)

