from sinf.SINF import *
from sinf.load_data import * 
import argparse
import time
from sinf.fid_score import evaluate_fid_score
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


def add_one_layer_inverse(model, data, sample, K, nsample_A, nsample, layer_type='regular', shape=None, kernel=None, shift=None, M=400, MSWD_max_iter=200, edge_bins=10, derivclip=1, extrapolate='regression', alpha=(0., 0.), noise_threshold=0, KDE=False, b_factor_data=1, b_factor_sample=1, batchsize=None, verbose=True, sample_test=None, put_data_on_disk=False, pool=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert layer_type in ['regular', 'patch']
    if layer_type == 'patch':
        assert shape is not None
        assert kernel is not None
        assert shift is not None
 
    if verbose:
        tstart = start_timing(device)

    t = time.time()
    if put_data_on_disk:
        assert isinstance(data, str) and isinstance(sample, str)
        if sample_test is not None:
            assert isinstance(sample_test, str)
        sample_address = sample
        data_address = data

        sample1 = np.load(sample_address)#np.lib.format.open_memmap(sample_address)
        perm_sample = np.random.permutation(len(sample1))
        sample1 = torch.as_tensor(sample1[perm_sample[:nsample]])
        
        data1 = np.load(data_address)#np.lib.format.open_memmap(data_address)
        perm_data = np.random.permutation(len(data1))
        data1 = preprocess(data1[perm_data[:nsample]])

    else:
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
    #print(time.time()-t, '1')

    if layer_type == 'regular':
        layer = SlicedTransport(ndim=model.ndim, K=K, M=M).requires_grad_(False).to(device)
    elif layer_type == 'patch':
        layer = PatchSlicedTransport(shape=shape, kernel=kernel, shift=shift, K=K, M=M).requires_grad_(False).to(device)

    layer.fit_A(data1, sample=sample1, ndata_A=nsample_A, MSWD_max_iter=MSWD_max_iter, pool=pool, verbose=verbose)

    del data1, sample1

    t = time.time()
    if put_data_on_disk:
        sample1 = np.load(sample_address)#np.lib.format.open_memmap(sample_address)
        sample1 = torch.as_tensor(sample1[perm_sample[-nsample:]])
        data1 = np.load(data_address)#np.lib.format.open_memmap(data_address)
        data1 = preprocess(data1[perm_data[-nsample:]])
    else:
        if nsample < len(sample):
            sample1 = sample[-nsample:]
        else:
            sample1 = sample
        if nsample < len(data):
            data1 = data[-nsample:]
        else:
            data1 = data
    #print(time.time()-t, '2')

    success = layer.fit_spline_inverse(data1, sample1, edge_bins=edge_bins, derivclip=derivclip, extrapolate=extrapolate, alpha=alpha, noise_threshold=noise_threshold,
                                       KDE=KDE, b_factor_data=b_factor_data, b_factor_sample=b_factor_sample, batchsize=batchsize, verbose=verbose)

    del data1, sample1
    if put_data_on_disk:
        del perm_sample, perm_data

    t = time.time()

    if success:
        if put_data_on_disk:
            if batchsize is None:
                sample = torch.as_tensor(np.load(sample_address))
                sample = layer.inverse(sample.to(device))[0].cpu().numpy()
                np.save(sample_address, sample)
                del sample
            else:
                sample = torch.as_tensor(np.load(sample_address))#np.lib.format.open_memmap(sample_address, mode='r+') 
                sample = transform_batch_layer(layer, sample, batchsize, direction='inverse', pool=pool)[0].numpy()
                np.save(sample_address, sample)
                del sample
                if sample_test is not None:
                    sample_test_address = sample_test
                    sample_test = torch.as_tensor(np.load(sample_test_address))#np.lib.format.open_memmap(sample_test_address, mode='r+') 
                    sample_test = transform_batch_layer(layer, sample_test, batchsize, direction='inverse', pool=pool)[0].numpy()
                    np.save(sample_test_address, sample_test)
                    del sample_test
                    sample_test = sample_test_address
            sample = sample_address

        else:
            if batchsize is None:
                sample = layer.inverse(sample.to(device))[0].to(sample.device)
                if sample_test is not None:
                    sample_test = layer.inverse(sample_test.to(device))[0].to(sample_test.device)
            else:
                sample = transform_batch_layer(layer, sample, batchsize, direction='inverse', pool=pool)[0]
                if sample_test is not None:
                    sample_test = transform_batch_layer(layer, sample_test, batchsize, direction='inverse', pool=pool)[0]

        model.add_layer(layer.cpu(), position=0)
    #print(time.time()-t, '3')
    if verbose:
        t = end_timing(tstart, device)
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
    
    parser.add_argument('--put_data_on_disk', type=str, 
                        help='Temporary address to save the data and samples. Only use this when the dataset is large and cannot load in the memory (e.g. CelebA).')
    
    parser.add_argument('--nohierarchy', action='store_true',
                        help='Whether to use hierarchical patch based modeling strategy.')
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            args.mp = True
        else:
            args.mp = False
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
        nsample_A = len(data_train) 
        nsample = 6*len(data_train)
    elif args.dataset == 'celeba':
        nsample_A = 40000 
        nsample = 100000 
    
    if args.put_data_on_disk:
        np.save(args.put_data_on_disk + args.dataset + '_data_train.npy', data_train)
        data_train = args.put_data_on_disk + args.dataset + '_data_train.npy'
    else:
        data_train = preprocess(data_train)
    
    if args.evaluateFID:
        FID = []
        data_test = data_test.reshape(-1, *shape)
        if args.put_data_on_disk:
            np.save(args.put_data_on_disk + args.dataset + '_data_test.npy', data_test)
            data_test = args.put_data_on_disk + args.dataset + '_data_test.npy'
    else:
        del data_test
    
    verbose = True
    update_iteration = 50
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
            sample = torch.randn(nsample, ndim)
            t = time.time()
            sample = transform_batch_model(model, sample, batchsize, start=None, end=0)[0]
            print ('Transform samples. time:', time.time()-t, 'iteration:', len(model.layer))
    else:
        sample = torch.randn(nsample, ndim)
        
    if args.put_data_on_disk:
        sample_address = args.put_data_on_disk + args.dataset + '_sample_train_temp' + '.npy'
        np.save(sample_address, sample)
        sample = sample_address 
    
    if args.evaluateFID:
        if args.restore:
            try:
                sample_test = torch.as_tensor(np.load(args.restore + '_sample_test.npy'))
            except:
                sample_test = torch.randn(10000, ndim)
                t = time.time()
                sample_test = transform_batch_model(model, sample_test, batchsize, start=None, end=0)[0]
                model = model.cpu()
                sample_test1 = toimage(sample_test, shape)
                if args.put_data_on_disk:
                    FID.append(evaluate_fid_score(sample_test1.astype(np.float32)/255., np.load(data_test)[:10000].astype(np.float32)/255.))
                else:
                    FID.append(evaluate_fid_score(sample_test1.astype(np.float32)/255., data_test.astype(np.float32)/255.))
                del sample_test1
                print ('Transform test samples. time:', time.time()-t, 'iteration:', len(model.layer), 'Current FID score:', FID[-1])
        else:
            sample_test = torch.randn(10000, ndim)
    
        if args.put_data_on_disk:
            sample_test_address = args.put_data_on_disk + args.dataset + '_sample_test_temp' + '.npy'
            np.save(sample_test_address, sample_test)
            sample_test = sample_test_address
    else:
        sample_test = None
    
    model = model.cpu()
    
    if args.put_data_on_disk:
        args.put_data_on_disk = True
    
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
                          [2,2,3],
                          [2,2,1]]
            K_factor3 = 2 #K = K_factor * patch_size[0]
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
            K_factor3 = 2 #K = K_factor * patch_size[0]
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
            K_factor1 = 2 #K = K_factor * patch_size[0]

        args.save = args.save + 'SIG_%s_seed%d_hierarchy' % (args.dataset, args.seed)
       
        for patch in patch_size:
            if patch[-1] == 3:
                Niter = 200
                K = K_factor3 * patch[0]
            elif patch[-1] == 1:
                Niter = 100
                K = K_factor1 * patch[0]
            #nsample_A = int(np.log10(np.prod(patch))*30000)
            for _ in range(Niter):
                nlayer += 1 
                if nlayer <= len(model.layer):
                    continue
                if patch[0] == shape[0] and patch[2] == shape[2]:
                    model, sample, sample_test = add_one_layer_inverse(model, data_train, sample, K, nsample_A, nsample, layer_type='regular', 
                                                                       batchsize=batchsize, sample_test=sample_test, put_data_on_disk=args.put_data_on_disk, pool=pool)
                else:
                    shift = torch.randint(shape[0], (2,)).tolist()
                    model, sample, sample_test = add_one_layer_inverse(model, data_train, sample, K, nsample_A, nsample, layer_type='patch', shape=shape, kernel=patch, 
                                                                       shift=shift, batchsize=batchsize, sample_test=sample_test, put_data_on_disk=args.put_data_on_disk, pool=pool)
                if len(model.layer) % update_iteration == 0:
                    print()
                    print('Finished %d iterations' % len(model.layer), 'Total Time:', time.time()-t_total)
                    print()
                    #save the model and the data 
                    torch.save(model, args.save)
                    if args.put_data_on_disk:
                        os.system('cp ' + sample + ' ' + args.save + '_sample_train.npy')
                    else:
                        np.save(args.save + '_sample_train.npy', sample.numpy())
                    if args.evaluateFID:
                        if args.put_data_on_disk:
                            sample_test1 = toimage(torch.tensor(np.load(sample_test)), shape)
                            FID.append(evaluate_fid_score(sample_test1.astype(np.float32)/255., np.load(data_test)[:10000].astype(np.float32)/255.))
                            os.system('cp ' + sample_test + ' ' + args.save + '_sample_test.npy')
                        else:  
                            sample_test1 = toimage(sample_test, shape)
                            FID.append(evaluate_fid_score(sample_test1.astype(np.float32)/255., data_test.astype(np.float32)/255.))
                            np.save(args.save + '_sample_test.npy', sample_test.numpy())
                        print('Current FID score:', FID[-1])
                        del sample_test1
        
    else:
        args.save + 'SIG_%s_seed%d' % (args.dataset, args.seed)
        if args.dataset == 'celeba':
            K = 128 
            Niter = 2700
        elif args.dataset == 'cifar10':
            K = 64
            Niter = 2500
        elif args.dataset in ['mnist', 'fmnist']:
            K = 56
            Niter = 800
        
        for _ in range(Niter):
            nlayer += 1
            if nlayer <= len(model.layer):
                continue
            model, sample, sample_test = add_one_layer_inverse(model, data_train, sample, K, nsample_A, nsample_spline, layer_type='regular', 
                                                               batchsize=batchsize, sample_test=sample_test, put_data_on_disk=args.put_data_on_disk)
            if len(model.layer) % update_iteration == 0:
                print()
                print('Finished %d iterations' % len(model.layer), 'Total Time:', time.time()-t_total)
                print()
                torch.save(model, args.save)
                if args.evaluateFID:
                    if args.put_data_on_disk:
                        sample_test1 = toimage(torch.tensor(np.load(sample_test)), shape)
                        FID.append(evaluate_fid_score(sample_test1.astype(np.float32)/255., np.load(data_test)[:10000].astype(np.float32)/255.))
                    else:  
                        sample_test1 = toimage(sample_test, shape)
                        FID.append(evaluate_fid_score(sample_test1.astype(np.float32)/255., data_test.astype(np.float32)/255.))
                    print('Current FID score:', FID[-1])
                    del sample_test1
    
    if pool:
        pool.close()
        pool.join()
    
    sys.exit(0)

