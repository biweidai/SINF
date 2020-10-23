from SIT import *
from load_data import * 
import argparse
import time
from fid_score import evaluate_fid_score

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


def add_one_layer_inverse(model, data, sample, n_component, nsample_wT, nsample_spline, layer_type='regular', shape=None, kernel=None, shift=None, interp_nbin=400, MSWD_p=2, MSWD_max_iter=200, edge_bins=10, derivclip=1, extrapolate='regression', alpha=(0., 0.), noise_threshold=0, KDE=False, bw_factor_data=1, bw_factor_sample=1, batchsize=None, verbose=True, sample_test=None, put_data_on_disk=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert layer_type in ['regular', 'patch']
    if layer_type == 'patch':
        assert shape is not None
        assert kernel is not None
        assert shift is not None
 
    if verbose:
        tstart = start_timing()

    t = time.time()
    if put_data_on_disk:
        assert isinstance(data, str) and isinstance(sample, str)
        if sample_test is not None:
            assert isinstance(sample_test, str)
        sample_address = sample
        data_address = data

        sample1 = np.load(sample_address)#np.lib.format.open_memmap(sample_address)
        perm_sample = np.random.permutation(len(sample1))
        #sample1 = torch.tensor(sample1[np.sort(perm_sample[:nsample_wT])]).to(device)
        sample1 = torch.tensor(sample1[perm_sample[:nsample_wT]]).to(device)
        
        data1 = np.load(data_address)#np.lib.format.open_memmap(data_address)
        perm_data = np.random.permutation(len(data1))
        #data1 = preprocess(data1[np.sort(perm_data[:nsample_wT])]).to(device)
        data1 = preprocess(data1[perm_data[:nsample_wT]]).to(device)

    else:
        assert isinstance(data, torch.Tensor) and isinstance(sample, torch.Tensor)
        sample = sample[torch.randperm(sample.shape[0], device=sample.device)]
        sample1 = sample[:nsample_wT].to(device)
        
        data = data[torch.randperm(data.shape[0], device=data.device)]
        data1 = data[:nsample_wT].to(device)
    print(time.time()-t)

    assert len(sample1) == len(data1)

    if layer_type == 'regular':
        layer = SlicedTransport(ndim=model.ndim, n_component=n_component, interp_nbin=interp_nbin).requires_grad_(False).to(device)
    elif layer_type == 'patch':
        layer = PatchSlicedTransport(shape=shape, kernel=kernel, shift=shift, n_component=n_component, interp_nbin=interp_nbin).requires_grad_(False).to(device)

    layer.fit_wT(data1, sample=sample1, MSWD_p=MSWD_p, MSWD_max_iter=MSWD_max_iter, verbose=verbose)

    del data1, sample1

    t = time.time()
    if put_data_on_disk:
        sample1 = np.load(sample_address)#np.lib.format.open_memmap(sample_address)
        #sample1 = torch.tensor(sample1[np.sort(perm_sample[-nsample_spline:])]).to(device)
        sample1 = torch.tensor(sample1[perm_sample[-nsample_spline:]]).to(device)
        data1 = np.load(data_address)#np.lib.format.open_memmap(data_address)
        #data1 = preprocess(data1[np.sort(perm_data[-nsample_spline:])]).to(device)
        data1 = preprocess(data1[perm_data[-nsample_spline:]]).to(device)
    else:
        sample1 = sample[-nsample_spline:].to(device)
        data1 = data[-nsample_spline:].to(device)
    print(time.time()-t)

    SWD = layer.fit_spline_inverse(data1, sample1, edge_bins=edge_bins, derivclip=derivclip, extrapolate=extrapolate, alpha=alpha, noise_threshold=noise_threshold,
                                   MSWD_p=MSWD_p, KDE=KDE, bw_factor_data=bw_factor_data, bw_factor_sample=bw_factor_sample, batchsize=batchsize, verbose=verbose)

    del data1, sample1
    if put_data_on_disk:
        del perm_sample, perm_data

    t = time.time()
    if (SWD>noise_threshold).any():

        if put_data_on_disk:
            if batchsize is None:
                sample = torch.tensor(np.load(sample_address))
                sample = layer.inverse(sample.to(device))[0].cpu().numpy()
                np.save(sample_address, sample)
                del sample
            else:
                sample = np.load(sample_address)#np.lib.format.open_memmap(sample_address, mode='r+') 
                i = 0
                while i * batchsize < len(sample):
                    sample[i*batchsize:(i+1)*batchsize] = layer.inverse(torch.tensor(sample[i*batchsize:(i+1)*batchsize]).to(device))[0].cpu().numpy()
                    i += 1
                np.save(sample_address, sample)
                del sample
                if sample_test is not None:
                    sample_test_address = sample_test
                    sample_test = np.load(sample_test_address)#np.lib.format.open_memmap(sample_test_address, mode='r+') 
                    i = 0
                    while i * batchsize < len(sample_test):
                        sample_test[i*batchsize:(i+1)*batchsize] = layer.inverse(torch.tensor(sample_test[i*batchsize:(i+1)*batchsize]).to(device))[0].cpu().numpy()
                        i += 1
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
                sample = transform_batch_layer(layer, sample, batchsize, direction='inverse')[0]
                if sample_test is not None:
                    sample_test = transform_batch_layer(layer, sample_test, batchsize, direction='inverse')[0]

        model.add_layer(layer.cpu(), position=0)
    print(time.time()-t)
    if verbose:
        t = end_timing(tstart)
        print ('Nlayer:', len(model.layer), 'Time:', t, layer_type)
        print ()

    return model, sample, sample_test



parser = argparse.ArgumentParser()

# data
parser.add_argument('--dataset', type=str, default='mnist',
                    choices=['mnist', 'fmnist', 'cifar10', 'celeba'],
                    help='Name of dataset to use.')

parser.add_argument('--evaluateFID', action='store_true',
                    help='Whether to evaluate FID scores between random samples and test data.')

parser.add_argument('--seed', type=int, default=738,
                    help='Random seed for PyTorch and NumPy.')

parser.add_argument('--save', type=str, default='./',
                    help='Where to save the trained model.')

parser.add_argument('--put_data_on_disk', type=str, 
                    help='Temporary address to save the data and samples. Only use this when the dataset is large and cannot load in the memory (e.g. CelebA).')

parser.add_argument('--nohierarchy', action='store_true',
                    help='Whether to use hierarchical patch based modeling strategy.')

args = parser.parse_args()

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
elif args.dataset == 'fmnist':
    data_train, data_test = load_data_fmnist()
    shape = [28,28,1]
elif args.dataset == 'cifar10':
    data_train, data_test = load_data_cifar10()
    shape = [32,32,3]
elif args.dataset == 'celeba':
    data_train = load_data_celeba(flag='training')
    data_test = load_data_celeba(flag='test')
    shape = [64,64,3]
    if not args.put_data_on_disk:
        args.put_data_on_disk = './'

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

if args.dataset in ['mnist', 'fmnist', 'cifar10']:
    nsample_wT = len(data_train) 
    nsample_spline = 5*len(data_train)
    nsample = nsample_wT + nsample_spline
elif args.dataset == 'celeba':
    nsample_wT = 40000 
    nsample_spline = 60000 
    nsample = 100000 

batchsize = 10000 #Avoid running out of memory. Do not affect performance
verbose = True
update_iteration = 100
ndim = shape[0]*shape[1]*shape[2]

t_total = time.time()

#define the model
model = SIT(ndim=ndim).requires_grad_(False)
#model = torch.load(args.save + 'SIG_celeba_seed738_hierarchy') 


sample = torch.randn(nsample, ndim)
if args.put_data_on_disk:
    jobid = str(int(time.time()))
    sample_address = args.put_data_on_disk + args.dataset + '_sample_' + jobid + '.npy'
    np.save(sample_address, sample)
    sample = sample_address 

if args.evaluateFID:
    sample_test = torch.randn(10000, ndim)
    if args.put_data_on_disk:
        sample_test_address = args.put_data_on_disk + args.dataset + '_sample_test_' + jobid + '.npy'
        np.save(sample_test_address, sample_test)
        sample_test = sample_test_address
else:
    sample_test = None

#sample = args.put_data_on_disk + args.dataset + '_sample_1602694511.npy'
#sample_test = args.put_data_on_disk + args.dataset + '_sample_test_1602694511.npy'

if args.put_data_on_disk:
    args.put_data_on_disk = True

if not args.nohierarchy:
    if args.dataset == 'celeba':
        patch_size = [[64,64,3], 
                      [32,32,3],
                      [16,16,3],
                      [8,8,3],
                      [7,7,3],
                      [6,6,3],
                      [5,5,3],
                      [4,4,3],
                      [3,3,3],
                      [2,2,3]]
        K_factor = 2 #n_component = K_factor * patch_size[0]
        Niter = 200 #number of iterations for each patch_size[0]
    elif args.dataset == 'cifar10':
        patch_size = [[32,32,3], 
                      [16,16,3],
                      [8,8,3],
                      [7,7,3],
                      [6,6,3],
                      [5,5,3],
                      [4,4,3],
                      [3,3,3],
                      [2,2,3]]
        K_factor = 2 #n_component = K_factor * patch_size[0]
        Niter = 200 #number of iterations for each patch_size[0]
    elif args.dataset in ['mnist', 'fmnist']:
        patch_size = [[28,28,1], 
                      [14,14,1],
                      [7,7,1],
                      [6,6,1],
                      [5,5,1],
                      [4,4,1],
                      [3,3,1],
                      [2,2,1]]
        K_factor = 2 #n_component = K_factor * patch_size[0]
        Niter = 100 #number of iterations for each patch_size[0]
   
    nlayer = 0 
    for patch in patch_size:
        n_component = K_factor * patch[0]
        for _ in range(Niter):
            nlayer += 1 
            if nlayer <= len(model.layer):
                continue
            if patch[0] == shape[0]:
                model, sample, sample_test = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='regular', 
                                                                   batchsize=batchsize, sample_test=sample_test, put_data_on_disk=args.put_data_on_disk)
            else:
                shift = torch.randint(shape[0], (2,)).tolist()
                model, sample, sample_test = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='patch', shape=shape, 
                                                                   kernel=patch, shift=shift, batchsize=batchsize, sample_test=sample_test, put_data_on_disk=args.put_data_on_disk)
            if len(model.layer) % update_iteration == 0:
                print()
                print('Finished %d iterations' % len(model.layer), 'Total Time:', time.time()-t_total)
                print()
                torch.save(model, args.save + 'SIG_%s_seed%d_hierarchy' % (args.dataset, args.seed))
                if args.evaluateFID:
                    if args.put_data_on_disk:
                        sample_test1 = toimage(torch.tensor(np.load(sample_test)), shape)
                        FID.append(evaluate_fid_score(sample_test1.astype(np.float32)/255., np.load(data_test)[:10000].astype(np.float32)/255.))
                    else:  
                        sample_test1 = toimage(sample_test, shape)
                        FID.append(evaluate_fid_score(sample_test1.astype(np.float32)/255., data_test.astype(np.float32)/255.))
                    print('Current FID score:', FID[-1])
                    del sample_test1
    
else:
    if args.dataset == 'cifar10':
        n_component = 64
        Niter = 2000
    elif args.dataset in ['mnist', 'fmnist']:
        n_component = 56
        Niter = 800
    
    for _ in range(Niter):
        model, sample, sample_test = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='regular', 
                                                           batchsize=batchsize, sample_test=sample_test, put_data_on_disk=args.put_data_on_disk)
        if len(model.layer) % update_iteration == 0:
            print()
            print('Finished %d iterations' % len(model.layer), 'Total Time:', time.time()-t_total)
            print()
            torch.save(model, args.save + 'SIG_%s_seed%d' % (args.dataset, args.seed))
            if args.evaluateFID:
                if args.put_data_on_disk:
                    sample_test1 = toimage(torch.tensor(np.load(sample_test)), shape)
                    FID.append(evaluate_fid_score(sample_test1.astype(np.float32)/255., np.load(data_test)[:10000].astype(np.float32)/255.))
                else:  
                    sample_test1 = toimage(sample_test, shape)
                    FID.append(evaluate_fid_score(sample_test1.astype(np.float32)/255., data_test.astype(np.float32)/255.))
                print('Current FID score:', FID[-1])
                del sample_test1

