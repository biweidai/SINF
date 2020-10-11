from SIT import *
from load_data import * 
import argparse
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


def add_one_layer_inverse(model, data, sample, n_component, nsample_wT, nsample_spline, layer_type='regular', shape=None, kernel=None, shift=None, interp_nbin=400, MSWD_p=2, MSWD_max_iter=200, edge_bins=10, derivclip=1, extrapolate='regression', alpha=(0., 0.), noise_threshold=0, KDE=False, bw_factor_data=1, bw_factor_sample=1, batchsize=None, verbose=True, device=torch.device('cuda'), sample_test=None):

    assert layer_type in ['regular', 'patch']
    if layer_type == 'patch':
        assert shape is not None
        assert kernel is not None
        assert shift is not None
 
    assert nsample_wT <= len(data)
    assert len(sample) >= nsample_wT 
    assert len(sample) >= nsample_spline 

    if verbose:
        tstart = start_timing()

    sample_device = sample.device
    sample = sample[torch.randperm(sample.shape[0])]
    if layer_type == 'regular':
        layer = SlicedTransport(ndim=model.ndim, n_component=n_component, interp_nbin=interp_nbin).requires_grad_(False).to(device)
    elif layer_type == 'patch':
        layer = PatchSlicedTransport(shape=shape, kernel=kernel, shift=shift, n_component=n_component, interp_nbin=interp_nbin).requires_grad_(False).to(device)

    if len(data) == nsample_wT:
        data1 = data.to(device)
    else:
        data = data[torch.randperm(data.shape[0])]
        data1 = data[:nsample_wT].to(device)
    sample1 = sample[:nsample_wT].to(device)

    layer.fit_wT(data1, sample=sample1, MSWD_p=MSWD_p, MSWD_max_iter=MSWD_max_iter, verbose=verbose)

    if len(data) <= nsample_spline:
        data1 = data.to(device)
    else:
        data1 = data[-nsample_spline:].to(device)
    sample1 = sample[-nsample_spline:].to(device)
    SWD = layer.fit_spline_inverse(data1, sample1, edge_bins=edge_bins, derivclip=derivclip, extrapolate=extrapolate, alpha=alpha, noise_threshold=noise_threshold,
                                   MSWD_p=MSWD_p, KDE=KDE, bw_factor_data=bw_factor_data, bw_factor_sample=bw_factor_sample, batchsize=batchsize, verbose=verbose)
    del data1, sample1

    if (SWD>noise_threshold).any():

        if batchsize is None:
            sample = layer.inverse(sample.to(device))[0].to(sample_device)
            sample_test = layer.inverse(sample_test.to(device))[0].to(sample_device)
        else:
            j = 0
            while j * batchsize < len(sample):
                sample[j*batchsize:(j+1)*batchsize] = layer.inverse(sample[j*batchsize:(j+1)*batchsize].to(device))[0].to(sample_device)
                j += 1
            if sample_test is not None:
                j = 0
                while j * batchsize < len(sample_test):
                    sample_test[j*batchsize:(j+1)*batchsize] = layer.inverse(sample_test[j*batchsize:(j+1)*batchsize].to(device))[0].to(sample_device)
                    j += 1

        model.add_layer(layer.to(sample_device), position=0)
    if verbose:
        t = end_timing(tstart)
        print ('Nlayer:', len(model.layer), 'Time:', t, layer_type)
        print ()

    return model, sample, sample_test



parser = argparse.ArgumentParser()

# data
parser.add_argument('--dataset', type=str, default='mnist',
                    choices=['mnist', 'fmnist', 'cifar10'],
                    help='Name of dataset to use.')

parser.add_argument('--evaluateFID', action='store_true',
                    help='Whether to evaluate FID scores between random samples and test data.')

parser.add_argument('--seed', type=int, default=738,
                    help='Random seed for PyTorch and NumPy.')

parser.add_argument('--save', type=str, default='./',
                    help='Where to save the trained model.')

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
    nsample_spline = 5*len(data_train)
elif args.dataset == 'fmnist':
    data_train, data_test = load_data_fmnist()
    shape = [28,28,1]
    nsample_spline = 5*len(data_train)
elif args.dataset == 'cifar10':
    data_train, data_test = load_data_cifar10()
    shape = [32,32,3]
    nsample_spline = 5*len(data_train)

data_train = preprocess(data_train)

if args.evaluateFID:
    FID = []
else:
    del data_test

nsample_wT = len(data_train) 
nsample = nsample_wT + nsample_spline

batchsize = 1000 #Avoid running out of memory. Do not affect performance
verbose = True
update_iteration = 200
ndim = shape[0]*shape[1]*shape[2]

t_total = time.time()

#define the model
model = SIT(ndim=ndim).requires_grad_(False)

sample = torch.randn(nsample, ndim)
if args.evaluateFID:
    sample_test = torch.randn(10000, ndim)
else:
    sample_test = None

if not args.nohierarchy:
    if args.dataset == 'cifar10':
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
    
    for patch in patch_size:
        n_component = K_factor * patch[0]
        for _ in range(Niter):
            if patch[0] == shape[0]:
                model, sample, sample_test = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='regular', batchsize=batchsize, sample_test=sample_test)
            else:
                shift = torch.randint(32, (2,)).tolist()
                model, sample, sample_test = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='patch',
                                                                   shape=shape, kernel=patch, shift=shift, batchsize=batchsize, sample_test=sample_test)
            if len(model.layer) % update_iteration == 0:
                print()
                print('Finished %d iterations' % len(model.layer), 'Total Time:', time.time()-t_total)
                print()
                torch.save(model, args.save + 'SIG_%s_seed%d_hierarchy' % (args.dataset, args.seed))
                if args.evaluateFID:
                    sample_test1 = toimage(sample_test, shape)
                    FID.append(evaluate_fid_score(sample_test1.astype(np.float32), data_test.astype(np.float32)))
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
        model, sample, sample_test = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='regular', batchsize=batchsize, sample_test=sample_test)
        if len(model.layer) % update_iteration == 0:
            print()
            print('Finished %d iterations' % len(model.layer), 'Total Time:', time.time()-t_total)
            print()
            torch.save(model, args.save + 'SIG_%s_seed%d' % (args.dataset, args.seed))
            if args.evaluateFID:
                sample_test1 = toimage(sample_test, shape)
                FID.append(evaluate_fid_score(sample_test1.astype(np.float32), data_test.astype(np.float32)))
                print('Current FID score:', FID[-1])
                del sample_test1

