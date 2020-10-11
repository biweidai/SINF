from SIT import *
from load_data import * 
import argparse
from fid_score import evaluate_fid_score

def toimage(sample, shape):
    sample = (sample+1)*128
    sample = sample.cpu().numpy().astype(int).reshape(len(sample), *shape)
    sample[sample<0] = 0
    sample[sample>255] = 255
    return sample

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

data_train = torch.tensor(data_train).float().reshape(len(data_train), -1)
data_train = (data_train + torch.rand_like(data_train)) / 128. - 1

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
                    FID.append(evaluate_fid_score(fake_images=sample_test1.astype('float'), real_images=data_test.astype('float'), norm=True))
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
                FID.append(evaluate_fid_score(fake_images=sample_test1.astype('float'), real_images=data_test.astype('float'), norm=True))
                print('Current FID score:', FID[-1])
                del sample_test1

