from sinf.SINF import *
from sinf.load_data import * 
import argparse

def GIS(data_train, data_validate=None, iteration=None, weight_train=None, weight_validate=None, K=None, M=None, KDE=True, b_factor=1, alpha=None, bounds=None,
        edge_bins=None, ndata_A=None, MSWD_max_iter=None, NBfirstlayer=False, Whiten=False, batchsize=None, nocuda=False, patch=False, shape=[28,28,1], model=None, verbose=True):
    
    '''
    data_train: (ndata_train, ndim).
    data_validate: (ndata_validate, ndim), optional. If provided, its logp will be used to determine the number of iterations.
    iteration: integer, optional. The maximum number of GIS iterations. Required if data_validate is not provided.
    weight_train: (ndata_train, ), optional. The weights of data_train.
    weight_validate: (ndata_train, ), optional. The weights of data_validate.
    K: integer, optional. The number of slices for each iteration. See max K-SWD in the SINF paper. 1 <= K <= ndim.
    M: integer, optional. The number of spline knots for rational quadratic splines.
    KDE: bool. Whether to use KDE for estimating 1D PDF. Recommended True.
    b_factor: positive float number, optional. The multiplicative factor for KDE kernel width.
    alpha: two non-negative float number in the format of (alpha1, alpha2), optional. Regularization parameter. See Equation 13 of SINF paper. alpha1 for interpolation, alpha2 for extrapolation slope. 0 <= alpha1,2 < 1. If not given, very heavy regularization will be used, which could result in slow training and a large number of iterations.
    bounds: sequence, optional. In the format of [[x1_min, x1_max], [x2_min, x2_max], ..., [xd_min, xd_max]]. Represent infinity and negative infinity with None.
    edge_bins: non-negative integer, optional. The number of spline knots at the boundary.
    ndata_A: positive integer, optional. The number of training data used for fitting A (slice axes).
    MSWD_max_iter: positive integer, optional. The maximum number of iterations for optimizing A (slice axes). See Algorithm 1 of SINF paper. Called L_iter in the paper.
    NBfirstlayer: bool, optional. Whether to use Naive Bayes (no rotation) at the first layer.
    Whiten: bool, optional. Whether to whiten the data before applying GIS.
    batchsize: positive integer, optional. The batch size for transforming the data. Does not change the performance. Only saves the memory. Useful when the data is too large and can't fit in the memory.
    nocuda: bool, optional. Whether to use gpu.
    patch: bool, optional. Whether to use patch-based modeling. Only useful for image datasets.
    shape: sequence, optional. The shape of the image datasets, if patch is enabled.
    model: GIS model, optional. Trained GIS model. If provided, new iterations will be added in the model.
    verbose: bool, optional. Whether to print training information.
    '''

    assert data_validate is not None or iteration is not None
 
    #hyperparameters
    ndim = data_train.shape[1]
    if weight_train is None:
        ndata = len(data_train)
    else:
        weight_train = weight_train / torch.sum(weight_train)
        select = weight_train > 1e-8
        if not select.all():
            weight_train = weight_train[select]
            data_train = data_train[select]
            weight_train = weight_train / torch.sum(weight_train)
        ndata = (torch.sum(weight_train)**2 / torch.sum(weight_train**2)).item()
    if M is None:
        M = max(min(200, int(ndata**0.5)), 50)
    if alpha is None:
        alpha = (1-0.02*math.log10(ndata), 1-0.001*math.log10(ndata))
    if bounds is not None:
        assert len(bounds) == ndim
        for i in range(ndim):
            assert len(bounds[i]) == 2
    if edge_bins is None:
        edge_bins = max(int(math.log10(ndata))-1, 0)
    if batchsize is None:
        batchsize = len(data_train)
    if not patch:
        if K is None:
            if ndim <= 8 or ndata / float(ndim) < 20:
                K = ndim
            else:
                K = 8
        if ndata_A is None:
            ndata_A = min(len(data_train), int(math.log10(ndim)*1e5))
        if MSWD_max_iter is None:
            MSWD_max_iter = min(round(ndata) // ndim, 200)
    else:
        assert shape[0] > 4 and shape[1] > 4
        K0 = K
        ndata_A0 = ndata_A
        MSWD_max_iter0 = MSWD_max_iter

    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() and not nocuda else "cpu")
    data_train = data_train.to(device)
    if weight_train is not None:
        weight_train = weight_train.to(device)
    if data_validate is not None:
        data_validate = data_validate.to(device)
        if weight_validate is not None:
            weight_validate = weight_validate.to(device)

    #define the model
    if model is None:
        model = SINF(ndim=ndim).requires_grad_(False).to(device)
        logj_train = torch.zeros(len(data_train), device=device)
        if data_validate is not None:
            logj_validate = torch.zeros(len(data_validate), device=device)
            best_logp_validate = -1e10
            best_Nlayer = 0
            wait = 0
            maxwait = 5 
    else:
        t = time.time()
        data_train, logj_train = transform_batch_model(model, data_train, batchsize, logj=None, start=0, end=None, nocuda=nocuda)
        if weight_train is None:
            logp_train = (torch.mean(logj_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_train**2,  dim=1)/2)).item()
        else:
            logp_train = (torch.sum(logj_train*weight_train)/torch.sum(weight_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_train*torch.sum(data_train**2,  dim=1)/2)/torch.sum(weight_train)).item()
    
        if data_validate is not None:
            data_validate, logj_validate = transform_batch_model(model, data_validate, batchsize, logj=None, start=0, end=None, nocuda=nocuda)
            if weight_validate is None:
                logp_validate = (torch.mean(logj_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_validate**2,  dim=1)/2)).item()
            else:
                logp_validate = (torch.sum(logj_validate*weight_validate)/torch.sum(weight_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_validate*torch.sum(data_validate**2,  dim=1)/2)/torch.sum(weight_validate)).item()
            print ('Initial logp:', logp_train, logp_validate, 'time:', time.time()-t, 'iteration:', len(model.layer))
        else:
            print ('Initial logp:', logp_train, 'time:', time.time()-t, 'iteration:', len(model.layer))

    #boundary
    if bounds is not None:
        layer = boundary(bounds=bounds, lambd=1e-5, beta=1).to(device)
        data_train, logj_train = layer(data_train)
        if weight_train is None:
            logp_train = (torch.mean(logj_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_train**2,  dim=1)/2)).item()
        else:
            logp_train = (torch.sum(logj_train*weight_train)/torch.sum(weight_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_train*torch.sum(data_train**2,  dim=1)/2)/torch.sum(weight_train)).item()

        if data_validate is not None:
            data_validate, logj_validate = layer(data_validate)
            if weight_validate is None:
                logp_validate = (torch.mean(logj_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_validate**2,  dim=1)/2)).item()
            else:
                logp_validate = (torch.sum(logj_validate*weight_validate)/torch.sum(weight_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_validate*torch.sum(data_validate**2,  dim=1)/2)/torch.sum(weight_validate)).item()
            best_logp_validate = logp_validate
            best_Nlayer = 1

        model.add_layer(layer)
        if verbose:
            if data_validate is not None:
                print('After boundary transform logp:', logp_train, logp_validate)
            else:
                print('After boundary transform logp:', logp_train)
    
    #whiten
    if Whiten:
        layer = whiten(ndim_data=ndim, scale=True, ndim_latent=ndim).requires_grad_(False).to(device)
        layer.fit(data_train, weight_train)

        data_train, logj_train0 = layer(data_train)
        logj_train += logj_train0
        if weight_train is None:
            logp_train = (torch.mean(logj_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_train**2,  dim=1)/2)).item()
        else:
            logp_train = (torch.sum(logj_train*weight_train)/torch.sum(weight_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_train*torch.sum(data_train**2,  dim=1)/2)/torch.sum(weight_train)).item()

        if data_validate is not None:
            data_validate, logj_validate0 = layer(data_validate)
            logj_validate += logj_validate0
            if weight_validate is None:
                logp_validate = (torch.mean(logj_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_validate**2,  dim=1)/2)).item()
            else:
                logp_validate = (torch.sum(logj_validate*weight_validate)/torch.sum(weight_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_validate*torch.sum(data_validate**2,  dim=1)/2)/torch.sum(weight_validate)).item()
            if logp_validate > best_logp_validate:
                best_logp_validate = logp_validate
                best_Nlayer = len(model.layer)

        model.add_layer(layer)
        if verbose:
            if data_validate is not None:
                print('After whiten logp:', logp_train, logp_validate)
            else:
                print('After whiten logp:', logp_train)


    #GIS iterations
    while True:
        t = time.time()
        if patch:
            #patch layers
            if len(model.layer) % 2 == 0:
                kernel = [4, 4, shape[-1]]
                shift = torch.randint(4, (2,)).tolist()
            else:
                kernel = [2, 2, shape[-1]]
                shift = torch.randint(2, (2,)).tolist()
            #hyperparameter
            ndim = np.prod(kernel)
            if K0 is None:
                if ndim <= 8 or len(data_train) / float(ndim) < 20:
                    K = ndim
                else:
                    K = 8
            elif K0 > ndim:
                K = ndim
            else:
                K = K0
            if ndata_A0 is None:
                ndata_A = min(len(data_train), int(math.log10(ndim)*1e5))
            if MSWD_max_iter0 is None:
                MSWD_max_iter = min(len(data_train) // ndim, 200)
            
            layer = PatchSlicedTransport(shape=shape, kernel=kernel, shift=shift, K=K, M=M).requires_grad_(False).to(device)
        else:
            #regular GIS layer
            if NBfirstlayer:
                layer = SlicedTransport(ndim=ndim, K=ndim, M=M).requires_grad_(False).to(device)
            else:
                layer = SlicedTransport(ndim=ndim, K=K, M=M).requires_grad_(False).to(device)
        
        #fit the layer
        if NBfirstlayer:
            layer.A[:] = torch.eye(ndim).to(device)
            NBfirstlayer = False
        elif ndim > 1:
            layer.fit_A(data=data_train, weight=weight_train, ndata_A=ndata_A, MSWD_max_iter=MSWD_max_iter, verbose=verbose)

        layer.fit_spline(data=data_train, weight=weight_train, edge_bins=edge_bins, alpha=alpha, KDE=KDE, b_factor=b_factor, batchsize=batchsize, verbose=verbose)

        #update the data
        data_train, logj_train = transform_batch_layer(layer, data_train, batchsize, logj=logj_train, direction='forward', nocuda=nocuda)
        if weight_train is None:
            logp_train = (torch.mean(logj_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_train**2,  dim=1)/2)).item()
        else:
            logp_train = (torch.sum(logj_train*weight_train)/torch.sum(weight_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_train*torch.sum(data_train**2,  dim=1)/2)/torch.sum(weight_train)).item()

        model.add_layer(layer)

        if data_validate is not None:
            data_validate, logj_validate = transform_batch_layer(layer, data_validate, batchsize, logj=logj_validate, direction='forward', nocuda=nocuda)
            if weight_validate is None:
                logp_validate = (torch.mean(logj_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_validate**2,  dim=1)/2)).item()
            else:
                logp_validate = (torch.sum(logj_validate*weight_validate)/torch.sum(weight_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_validate*torch.sum(data_validate**2,  dim=1)/2)/torch.sum(weight_validate)).item()
            if logp_validate > best_logp_validate:
                best_logp_validate = logp_validate
                best_Nlayer = len(model.layer)
                wait = 0
            else:
                wait += 1
            if wait == maxwait:
                model.layer = model.layer[:best_Nlayer]
                break

        if verbose:
            if data_validate is not None: 
                print ('logp:', logp_train, logp_validate, 'time:', time.time()-t, 'iteration:', len(model.layer), 'best:', best_Nlayer)
            else:
                print ('logp:', logp_train, 'time:', time.time()-t, 'iteration:', len(model.layer))

        if iteration is not None and len(model.layer) >= iteration:
            if data_validate is not None:
                model.layer = model.layer[:best_Nlayer]
            break

    return model


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument('--dataset', type=str, default='power',
                        choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300', 'mnist', 'fmnist', 'cifar10'],
                        help='Name of dataset to use.')
    
    parser.add_argument('--train_size', type=int, default=-1,
                        help='Size of training data. Negative or zero means all the training data.') 
    
    parser.add_argument('--validate_size', type=int, default=-1,
                        help='Size of validation data. Negative or zero means all the validation data.') 
    
    parser.add_argument('--seed', type=int, default=738,
                        help='Random seed for PyTorch and NumPy.')
    
    parser.add_argument('--whiten', action='store_true',
                        help='Whether to whiten the data before applying GIS. Not recommended for small datasets.')
    
    parser.add_argument('--save', type=str, default='/global/scratch/biwei/model/GIS/',
                        help='Where to save the trained model.')
    
    parser.add_argument('--restore', type=str, help='Path to model to restore.')

    parser.add_argument('--nocuda', action='store_true', help='Use cpu instead of gpu.')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and not args.nocuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')

    if args.dataset == 'power':
        data_train, data_validate, data_test = load_data_power()
    elif args.dataset == 'gas':
        data_train, data_validate, data_test = load_data_gas()
    elif args.dataset == 'hepmass':
        data_train, data_validate, data_test = load_data_hepmass()
    elif args.dataset == 'miniboone':
        data_train, data_validate, data_test = load_data_miniboone()
    elif args.dataset == 'bsds300':
        data_train, data_validate, data_test = load_data_bsds300()
    elif args.dataset == 'mnist':
        data_train, data_test = load_data_mnist()
        shape = [28,28,1]
    elif args.dataset == 'fmnist':
        data_train, data_test = load_data_fmnist()
        shape = [28,28,1]
    elif args.dataset == 'cifar10':
        data_train, data_test = load_data_cifar10()
        shape = [32,32,3]
    
    if args.dataset in ['power', 'gas', 'hepmass', 'miniboone', 'bsds300']:
        data_train = torch.tensor(data_train).float().to(device)
        data_validate = torch.tensor(data_validate).float().to(device)
        data_test = torch.tensor(data_test).float().to(device)
    else:
        data_train = torch.tensor(data_train).float().reshape(len(data_train), -1).to(device)
        data_train = data_train + torch.rand_like(data_train)
        data_test = torch.tensor(data_test).float().reshape(len(data_test), -1).to(device)
        data_test = data_test + torch.rand_like(data_test)
        if args.dataset in ['mnist', 'fmnist']:
            bounds = [[0,256]] * 784
        else:
            bounds = [[0,256]] * 3072
        
        data_validate = data_train[-10000:]
        data_train = data_train[:-10000]
    
    if args.train_size > 0:
        assert args.train_size <= len(data_train)
        data_train = data_train[torch.randperm(len(data_train))][:args.train_size]
    
    if args.validate_size > 0:
        if args.validate_size >= len(data_validate):
            args.validate_size = len(data_validate)
        else:
            data_validate = data_validate[torch.randperm(len(data_validate))][:args.validate_size]
    
    
    #define the model
    if args.restore:    
        t = time.time()
        model = torch.load(args.restore)
        print('Successfully load in the model. Time:', time.time()-t)
    else:
        model = None 
    
    t = time.time()

    #training
    if args.dataset in ['power', 'gas', 'hepmass', 'miniboone', 'bsds300']:
        model = GIS(data_train, data_validate, Whiten=args.whiten, batchsize=2**15, model=model, nocuda=args.nocuda)
    else:
        model = GIS(data_train, data_validate, bounds=bounds, Whiten=args.whiten, batchsize=2**15, patch=True, shape=shape, model=model, nocuda=args.nocuda)

    print('Training time:', time.time()-t)
    
    torch.save(model, args.save + 'GIS_%s_train%d_validate%d_seed%d' % (args.dataset, len(data_train), len(data_validate), args.seed))

    print('Test logp:', torch.mean(model.evaluate_density(data_test)).item())
    print ()
    print ()
    print ()
     
