from SIT import *
from load_data import load_data_celeba

data_train = load_data_celeba(flag='training', side_length=32)

data_train = torch.tensor(data_train).float().view(len(data_train), 32*32*3)
data_train = (data_train + torch.rand_like(data_train)) / 256
data_train = 2*data_train - 1

KDE = False
nsample_wT = 50000 
nsample_spline = 50000
nsample = nsample_wT + nsample_spline 

batchsize = 5000 #to avoid running out of memory

shape = [32, 32, 3]
ndim_latent = shape[0]*shape[1]*shape[2]

t_total = time.time()

#define the model
model = SIT(ndim=ndim_latent).requires_grad_(False).cuda()

sample = torch.randn(nsample, ndim_latent)

#Add 200 regular layers
for i in range(200):
    if len(model.layer) < 100:
        n_component = 64 
    else:
        n_component = 32 

    model, sample = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='regular', KDE=KDE, batchsize=batchsize)

#Add 1 * 100 InterPatch layer
for i in range(1):
    for j in range(10):
        n_component = 2 
        kernel_size = [16,16]
        shift = [0,0] 
        #shift = torch.randint(14, (2,))

        model, sample = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='interpatch', 
                                              shape=shape, kernel_size=kernel_size, shift=shift, KDE=KDE, batchsize=batchsize)
            
    for j in range(20):
        n_component = 4 
        kernel_size = [8,8]
        shift = torch.randint(8, (2,)).tolist()

        model, sample = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='interpatch', 
                                              shape=shape, kernel_size=kernel_size, shift=shift, KDE=KDE, batchsize=batchsize)

    for j in range(30):
        n_component = 8
        kernel_size = [4,4]
        shift = torch.randint(4, (2,)).tolist()

        model, sample = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='interpatch', 
                                              shape=shape, kernel_size=kernel_size, shift=shift, KDE=KDE, batchsize=batchsize)

    for j in range(40):
        n_component = 16 
        kernel_size = [2,2]
        shift = torch.randint(2, (2,)).tolist()

        model, sample = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='interpatch', 
                                              shape=shape, kernel_size=kernel_size, shift=shift, KDE=KDE, batchsize=batchsize)

#Add 17 * 100 Patch layers
for i in range(12):
    
    for j in range(40):
        kernel_size = [16,16]
        n_component = 16
        shift = torch.randint(16, (2,)).tolist()

        model, sample = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='patch', 
                                              shape=shape, kernel_size=kernel_size, shift=shift, KDE=KDE, batchsize=batchsize)

    for j in range(30):
        kernel_size = [8,8]
        n_component = 8
        shift = torch.randint(8, (2,)).tolist()

        model, sample = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='patch', 
                                              shape=shape, kernel_size=kernel_size, shift=shift, KDE=KDE, batchsize=batchsize)

    for j in range(20):
        kernel_size = [4,4]
        n_component = 4
        shift = torch.randint(4, (2,)).tolist()

        model, sample = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='patch', 
                                              shape=shape, kernel_size=kernel_size, shift=shift, KDE=KDE, batchsize=batchsize)

    for j in range(10):
        kernel_size = [2,2]
        n_component = 2 
        shift = torch.randint(2, (2,)).tolist()

        model, sample = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='patch', 
                                              shape=shape, kernel_size=kernel_size, shift=shift, KDE=KDE, batchsize=batchsize)

    print (time.time()-t_total) 
    torch.save(model, 'SIG_CelebA_hierarchy')
