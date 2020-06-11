from SIT import *
from load_data import load_data_fmnist

data_train, data_test = load_data_fmnist()

del data_test

data_train = data_train.float().view(len(data_train), 28*28)
data_train = (data_train + torch.rand_like(data_train)) / 256
data_train = 2*data_train - 1

KDE = False
nsample_wT = len(data_train) 
nsample_spline = len(data_train)
nsample = nsample_wT + nsample_spline 

batchsize = 30000 #to avoid running out of memory

shape = [28, 28, 1]
ndim_latent = shape[0]*shape[1]*shape[2]

t_total = time.time()

#define the model
model = SIT(ndim=ndim_latent).requires_grad_(False).cuda()

sample = torch.randn(nsample, ndim_latent)

#Add 2000 regular layers
for i in range(2000):
    if len(model.layer) < 100:
        n_component = 24 
    else:
        n_component = 16 

    model, sample = add_one_layer_inverse(model, data_train, sample, n_component, nsample_wT, nsample_spline, layer_type='regular', KDE=KDE, batchsize=batchsize)

print (time.time()-t_total) 
torch.save(model, 'SIG_FashionMNIST')
