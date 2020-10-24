from SIT import *

import torchvision
train_data = torchvision.datasets.MNIST(root="/global/scratch/biwei/data", train=True, download=True, transform=None)
test_data = torchvision.datasets.MNIST(root="/global/scratch/biwei/data", train=False, download=True, transform=None)

data_train = train_data.data[:50000].reshape(50000, -1).float().cuda()
data_train = (torch.rand_like(data_train) + data_train) / 256.
data_validate = train_data.data[50000:].reshape(10000, -1).float().cuda()
data_validate = (torch.rand_like(data_validate) + data_validate) / 256.
data_test = test_data.data.reshape(10000, -1).float().cuda()
data_test = (torch.rand_like(data_test) + data_test) / 256.

data_train0 = data_train.clone()
data_validate0 = data_validate.clone()
data_test0 = data_test.clone()

label_train = train_data.targets[:50000].cuda()
label_validate = train_data.targets[50000:].cuda()
label_test = test_data.targets.cuda()

del train_data, test_data

#hyperparameter
ndim = data_train.shape[1]
interp_nbin = 50
KDE = True
bw_factor_data = 1
alpha = (0.5, 0.9)
edge_bins = 0
batchsize = 2**15 #does not affect performance, only prevent out of memory
verbose = True
MSWD_max_iter = 200
n_class = 10
derivclip = None

#model
model = SIT(ndim=ndim).requires_grad_(False).cuda()

accuracy_train = []
accuracy_validate = []
accuracy_test = []

best_validate_accuracy = 0
best_Nlayer = 0

#logit transform
layer = logit(lambd=1e-5).cuda()
data_train, logj_train = layer(data_train)
data_validate, logj_validate = layer(data_validate)
data_test, logj_test = layer(data_test)

model.add_layer(layer)

data_train = data_train[None,:].repeat_interleave(n_class, axis=0)
data_validate = data_validate[None,:].repeat_interleave(n_class, axis=0)
data_test = data_test[None,:].repeat_interleave(n_class, axis=0)

logj_train = logj_train[None,:].repeat_interleave(n_class, axis=0)
logj_validate = logj_validate[None,:].repeat_interleave(n_class, axis=0)
logj_test = logj_test[None,:].repeat_interleave(n_class, axis=0)

#Gaussianization
n_component = 64
for i in range(10):
    t = time.time()

    layer = ConditionalSlicedTransport_discrete(ndim=ndim, n_class=n_class, n_component=n_component, interp_nbin=interp_nbin).requires_grad_(False).cuda()

    layer.fit_wT(data=data_train[label, torch.arange(data_train.shape[1], device=data_train.device)], verbose=verbose)

    SWD = layer.fit_spline(data=data_train[label, torch.arange(data_train.shape[1], device=data_train.device)], label=label_train, edge_bins=edge_bins, 
                           derivclip=derivclip, alpha=alpha, KDE=KDE, bw_factor=bw_factor_data, batchsize=batchsize, verbose=verbose)

    for label in range(n_class):
        data_train[label], logj_train1 = layer(data_train[label], torch.ones(data_train.shape[1], dtype=torch.int, device=data_train.device)*label)
        logj_train[label] = logj_train[label] + logj_train1

        data_validate[label], logj_validate1 = layer(data_validate[label], torch.ones(data_validate.shape[1], dtype=torch.int, device=data_validate.device)*label)
        logj_validate[label] = logj_validate[label] + logj_validate1

        data_test[label], logj_test1 = layer(data_test[label], torch.ones(data_test.shape[1], dtype=torch.int, device=data_test.device)*label)
        logj_test[label] = logj_test[label] + logj_test1

    model.add_layer(layer)

    predict_label_train = torch.argmax(logj_train-torch.sum(data_train**2, dim=2)/2., dim=0) 
    accuracy_train.append(torch.sum(predict_label_train==label_train).item() / len(label_train))

    predict_label_validate = torch.argmax(logj_validate-torch.sum(data_validate**2, dim=2)/2., dim=0) 
    accuracy_validate.append(torch.sum(predict_label_validate==label_validate).item() / len(label_validate))

    predict_label_test = torch.argmax(logj_test-torch.sum(data_test**2, dim=2)/2., dim=0) 
    accuracy_test.append(torch.sum(predict_label_test==label_test).item() / len(label_test))
    print ('accuracy:', accuracy_train[-1], accuracy_validate[-1], accuracy_test[-1], 'Time:', time.time()-t)

    if accuracy_validate[-1] > best_validate_accuracy:
        best_Nlayer = len(model.layer)
        best_validate_accuracy = accuracy_validate[-1]


#discriminative loss
n_component = 8
batchsize = 1000
interp_nbin = 20
margin = 10
while True:
    t = time.time()

    layer = ConditionalSlicedTransport_discrete(ndim=ndim, n_class=n_class, n_component=n_component, interp_nbin=interp_nbin).requires_grad_(False).cuda()

    loss = layer.fit(data=data_train, label=label_train, logj=logj_train, margin=margin, lr=(2e-3, 2e-3), maxepoch=100, batchsize=batchsize, verbose=verbose)

    for label in range(n_class):
        data_train[label], logj_train1 = layer(data_train[label], torch.ones(data_train.shape[1], dtype=torch.int, device=data_train.device)*label)
        logj_train[label] = logj_train[label] + logj_train1

        data_validate[label], logj_validate1 = layer(data_validate[label], torch.ones(data_validate.shape[1], dtype=torch.int, device=data_validate.device)*label)
        logj_validate[label] = logj_validate[label] + logj_validate1

        data_test[label], logj_test1 = layer(data_test[label], torch.ones(data_test.shape[1], dtype=torch.int, device=data_test.device)*label)
        logj_test[label] = logj_test[label] + logj_test1

    model.add_layer(layer)

    predict_label_train = torch.argmax(logj_train-torch.sum(data_train**2, dim=2)/2., dim=0) 
    accuracy_train.append(torch.sum(predict_label_train==label_train).item() / len(label_train))

    predict_label_validate = torch.argmax(logj_validate-torch.sum(data_validate**2, dim=2)/2., dim=0) 
    accuracy_validate.append(torch.sum(predict_label_validate==label_validate).item() / len(label_validate))

    predict_label_test = torch.argmax(logj_test-torch.sum(data_test**2, dim=2)/2., dim=0) 
    accuracy_test.append(torch.sum(predict_label_test==label_test).item() / len(label_test))
    print ('accuracy:', accuracy_train[-1], accuracy_validate[-1], accuracy_test[-1], 'Time:', time.time()-t)

    if accuracy_validate[-1] > best_validate_accuracy:
        best_Nlayer = len(model.layer)
        best_validate_accuracy = accuracy_validate[-1]
