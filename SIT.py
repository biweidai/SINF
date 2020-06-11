import torch
import torch.nn as nn
import numpy as np
import time
import math
import torch.optim as optim
from SlicedWasserstein import *
from RQspline import *

quiet = False

def train(model, train_loader, optimizer_ortho, optimizer_spline):
    model.train()
    
    train_losses = []
    for x in train_loader:
        x = x.cuda().contiguous()
        loss = model.loss(x)
        optimizer_ortho.zero_grad()
        optimizer_spline.zero_grad()
        loss.backward()
        optimizer_ortho.step()
        optimizer_spline.step()
        train_losses.append(loss.item())
    return train_losses


def eval_loss(model, data_loader):
    model.eval()
    total_loss = 0
    ntotal = 0
    with torch.no_grad():
        for x in data_loader:
            x = x.cuda().contiguous()
            loss = model.loss(x)
            total_loss += loss * x.shape[0]
            ntotal += x.shape[0]
        avg_loss = total_loss / ntotal

    return avg_loss.item()



class SIT(nn.Module):

    #sliced iterative transport model
    
    def __init__(self, ndim):
        
        super().__init__()
        
        self.layer = nn.ModuleList([])
        self.ndim = ndim
    
    def forward(self, data, start=0, end=None, param=None):
        
        if data.ndim == 1:
            data = data.view(1,-1)
        if end is None:
            end = len(self.layer)
        elif end < 0:
            end += len(self.layer)
        if start < 0:
            start += len(self.layer)
        
        assert start >= 0 and end >= 0 and end >= start

        logj = torch.zeros(data.shape[0], device=data.device)
        
        for i in range(start, end):
            data, log_j = self.layer[i](data, param=param)
            logj += log_j

        return data, logj
    
    
    def inverse(self, data, start=None, end=0, param=None):

        if data.ndim == 1:
            data = data.view(1,-1)
        if end < 0:
            end += len(self.layer)
        if start is None:
            start = len(self.layer)
        elif start < 0:
            start += len(self.layer)
        
        assert start >= 0 and end >= 0 and end <= start

        logj = torch.zeros(data.shape[0], device=data.device)
        
        for i in reversed(range(end, start)):
            data, log_j = self.layer[i].inverse(data, param=param)
            logj += log_j

        return data, logj
    
    
    def add_layer(self, layer, position=None):
        
        if position is None or position == len(self.layer):
            self.layer.append(layer)
        else:
            if position < 0:
                position += len(self.layer)
            assert position >= 0 and position < len(self.layer)
            self.layer.insert(position, layer)
        
        return self
    
    
    def delete_layer(self, position=-1):
        
        if position == -1 or position == len(self.layer)-1:
            self.layer = self.layer[:-1]
        else:
            if position < 0:
                position += len(self.layer)
            assert position >= 0 and position < len(self.layer)-1
            
            for i in range(position, len(self.layer)-1):
                self.layer._modules[str(i)] = self.layer._modules[str(i + 1)]
            self.layer = self.layer[:-1]
        
        return self
    
    
    def evaluate_density(self, data, start=0, end=None, param=None):
        
        data, logj = self.forward(data, start=start, end=end, param=param)
        logq = -self.ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(data**2,  dim=1)/2
        logp = logj + logq
        
        return logp


    def loss(self, data, start=0, end=None, param=None):
        return -torch.mean(self.evaluate_density(data, start=start, end=end, param=param))
    
    
    def sample(self, nsample, start=None, end=0, device=torch.device('cuda'), param=None):

        #device must be the same as the device of the model
        
        x = torch.randn(nsample, self.ndim, device=device)
        logq = -self.ndim/2.*torch.log(torch.tensor(2.*math.pi)) - torch.sum(x**2,  dim=1)/2
        x, logj = self.inverse(x, start=start, end=end, param=param)
        logp = logj + logq

        return x, logp



class whiten(nn.Module):

    #whiten layer

    def __init__(self, ndim_data, scale=True, ndim_latent=None):

        super().__init__()
        if ndim_latent is None:
            ndim_latent = ndim_data
        assert ndim_latent <= ndim_data
        self.ndim_data = ndim_data
        self.ndim_latent = ndim_latent
        self.scale = scale

        self.mean = nn.Parameter(torch.zeros(ndim_data))
        self.D = nn.Parameter(torch.ones(ndim_data))
        self.E = nn.Parameter(torch.eye(ndim_data))
        select = torch.zeros(ndim_data, dtype=torch.bool)
        select[:ndim_latent] = True
        self.register_buffer('select', select)


    def fit(self, data):

        assert data.ndim == 2

        with torch.no_grad():
            self.mean[:] = torch.mean(data, dim=0)
            data0 = data - self.mean
            covariance = data0.T @ data0 / (data0.shape[0]-1)
            D, E = torch.symeig(covariance, eigenvectors=True)
            self.D[:] = torch.flip(D, dims=(0,))
            self.E[:] = torch.flip(E, dims=(1,))

            return self


    def forward(self, data, param=None):

        data0 = data - self.mean

        if self.scale:
            D1 = self.D[self.select]**(-0.5)
            data0 = (torch.diag(D1) @ (self.E.T @ data0.T)[self.select]).T
            logj = torch.repeat_interleave(torch.sum(torch.log(D1)), len(data))
        else:
            data0 = (self.E.T @ data0.T)[self.select].T
            logj = torch.zeros(len(data), device=data.device)

        return data0, logj


    def inverse(self, data, param=None):

        data0 = torch.zeros([data.shape[0], self.ndim_data], device=data.device)
        data0[:, self.select] = data[:]
        if self.scale:
            D1 = self.D**0.5
            D1[~self.select] = 0.
            data0 = (self.E @ torch.diag(D1) @ data0.T).T
            logj = -torch.repeat_interleave(torch.sum(torch.log(D1[self.select])), len(data))
        else:
            data0 = (self.E @ data0.T).T
            logj = torch.zeros(len(data), device=data.device)
        data0 += self.mean

        return data0, logj



class SlicedTransport(nn.Module):

    #1 layer of sliced transport
    def __init__(self, ndim, n_component=None, interp_nbin=100):

        super().__init__()
        self.ndim = ndim
        if n_component is None:
            self.n_component = ndim
        else:
            self.n_component = n_component
        self.interp_nbin = interp_nbin

        wi = torch.randn(self.ndim, self.n_component)
        Q, R = torch.qr(wi)
        L = torch.sign(torch.diag(R))
        wT = (Q * L)

        self.wT = nn.Parameter(wT)
        self.transform1D = RQspline(self.n_component, interp_nbin)


    def fit_wT(self, data, sample='gaussian', MSWD_p=2, MSWD_max_iter=200, verbose=True):

        #fit the directions to apply 1D transform

        if verbose:
            tstart = torch.cuda.Event(enable_timing=True)
            tend = torch.cuda.Event(enable_timing=True)
            tstart.record()

        wT, SWD = maxSWDdirection(data, x2=sample, n_component=self.n_component, maxiter=MSWD_max_iter, p=MSWD_p)
        with torch.no_grad():
            SWD, indices = torch.sort(SWD, descending=True)
            wT = wT[:,indices]
            self.wT[:] = torch.qr(wT)[0] 

        if verbose:
            tend.record()
            torch.cuda.synchronize()
            t = tstart.elapsed_time(tend)
            print ('Fit wT:', 'Time:', t/1000., 'Wasserstein Distance:', SWD.tolist())
        return self 


    def fit_spline(self, data, edge_bins=4, derivclip=None, extrapolate='regression', alpha=0, noise_threshold=0, MSWD_p=2, KDE=True, bw_factor=1, batchsize=None, verbose=True):

        #fit the 1D transform \Psi

        assert extrapolate in ['endpoint', 'regression']
        assert self.interp_nbin > 2 * edge_bins
        assert derivclip is None or derivclip >= 1

        with torch.no_grad():
            if verbose:
                tstart = torch.cuda.Event(enable_timing=True)
                tend = torch.cuda.Event(enable_timing=True)
                tstart.record()
            SWD = SlicedWasserstein_direction(data, self.wT, second='gaussian', p=MSWD_p)
            data0 = data @ self.wT

            #build rational quadratic spline transform
            x, y, deriv = estimate_knots_gaussian(data0, interp_nbin=self.interp_nbin, above_noise=(SWD>noise_threshold), edge_bins=edge_bins, 
                                                  derivclip=derivclip, extrapolate=extrapolate, alpha=alpha, KDE=KDE, bw_factor=bw_factor, batchsize=batchsize)
            self.transform1D.set_param(x, y, deriv)

            if verbose:
                tend.record()
                torch.cuda.synchronize()
                t = tstart.elapsed_time(tend)
                print ('Fit spline:', 'Time:', t/1000., 'Wasserstein Distance:', SWD.tolist())

            return SWD


    def fit_spline_inverse(self, data, sample, edge_bins=4, derivclip=None, extrapolate='regression', alpha=0, noise_threshold=0, MSWD_p=2, KDE=True, bw_factor_data=1, bw_factor_sample=1, batchsize=None, verbose=True):

        #fit the 1D transform \Psi
        #inverse method

        assert extrapolate in ['endpoint', 'regression']
        assert self.interp_nbin > 2 * edge_bins
        assert derivclip is None or derivclip >= 1

        with torch.no_grad():
            if verbose:
                tstart = torch.cuda.Event(enable_timing=True)
                tend = torch.cuda.Event(enable_timing=True)
                tstart.record()
            SWD = SlicedWasserstein_direction(data, self.wT, second=sample, p=MSWD_p)
            data0 = data @ self.wT
            sample0 = sample @ self.wT

            #build rational quadratic spline transform
            x, y, deriv = estimate_knots(data0, sample0, interp_nbin=self.interp_nbin, above_noise=(SWD>noise_threshold), edge_bins=edge_bins, derivclip=derivclip, 
                                         extrapolate=extrapolate, alpha=alpha, KDE=KDE, bw_factor_data=bw_factor_data, bw_factor_sample=bw_factor_sample, batchsize=batchsize)
            self.transform1D.set_param(x, y, deriv)

            if verbose:
                tend.record()
                torch.cuda.synchronize()
                t = tstart.elapsed_time(tend)
                print ('Fit spline:', 'Time:', t/1000., 'Wasserstein Distance:', SWD.tolist())

            return SWD


    def transform(self, data, mode='forward', param=None):

        data0 = data @ self.wT
        remaining = data - data0 @ self.wT.T
        if mode is 'forward':
            data0, logj = self.transform1D(data0)
        elif mode is 'inverse':
            data0, logj = self.transform1D.inverse(data0)
        logj = torch.sum(logj, dim=1)
        data = remaining + data0 @ self.wT.T

        return data, logj


    def forward(self, data, param=None):
        return self.transform(data, mode='forward', param=param)


    def inverse(self, data, param=None):
        return self.transform(data, mode='inverse', param=param)



def Shift(data, shift):
    if shift[0] != 0:
        shiftx = shift[0]
        left = data.shape[1] - shiftx
        temp = torch.clone(data[:,:shiftx,:,:])
        data[:,:left,:,:] = torch.clone(data[:,shiftx:,:,:])
        data[:,left:,:,:] = temp
    if shift[1] != 0:
        shifty = shift[1]
        left = data.shape[2] - shifty
        temp = torch.clone(data[:,:,:shifty,:])
        data[:,:,:left,:] = torch.clone(data[:,:,shifty:,:])
        data[:,:,left:,:] = temp
    return data
     

def UnShift(data, shift):
    if shift[0] != 0:
        shiftx = shift[0]
        left = data.shape[1] - shiftx
        temp = torch.clone(data[:,left:,:,:])
        data[:,shiftx:,:,:] = torch.clone(data[:,:left,:,:])
        data[:,:shiftx,:,:] = temp
    if shift[1] != 0:
        shifty = shift[1]
        left = data.shape[2] - shifty
        temp = torch.clone(data[:,:,left:,:])
        data[:,:,shifty:,:] = torch.clone(data[:,:,:left,:])
        data[:,:,:shifty,:] = temp
    return data


class PatchSlicedTransport(nn.Module):

    #1 layer of patch based sliced transport 

    def __init__(self, shape=[28,28,1], kernel_size=[4,4], shift=[0,0], n_component=None, interp_nbin=100):

        super().__init__()
        self.register_buffer('shape', torch.tensor(shape))
        self.register_buffer('kernel', torch.tensor(kernel_size)) 
        self.register_buffer('shift', torch.tensor(shift))
        
        while self.shift[0] >= self.kernel[0]:
            self.shift[0] -= self.kernel[0]
        while self.shift[1] >= self.kernel[1]:
            self.shift[1] -= self.kernel[1]
        if n_component is None:
            self.n_component = torch.prod(self.kernel).item()
        else:
            self.n_component = n_component
            assert n_component <= torch.prod(self.kernel)
        self.interp_nbin = interp_nbin
        
        self.ndim_sub = (self.kernel[0]*self.kernel[1]*shape[2]).item()
        self.Nkernel_x = (self.shape[0] // self.kernel[0]).item()
        self.Nkernel_y = (self.shape[1] // self.kernel[1]).item()
        self.Nkernel = self.Nkernel_x * self.Nkernel_y

        wT = torch.zeros(self.Nkernel, self.ndim_sub, self.n_component)
        for i in range(self.Nkernel):
            wi = torch.randn(self.ndim_sub, self.n_component)
            Q, R = torch.qr(wi)
            L = torch.sign(torch.diag(R))
            wT[i] = (Q * L)

        self.wT = nn.Parameter(wT)
        self.transform1D = nn.ModuleList([RQspline(self.n_component, interp_nbin) for i in range(self.Nkernel)])

    
    def fit_wT(self, data, sample, MSWD_p=2, MSWD_max_iter=200, verbose=True):

        #fit the directions to apply 1D transform

        if verbose:
            tstart = torch.cuda.Event(enable_timing=True)
            tend = torch.cuda.Event(enable_timing=True)
            tstart.record()

        data = data.reshape(len(data), *self.shape)
        sample = sample.reshape(len(sample), *self.shape)
        data = Shift(data, self.shift)
        sample = Shift(sample, self.shift)

        SWD = torch.zeros(self.Nkernel, self.n_component, device=data.device)

        for j in range(self.Nkernel_y):
            for i in range(self.Nkernel_x):
                data0 = data[:, i*self.kernel[0]:(i+1)*self.kernel[0], j*self.kernel[1]:(j+1)*self.kernel[1], :].reshape(len(data), -1)
                sample0 = sample[:, i*self.kernel[0]:(i+1)*self.kernel[0], j*self.kernel[1]:(j+1)*self.kernel[1], :].reshape(len(sample), -1)
                index = j*self.Nkernel_x+i
                wT, SWD[index] = maxSWDdirection(data0, sample0, n_component=self.n_component, maxiter=MSWD_max_iter, p=MSWD_p)
                with torch.no_grad():
                    SWD[index], indices = torch.sort(SWD[index], descending=True)
                    wT = wT[:, indices]
                    self.wT[index] = torch.qr(wT)[0]

        data = UnShift(data, self.shift)
        sample = UnShift(sample, self.shift)
        data = data.reshape(len(data), -1)
        sample = sample.reshape(len(sample), -1)

        if verbose:
            tend.record()
            torch.cuda.synchronize()
            t = tstart.elapsed_time(tend)
            print ('Fit wT:', 'Time:', t/1000., 'Wasserstein Distance:', SWD.tolist())

        return self 


    def fit_spline(self, data, edge_bins=4, derivclip=None, extrapolate='regression', alpha=0, noise_threshold=0, MSWD_p=2, KDE=True, bw_factor=1, batchsize=None, verbose=True):
        raise NotImplementedError


    def fit_spline_inverse(self, data, sample, edge_bins=4, derivclip=None, extrapolate='regression', alpha=0, noise_threshold=0, MSWD_p=2, KDE=True, bw_factor_data=1, bw_factor_sample=1, batchsize=None, verbose=True):

        #fit the 1D transform \Psi
        #inverse method

        assert extrapolate in ['endpoint', 'regression']
        assert self.interp_nbin > 2 * edge_bins
        assert derivclip is None or derivclip >= 1

        with torch.no_grad():
            if verbose:
                tstart = torch.cuda.Event(enable_timing=True)
                tend = torch.cuda.Event(enable_timing=True)
                tstart.record()

            data = data.reshape(len(data), *self.shape)
            sample = sample.reshape(len(sample), *self.shape)
            data = Shift(data, self.shift)
            sample = Shift(sample, self.shift)

            SWD = torch.zeros(self.Nkernel, self.n_component, device=data.device)

            for indexy in range(self.Nkernel_y):
                for indexx in range(self.Nkernel_x):
                    index = indexy*self.Nkernel_x+indexx

                    data0 = data[:, indexx*self.kernel[0]:(indexx+1)*self.kernel[0], indexy*self.kernel[1]:(indexy+1)*self.kernel[1], :].reshape(len(data), -1) @ self.wT[index]
                    sample0 = sample[:, indexx*self.kernel[0]:(indexx+1)*self.kernel[0], indexy*self.kernel[1]:(indexy+1)*self.kernel[1], :].reshape(len(sample), -1) @ self.wT[index]
                    SWD[index] = SlicedWasserstein_direction(data0, None, second=sample0, p=MSWD_p)

                    #build rational quadratic spline transform using kde
                    x, y, deriv = estimate_knots(data0, sample0, interp_nbin=self.interp_nbin, above_noise=(SWD[index]>noise_threshold), edge_bins=edge_bins, derivclip=derivclip, 
                                                 extrapolate=extrapolate, alpha=alpha, KDE=KDE, bw_factor_data=bw_factor_data, bw_factor_sample=bw_factor_sample, batchsize=batchsize)

                    self.transform1D[index].set_param(x, y, deriv)

            data = UnShift(data, self.shift)
            sample = UnShift(sample, self.shift)
            data = data.reshape(len(data), -1)
            sample = sample.reshape(len(sample), -1)

            if verbose:
                tend.record()
                torch.cuda.synchronize()
                t = tstart.elapsed_time(tend)
                print ('Fit spline:', 'Time:', t/1000., 'Wasserstein Distance:', SWD.tolist())

            return SWD


    def transform(self, data, mode='forward', param=None):

        logj = torch.zeros(len(data), device=data.device)
        data = data.reshape(len(data), *self.shape)
        data = Shift(data, self.shift)

        for indexy in range(self.Nkernel_y):
            for indexx in range(self.Nkernel_x):            
                index = indexy*self.Nkernel_x+indexx
                data0 = data[:, indexx*self.kernel[0]:(indexx+1)*self.kernel[0], indexy*self.kernel[1]:(indexy+1)*self.kernel[1], :].reshape(len(data), -1) 
                data1 = data0 @ self.wT[index]
                remaining = data0 - data1 @ self.wT[index].T
                if mode is 'forward':
                    data1, logj1 = self.transform1D[index](data1)
                elif mode is 'inverse':
                    data1, logj1 = self.transform1D[index].inverse(data1)
                logj += torch.sum(logj1, dim=1)
                data0 = (remaining + data1 @ self.wT[index].T).reshape(len(data), self.kernel[0], self.kernel[1], self.shape[-1])
                data[:, indexx*self.kernel[0]:(indexx+1)*self.kernel[0], indexy*self.kernel[1]:(indexy+1)*self.kernel[1], :] = data0 

        data = UnShift(data, self.shift)
        data = data.reshape(len(data), -1)

        return data, logj


    def forward(self, data, param=None):
        return self.transform(data, mode='forward', param=param)


    def inverse(self, data, param=None):
        return self.transform(data, mode='inverse', param=param)



class InterPatchSlicedTransport(nn.Module):

    #1 layer of sliced transport on smoothed images

    def __init__(self, shape=[28,28,1], kernel_size=[2,2], shift=[0,0], n_component=None, interp_nbin=100):

        super().__init__()
        self.register_buffer('shape', torch.tensor(shape))
        self.register_buffer('kernel', torch.tensor(kernel_size)) 
        self.register_buffer('shift', torch.tensor(shift))
        
        self.kernel_Npixel = self.kernel[0] * self.kernel[1]
        self.Nkernel_x = self.shape[0] // self.kernel[0]
        self.Nkernel_y = self.shape[1] // self.kernel[1]
        self.Nkernel = self.Nkernel_x * self.Nkernel_y * shape[2]
        while self.shift[0] >= self.kernel[0]:
            self.shift[0] -= self.kernel[0]
        while self.shift[1] >= self.kernel[1]:
            self.shift[1] -= self.kernel[1]
        if n_component is None:
            self.n_component = self.Nkernel
        else:
            self.n_component = n_component
            assert n_component <= self.Nkernel
        self.interp_nbin = interp_nbin

        wi = torch.randn(self.Nkernel, self.n_component)
        Q, R = torch.qr(wi)
        L = torch.sign(torch.diag(R))
        wT = (Q * L)

        self.wT = nn.Parameter(wT)
        self.transform1D = RQspline(self.n_component, interp_nbin)


    def fit_wT(self, data, sample, MSWD_p=2, MSWD_max_iter=200, verbose=True):

        #fit the directions to apply 1D transform

        if verbose:
            tstart = torch.cuda.Event(enable_timing=True)
            tend = torch.cuda.Event(enable_timing=True)
            tstart.record()

        data = data.reshape(len(data), *self.shape)
        sample = sample.reshape(len(sample), *self.shape)
        data = Shift(data, self.shift)
        sample = Shift(sample, self.shift)

        data0 = torch.zeros(len(data), self.Nkernel_x, self.Nkernel_y, self.shape[-1], device=data.device)
        sample0 = torch.zeros(len(sample), self.Nkernel_x, self.Nkernel_y, self.shape[-1], device=sample.device)
        for j in range(self.Nkernel_y):
            for i in range(self.Nkernel_x):
                data0[:,i,j,:] = torch.mean(data[:, i*self.kernel[0]:(i+1)*self.kernel[0], j*self.kernel[1]:(j+1)*self.kernel[1], :], dim=[1,2])
                sample0[:,i,j,:] = torch.mean(sample[:, i*self.kernel[0]:(i+1)*self.kernel[0], j*self.kernel[1]:(j+1)*self.kernel[1], :], dim=[1,2])
        data0 = data0.reshape(len(data0), -1)
        sample0 = sample0.reshape(len(sample0), -1)

        wT, SWD = maxSWDdirection(data0, sample0, n_component=self.n_component, maxiter=MSWD_max_iter, p=MSWD_p)
        with torch.no_grad():
            SWD, indices = torch.sort(SWD, descending=True)
            wT = wT[:, indices]
            self.wT[:] = torch.qr(wT)[0]

        data = UnShift(data, self.shift)
        sample = UnShift(sample, self.shift)
        data = data.reshape(len(data), -1)
        sample = sample.reshape(len(sample), -1)

        if verbose:
            tend.record()
            torch.cuda.synchronize()
            t = tstart.elapsed_time(tend)
            print ('Fit wT:', 'Time:', t/1000., 'Wasserstein Distance:', SWD.tolist())

        return self 


    def fit_spline(self, data, edge_bins=4, derivclip=None, extrapolate='regression', alpha=0, noise_threshold=0, MSWD_p=2, KDE=True, bw_factor=1, batchsize=None, verbose=True):
        raise NotImplementedError


    def fit_spline_inverse(self, data, sample, edge_bins=4, derivclip=None, extrapolate='regression', alpha=0, noise_threshold=0, MSWD_p=2, KDE=True, bw_factor_data=1, bw_factor_sample=1, batchsize=None, verbose=True):

        #fit the 1D transform \Psi
        #inverse method

        assert extrapolate in ['endpoint', 'regression']
        assert self.interp_nbin > 2 * edge_bins
        assert derivclip is None or derivclip >= 1

        with torch.no_grad():
            if verbose:
                tstart = torch.cuda.Event(enable_timing=True)
                tend = torch.cuda.Event(enable_timing=True)
                tstart.record()

            data = data.reshape(len(data), *self.shape)
            sample = sample.reshape(len(sample), *self.shape)
            data = Shift(data, self.shift)
            sample = Shift(sample, self.shift)

            data0 = torch.zeros(len(data), self.Nkernel_x, self.Nkernel_y, self.shape[-1], device=data.device)
            sample0 = torch.zeros(len(sample), self.Nkernel_x, self.Nkernel_y, self.shape[-1], device=sample.device)
            for j in range(self.Nkernel_y):
                for i in range(self.Nkernel_x):
                    data0[:,i,j,:] = torch.mean(data[:, i*self.kernel[0]:(i+1)*self.kernel[0], j*self.kernel[1]:(j+1)*self.kernel[1], :], dim=[1,2])
                    sample0[:,i,j,:] = torch.mean(sample[:, i*self.kernel[0]:(i+1)*self.kernel[0], j*self.kernel[1]:(j+1)*self.kernel[1], :], dim=[1,2])
            data0 = data0.reshape(len(data0), -1) @ self.wT
            sample0 = sample0.reshape(len(sample0), -1) @ self.wT

            SWD = SlicedWasserstein_direction(data0, None, second=sample0, p=MSWD_p)

            #build rational quadratic spline transform using kde
            x, y, deriv = estimate_knots(data0, sample0, interp_nbin=self.interp_nbin, above_noise=(SWD>noise_threshold), edge_bins=edge_bins, derivclip=derivclip, 
                                         extrapolate=extrapolate, alpha=alpha, KDE=KDE, bw_factor_data=bw_factor_data, bw_factor_sample=bw_factor_sample, batchsize=batchsize)

            self.transform1D.set_param(x, y, deriv)

            data = UnShift(data, self.shift)
            sample = UnShift(sample, self.shift)
            data = data.reshape(len(data), -1)
            sample = sample.reshape(len(sample), -1)

            if verbose:
                tend.record()
                torch.cuda.synchronize()
                t = tstart.elapsed_time(tend)
                print ('Fit spline:', 'Time:', t/1000., 'Wasserstein Distance:', SWD.tolist())

            return SWD


    def transform(self, data, mode='forward', param=None):

        data = data.reshape(len(data), *self.shape)
        data = Shift(data, self.shift)
        data0 = torch.zeros(len(data), self.Nkernel_x, self.Nkernel_y, self.shape[-1], device=data.device)
        for j in range(self.Nkernel_y):
            for i in range(self.Nkernel_x):
                data0[:,i,j,:] = torch.mean(data[:, i*self.kernel[0]:(i+1)*self.kernel[0], j*self.kernel[1]:(j+1)*self.kernel[1], :], dim=[1,2])
        
        data1 = data0.reshape(len(data0), -1) @ self.wT
        remaining = data0.reshape(len(data0), -1) - data1 @ self.wT.T
        if mode is 'forward':
            data1, logj = self.transform1D(data1)
        elif mode is 'inverse':
            data1, logj = self.transform1D.inverse(data1)
        logj = torch.sum(logj, dim=1)
        data1 = (remaining + data1 @ self.wT.T).reshape(len(data), self.Nkernel_x, self.Nkernel_y, self.shape[-1])
        delta = data1 - data0
        del data0, data1

        for j in range(self.Nkernel_y):
            for i in range(self.Nkernel_x):
                data[:, i*self.kernel[0]:(i+1)*self.kernel[0], j*self.kernel[1]:(j+1)*self.kernel[1], :] += delta[:,i,j,:].view(len(data), 1, 1, self.shape[-1])

        data = UnShift(data, self.shift)
        data = data.reshape(len(data), -1)

        return data, logj


    def forward(self, data, param=None):
        return self.transform(data, mode='forward', param=param)


    def inverse(self, data, param=None):
        return self.transform(data, mode='inverse', param=param)



def add_one_layer_inverse(model, data, sample, n_component, nsample_wT, nsample_spline, layer_type='regular', shape=None, kernel_size=None, shift=None, interp_nbin=400, MSWD_p=2, MSWD_max_iter=200, edge_bins=10, derivclip=None, extrapolate='regression', alpha=0, noise_threshold=0, KDE=True, bw_factor_data=1, bw_factor_sample=1, batchsize=None, verbose=True, device=torch.device('cuda')):

    assert layer_type in ['regular', 'patch', 'interpatch']
    if layer_type == 'patch' or layer_type == 'interpatch':
        assert shape is not None
        assert kernel_size is not None
        assert shift is not None
    assert nsample_wT <= len(data)
    assert len(sample) >= nsample_wT + nsample_spline

    t = time.time()

    sample_device = sample.device
    sample = sample[torch.randperm(sample.shape[0])]
    if layer_type == 'regular':
        layer = SlicedTransport(ndim=model.ndim, n_component=n_component, interp_nbin=interp_nbin).requires_grad_(False).to(device)
    elif layer_type == 'patch':
        layer = PatchSlicedTransport(shape=shape, kernel_size=kernel_size, shift=shift, n_component=n_component, interp_nbin=interp_nbin).requires_grad_(False).to(device)
    elif layer_type == 'interpatch':
        layer = InterPatchSlicedTransport(shape=shape, kernel_size=kernel_size, shift=shift, n_component=n_component, interp_nbin=interp_nbin).requires_grad_(False).to(device)

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
        data = data[torch.randperm(data.shape[0])]
        data1 = data[:nsample_spline].to(device)
    sample1 = sample[nsample_wT:nsample_wT+nsample_spline].cuda()
    SWD = layer.fit_spline_inverse(data1, sample1, edge_bins=edge_bins, derivclip=derivclip, extrapolate=extrapolate, alpha=alpha, noise_threshold=noise_threshold,
                                   MSWD_p=MSWD_p, KDE=KDE, bw_factor_data=bw_factor_data, bw_factor_sample=bw_factor_sample, batchsize=batchsize, verbose=verbose)
    del data1, sample1

    if (SWD>noise_threshold).any():

        if batchsize is None:
            sample = layer.inverse(sample.to(device))[0].to(sample_device)
        else:
            j = 0
            while j * batchsize < len(sample):
                sample[j*batchsize:(j+1)*batchsize] = layer.inverse(sample[j*batchsize:(j+1)*batchsize].cuda())[0].cpu()
                j += 1

        model.add_layer(layer, position=0)
    if verbose:
        print ('Nlayer:', len(model.layer), 'Time:', time.time()-t, layer_type)
        print ()

    return model, sample
