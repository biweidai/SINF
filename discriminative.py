import torch
from RQspline import RQspline, estimate_knots_gaussian
from SlicedWasserstein import Stiefel_SGD
import time
import copy

def estimate_delta_logp(data, wT, alldata, interp_nbin=50, alpha=(0,0.9), edge_bins=0, derivclip=None, extrapolate='regression', KDE=True, bw_factor=1):
    
    alldata0 = alldata @ wT
    data0 = data @ wT
    #alldata0 = torch.sort(alldata0, dim=0)[0]

    x, y, deriv = estimate_knots_gaussian(alldata0, interp_nbin, above_noise=torch.ones(wT.shape[1], dtype=torch.bool), edge_bins=edge_bins, 
                                          derivclip=derivclip, extrapolate=extrapolate, alpha=alpha, KDE=KDE, bw_factor=bw_factor)
    
    transform1D = RQspline(wT.shape[1], interp_nbin).to(data.device)
    transform1D.set_param(x, y, deriv)
    data1, logj = transform1D(data0)
    delta_logp = torch.sum(logj, dim=1) + torch.sum(data0**2, dim=1)/2. - torch.sum(data1**2, dim=1)/2. 

    return delta_logp 


def mean_delta_logp(wT, data, data_falselabel, misclassify, label_true, label_false, Nlabel):
    
    delta_logp = 0
    data_mis = data[misclassify]
    for label in range(Nlabel):
        select_mis = (label_true[misclassify] == label)
        select = (label_true == label)
        if torch.sum(select_mis) > 0:
            delta_logp += torch.sum(estimate_delta_logp(data_mis[select_mis], wT, data[select]))
        if label_false is not None:
            select_mis = (label_false[misclassify] == label)
            if torch.sum(select_mis) > 0:
                delta_logp -= torch.sum(estimate_delta_logp(data_falselabel[select_mis], wT, data[select]))
    return delta_logp / len(data_mis)


def delta_logp_discriminative(data, data_falselabel, misclassify, label_true, label_false, n_component, Nlabel=10):

    ndim = data.shape[1]
    if n_component is None:
        n_component = ndim

    #initialize w. algorithm from https://arxiv.org/pdf/math-ph/0609050.pdf
    wi = torch.randn(ndim, n_component, device=data.device)
    Q, R = torch.qr(wi)
    L = torch.sign(torch.diag(R))
    w = (Q * L).T

    lr = 0.1
    down_fac = 0.5
    up_fac = 1.5
    c = 0.5
    maxiter = 200
    eps = 1e-6

    #algorithm from http://noodle.med.yale.edu/~hdtag/notes/steifel_notes.pdf
    #note that here w = X.T
    #use backtracking line search
    w1 = w.clone()

    for i in range(maxiter):
        w.requires_grad_(True)
        loss = -mean_delta_logp(w.T, data, data_falselabel, misclassify, label_true, label_false, Nlabel=Nlabel) 
        loss1 = loss
        GT = torch.autograd.grad(loss, w)[0]
        w.requires_grad_(False)
        with torch.no_grad():
            WT = w.T @ GT - GT.T @ w
            e = - w @ WT #dw/dlr
            m = torch.sum(GT * e) #dloss/dlr

            lr /= down_fac
            while loss1 > loss + c*m*lr:
                lr *= down_fac
                if 2*n_component < ndim:
                    UT = torch.cat((GT, w), dim=0).double()
                    V = torch.cat((w.T, -GT.T), dim=1).double()
                    w1 = (w.double() - lr * w.double() @ V @ torch.pinverse(torch.eye(2*n_component, dtype=torch.double, device=data.device)+lr/2*UT@V) @ UT).to(torch.get_default_dtype())
                else:
                    w1 = (w.double() @ (torch.eye(ndim, dtype=torch.double, device=data.device)-lr/2*WT.double()) @ torch.pinverse(torch.eye(ndim, dtype=torch.double, device=data.device)+lr/2*WT.double())).to(torch.get_default_dtype())

                loss1 = -mean_delta_logp(w1.T, data, data_falselabel, misclassify, label_true, label_false, Nlabel=Nlabel) 
                #print(loss1, lr)

            if torch.max(torch.abs(w1-w)) < eps:
                w = w1
                break

            lr *= up_fac
            loss = loss1
            w = w1
    return w.T


def loss_(layer, data, label, logj, nclass=10, margin=10, L2=0):
    
    logp = torch.zeros_like(logj)
    reg = 0 
    for i in range(nclass):
        data1, logj1 = layer(data[i], param=torch.ones(data.shape[1], dtype=torch.int, device=data.device)*i)
        logp[i] = logj[i] + logj1 - torch.sum(data1**2, dim=1)/2.
        x, y, deriv = layer.transform1D[i]._prepare()
        reg += L2 * torch.sum((y-x)**2 + (deriv-1)**2)

    logp_true = logp[label, torch.arange(logp.shape[1], device=logp.device)]
    delta_logp = logp_true - logp - margin
    delta_logp[delta_logp>0] = 0

    return -torch.mean(delta_logp) - margin/nclass + reg


def train_discriminative(layer, optimizer_ortho, optimizer_spline, data, label, logj, maxepoch, batchsize, nclass=10, margin=10, L2=0, quiet=False, data_validate=None, label_validate=None, logj_validate=None):

    with torch.no_grad():
        train_losses = [loss_(layer, data, label, logj, nclass, margin).item()]
        if data_validate is None:
            best_loss = train_losses[-1]
        else:
            validate_losses = [loss_(layer, data_validate, label_validate, logj_validate, nclass, margin).item()]
            best_loss = validate_losses[-1]
        best_p = copy.deepcopy(layer.state_dict()) 
    if not quiet:
        if data_validate is None:
            print(f'Epoch 0, Train loss {train_losses[-1]:.4f}')
        else:
            print(f'Epoch 0, Train loss {train_losses[-1]:.4f}, Validate loss {validate_losses[-1]:.4f}')

    wait = 0
    maxwait = 1
    for epoch in range(maxepoch):    
        t = time.time()
        if batchsize is None:
            loss = loss_(layer, data, label, logj, nclass, margin, L2)
            optimizer_ortho.zero_grad()
            optimizer_spline.zero_grad()
            loss.backward()
            optimizer_ortho.step()
            optimizer_spline.step()
            train_losses.append(loss.item())
        else:
            with torch.no_grad():
                perm = torch.randperm(data.shape[1], device=data.device)
                data = data[:,perm]
                label = label[perm]
                logj = logj[:,perm]
            i = 0
            start = i*batchsize
            end = (i+1)*batchsize
            while end <= data.shape[1]:
                loss = loss_(layer, data[:,start:end], label[start:end], logj[:,start:end], nclass, margin, L2)
                optimizer_ortho.zero_grad()
                optimizer_spline.zero_grad()
                loss.backward()
                optimizer_ortho.step()
                optimizer_spline.step()
                i += 1
                start = i*batchsize
                end = (i+1)*batchsize
            with torch.no_grad():
                train_losses.append(loss_(layer, data, label, logj, nclass, margin).item())
        
        with torch.no_grad():
            if data_validate is None:
                if train_losses[-1] < best_loss:
                    best_loss = train_losses[-1]
                    best_p = copy.deepcopy(layer.state_dict()) 
                    wait = 0
                else:
                    wait += 1
            else:
                validate_losses.append(loss_(layer, data_validate, label_validate, logj_validate, nclass, margin).item())
                if validate_losses[-1] < best_loss:
                    best_loss = validate_losses[-1]
                    best_p = copy.deepcopy(layer.state_dict()) 
                    wait = 0
                else:
                    wait += 1

        t = time.time() - t
        if not quiet:
            if data_validate is None:
                print(f'Epoch {epoch+1}, Train loss {train_losses[-1]:.4f}, Time {t:.3f} s, Best loss {best_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}, Train loss {train_losses[-1]:.4f}, Validate loss {validate_losses[-1]:.4f}, Time {t:.3f} s, Best loss {best_loss:.4f}')
        if wait >= maxwait:
            break

    with torch.no_grad():
        layer.load_state_dict(best_p)

    return train_losses


