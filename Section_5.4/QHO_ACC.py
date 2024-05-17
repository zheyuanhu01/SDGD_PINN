import torch
import torch.nn as nn
import torch.optim as optim
from quadrature import *
from integration import *
from tnn import *
import os
import pandas as pd
from tqdm import tqdm

# ********** eigenvalue problem of quantum harmonic oscillator **********


pi = 3.14159265358979323846

# ********** choose data type and device **********
dtype = torch.double
# dtype = torch.float
# device = 'cpu'
device = 'cuda'


# ********** generate data points **********
# computation domain: [a,b]^dim
a = -5
b = 5
dim = 1000
batch_size = 1000
# quadrature rule:
# number of quad points
quad = 16
# number of partitions for [a,b]
n = 10
# quad ponits and quad weights.
x, w = composite_quadrature_1d(quad, a, b, n, device=device, dtype=dtype)
N = len(x)

# ********** create a neural network model **********
p = 1
sizes = [1, 100, 100, 100, p]
# activation = TNN_Tanh
activation = TNN_Sin
# activation = TNN_ReQU
# activation = TNN_Sigmoid

# define forced boundary condition function.
def bd(x):
    return (x-a)*(b-x)
    # return torch.sin(pi*x)
# define derivative of forced boundary condition function.
def grad_bd(x):
    return -2*x+a+b
    # return pi*torch.cos(pi*x)

model = TNN(dim,sizes,activation,bd=bd,grad_bd=grad_bd,scaling=False).to(dtype).to(device)
print(model)
print(dim, batch_size)


# ********** define loss function **********
def criterion(model, w, x):
    phi, grad_phi = model(w,x,need_grad=1)
    A = torch.sum(torch.sum(((w*grad_phi)@grad_phi.transpose(1,2)),dim=0))
    A += torch.sum(torch.sum(((w*x**2*phi)@phi.transpose(1,2)),dim=0))
    loss = torch.sum(A)
    return loss

def criterion_ACC(model, w, x, dims):
    phi, grad_phi = model.forward_SGD(w,x,dims)
    alpha = torch.ones(p,device=device,dtype=dtype)
    A = torch.sum(torch.sum(((w*grad_phi)@grad_phi.transpose(1,2)),dim=0))
    A += torch.sum(torch.sum(((w*x**2*phi)@phi.transpose(1,2)),dim=0))
    loss = torch.sum(A)
    return loss

# exact eigenvalue
exactlam = dim
# ********** training process **********
# parameters
lr = 1e-3
epochs = 50000
print_every = 1
save = False
# optimizer used
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

ERRORLAM, ERROR0, ERROR1 = [], [], []
# training
for e in tqdm(range(epochs)):
    optimizer.zero_grad()
    if dim == batch_size:
        loss = criterion(model, w, x)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss1 = loss.item()
    else:
        loss1 = 0
        for i in range(dim // batch_size):
            dims = torch.arange(batch_size) + i * batch_size
            loss = criterion_ACC(model, w, x, dims)
            # optimization process
            loss = loss / dim * batch_size
            loss1 += loss.item()
            loss.backward()
        optimizer.step()
        scheduler.step()

    # post process
    if (e + 1) % print_every == 0 or e == 0:
        errorlam = (loss1 - exactlam) / exactlam
        if e % 1000 == 0:

            print('*'*40)
            print('{:<9}{:<25}'.format('epoch = ', e+1))
            print('{:<9}{:<25}'.format('loss = ', loss1))
            # print('{:<9}{:}'.format('scaling = ',model.ms['TNN_Scaling'].alpha.data.numpy()))
            # user-defined post-process
            print('{:<9}{:<25}'.format('errorE = ', errorlam))

        ERRORLAM.append(errorlam)
print('*'*40)
print('Done!')
print(ERRORLAM)
info_dict = {"Lam": ERRORLAM}
df = pd.DataFrame(data=info_dict, index=None)
df.to_excel(
    "QHO_dim="+str(dim)+"_batch="+str(batch_size)+".xlsx",
    index=False
)

