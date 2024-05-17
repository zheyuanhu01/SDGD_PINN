import numpy as np
import torch
import time
from abc import ABC, abstractmethod
from tqdm import tqdm
import pandas as pd

class FBSNN(ABC): # Forward-Backward Stochastic Neural Network
    def __init__(self, Xi, T, M, N, D, layers, device, PINN_method, batch_size_PINN, PDE):
        self.PDE = PDE

        self.Xi = torch.from_numpy(Xi).float().to(device) # initial point
        self.T = T # terminal time

        self.M = M # number of trajectories
        self.N = N # number of time snapshots
        self.D = D # number of dimensions

        # layers
        self.layers = layers # (D+1) --> 1
        self.device = device
        self.PINN_method = PINN_method
        self.batch_size = batch_size_PINN

        """self.x = np.concatenate([np.random.rand(N_test, 1), np.random.randn(N_test, D)], axis=-1)
        self.x = torch.from_numpy(self.x).float().to(device)
        self.u = self.u_exact(self.x[:, :1], self.x[:, 1:]).reshape(-1)"""
        self.x = np.concatenate([np.zeros((1, 1)), Xi], axis=-1)
        self.x = torch.from_numpy(self.x).float().to(device)

        self.saved_loss, self.saved_l2, self.saved_l1 = [], [], []

    def neural_net(self, X):
        return
    
    def net_u(self, t, X): # M x 1, M x D
        return

    def Dg_tf(self, X): # M x D
        X.requires_grad_()
        u = self.g_tf(X)
        Du = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
        return Du # M x D
        
    def loss_function(self, t, W, Xi): # M x (N+1) x 1, M x (N+1) x D, 1 x D
        loss = 0
        residual_loss = 0
        X_list = []
        Y_list = []
        
        t0 = t[:, 0, :]
        W0 = W[:, 0, :]
        X0 = torch.tile(Xi, [self.M, 1]) # M x D
        Y0, Z0 = self.net_u(t0, X0) # M x 1, M x D
        
        X_list.append(X0)
        Y_list.append(Y0) 
        
        for n in range(0, self.N):
            t1 = t[:, n+1, :]
            W1 = W[:, n+1, :]
            # sigma: M X D X D; W1 - W0: M X D
            temp = self.sigma_tf(t0, X0, Y0) * (W1 - W0)
            X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + temp
            Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.sum(Z0 * temp, 1, keepdims=True)
            Y1, Z1 = self.net_u(t1, X1)
            if self.PINN_method == 0:
                residual_loss += self.residual(t1, X1)
            elif self.PINN_method == 1:
                residual_loss += self.residual_1(t1, X1)
            elif self.PINN_method == 2:
                residual_loss += self.residual_2(t1, X1)
            
            loss += torch.nn.MSELoss()(Y1, Y1_tilde)
            
            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1
            
            X_list.append(X0)
            Y_list.append(Y0)
            
        loss += torch.nn.MSELoss()(Y1, self.g_tf(X1))
        loss += torch.nn.MSELoss()(Z1, self.Dg_tf(X1))
        
        residual_loss += 20 * torch.nn.MSELoss()(Y1, self.g_tf(X1)) + torch.nn.MSELoss()(Z1, self.Dg_tf(X1))

        X = torch.stack(X_list, 1)
        Y = torch.stack(Y_list, 1)
        
        return loss, residual_loss, X, Y, Y[0, 0, 0]

    def fetch_minibatch(self):
        T = self.T
        
        M = self.M
        N = self.N
        D = self.D
        
        Dt = np.zeros((M, N+1, 1)) # M x (N+1) x 1
        DW = np.zeros((M, N+1, D)) # M x (N+1) x D
        
        dt = T / N
        
        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))
        
        t = np.cumsum(Dt, axis=1) # M x (N+1) x 1
        W = np.cumsum(DW, axis=1) # M x (N+1) x D

        t, W = torch.from_numpy(t).float().to(self.device), torch.from_numpy(W).float().to(self.device)
        
        return t, W
    
    def train(self, N_Iter, learning_rate):
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9995)
        
        start_time = time.time()
        for it in tqdm(range(N_Iter)):
            
            t_batch, W_batch = self.fetch_minibatch() # M x (N+1) x 1, M x (N+1) x D
            
            loss, residual_loss, _, _, _ = self.loss_function(t_batch, W_batch, self.Xi)

            self.optimizer.zero_grad()
            #loss.backward()
            (0.00 * loss + 1.0 * residual_loss).backward()
            self.optimizer.step()
            self.scheduler.step()

            # Print
            elapsed = time.time() - start_time
            error2 = self.test()
            if it % 100 == 0:
                print("Train Loss %e, Residual Loss %e, Relative L2 Error %e"%(loss, residual_loss, error2))
            start_time = time.time()
            self.saved_loss.append(loss.item())
            self.saved_l2.append(error2)
            self.saved_l1.append(error2)
    
    def test(self):
        Y_pred = self.neural_net(self.x).reshape(-1).item()
        error2 = np.abs(self.u_exact - Y_pred) / np.abs(self.u_exact)
        error1 = error2
        # print("Relative L2 Error ", error)
        return error1
    
    @abstractmethod
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        pass # M x1
    
    @abstractmethod
    def g_tf(self, X): # M x D
        pass # M x 1
    
    @abstractmethod
    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        M = self.M
        D = self.D
        return torch.zeros([M, D]).to(self.device) # M x D
    
    @abstractmethod
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        M = self.M
        D = self.D
        return torch.ones([M, D]).to(self.device) # M x D
    ###########################################################################