import numpy as np
import torch
from FBSNNs_PINN import FBSNN
import matplotlib.pyplot as plt
from torch import nn
import argparse
import pandas as pd

class MLP(nn.Module):
    def __init__(self, layers:list, D):
        super(MLP, self).__init__()
        models = []
        for i in range(len(layers)-1):
            models.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                models.append(nn.Tanh())
        self.nn = nn.Sequential(*models)
        self.D = D
    def forward(self, x):
        return self.nn(x)
        """X, t = x[:, 1:], x[:, :1]
        return self.nn(x) * (t - 0.3) + torch.arctan(torch.max(X, 1, keepdim=True)[0])"""

class Semilinear_Heat(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers, device, method_PINN, batch_size_PINN):
        super().__init__(Xi, T, M, N, D, layers, device, method_PINN, batch_size_PINN, "Heat")
        self.model = MLP(layers, D).to(device)
        if D == 10:
            self.u_exact = 0.47006
        elif D == 100:
            self.u_exact = 0.31674
        elif D == 1000:
            self.u_exact = 0.28753
        elif D == 10000:
            self.u_exact = 0.28433
    
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return -(1 - Y**2) / (1 + Y**2) # M x 1
    
    def g_tf(self, X): # M x D
        #return torch.sum(X, 1, keepdims=True) # M x 1
        return 5 / (10 + 2 * torch.sum(X**2, 1, keepdims=True)) # M x 1

    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z) # M x D
    
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        return np.sqrt(2.0) * super().sigma_tf(t, X, Y) # M x D x D
    
    def neural_net(self, X):
        return self.model(X)
    
    def net_u(self, t, X): # M x 1, M x D
        X.requires_grad_()
        u = self.neural_net(torch.concat([t,X], 1)) # M x 1
        Du = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
        return u, Du

    def residual(self, t, X):
        t.requires_grad_()
        u = self.neural_net(torch.concat([t,X], 1)) # M x 1
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0] # M x 1
        X.requires_grad_()
        u = self.neural_net(torch.concat([t,X], 1)) # M x 1
        u_x = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
        u_xx = 0
        for i in range(self.D):
            d2f_dxidxi = torch.autograd.grad(u_x[:, i].sum(), X, create_graph=True)[0][:, i]
            u_xx += d2f_dxidxi
        u = u.reshape(-1)
        residual_pred = u_t.reshape(-1) + (u_xx + (1 - u**2) / (1 + u**2))
        return residual_pred.square().mean()

    def residual_1(self, t, X):
        t.requires_grad_()
        u = self.neural_net(torch.concat([t,X], 1)) # M x 1
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0] # M x 1
        X.requires_grad_()
        u = self.neural_net(torch.concat([t,X], 1)) # M x 1
        u_x = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
        u = u.reshape(-1)
        residual_pred = (u_t.reshape(-1) + (1 - u**2) / (1 + u**2)).detach()
        gradient_j = u_t.reshape(-1) + (1 - u**2) / (1 + u**2)
        idx = np.random.choice(self.D, self.batch_size, replace=False)
        for i in range(self.D):
            d2f_dxidxi = torch.autograd.grad(u_x[:, i].sum(), X, create_graph=True)[0][:, i]
            residual_pred += d2f_dxidxi.detach()
            if i in idx:
                gradient_j += self.D / self.batch_size * d2f_dxidxi
        loss = (2 * residual_pred.detach() * gradient_j).mean()
        return loss

    def residual_2(self, t, X):
        t.requires_grad_()
        u = self.neural_net(torch.concat([t,X], 1)) # M x 1
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0] # M x 1
        X.requires_grad_()
        u = self.neural_net(torch.concat([t,X], 1)) # M x 1
        u_x = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
        u = u.reshape(-1)
        residual_pred = (u_t.reshape(-1) + (1 - u**2) / (1 + u**2)).detach()
        gradient_j = u_t.reshape(-1) + (1 - u**2) / (1 + u**2)
        idx = np.random.choice(self.D, self.batch_size, replace=False)
        for i in (idx):
            d2f_dxidxi = torch.autograd.grad(u_x[:, i].sum(), X, create_graph=True)[0][:, i]
            residual_pred += self.D / self.batch_size * d2f_dxidxi.detach()
        idx = np.random.choice(self.D, self.batch_size, replace=False)
        for i in (idx):
            d2f_dxidxi = torch.autograd.grad(u_x[:, i].sum(), X, create_graph=True)[0][:, i]
            gradient_j += self.D / self.batch_size * d2f_dxidxi
        loss = (2 * residual_pred.detach() * gradient_j).mean()
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FBSNN Training for BSB Equaions')
    parser.add_argument('--SEED', type=int, default=0)
    parser.add_argument('--dim', type=int, default=10) # dimension of the problem.
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--epochs', type=int, default=10000) # Adam epochs
    parser.add_argument('--lr', type=float, default=1e-3) # Adam lr
    parser.add_argument('--NN_h', type=int, default=1024) # width of NN
    parser.add_argument('--NN_L', type=int, default=4) # depth of NN
    parser.add_argument('--save_loss', type=bool, default=True) # save the optimization trajectory?
    parser.add_argument('--M', type=int, default=100) # number of trajectories (batch size)
    parser.add_argument('--N', type=int, default=int(20)) # number of time snapshots
    parser.add_argument('--batch_size_PINN', type=int, default=int(10)) # 
    parser.add_argument('--method_PINN', type=int, default=0) # 
    args = parser.parse_args()
    print(args)
    device = torch.device(args.device)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    np.random.seed(args.SEED)
    
    M = args.M # number of trajectories (batch size)
    N = args.N # number of time snapshots
    D = args.dim # number of dimensions
    
    layers = [D + 1] + (args.NN_L - 1) * [args.NN_h] + [1]
    #layers = [D + 1] + (args.NN_L - 1) * [D + 100] + [1]
    print(layers)

    Xi = np.zeros([1, D])
    T = 0.3
         
    # Training
    model = Semilinear_Heat(Xi, T, M, N, D, layers, device, args.method_PINN, args.batch_size_PINN)
        
    model.train(N_Iter = args.epochs, learning_rate=1e-3)
    
    model.saved_loss = np.asarray(model.saved_loss)
    model.saved_l2 = np.asarray(model.saved_l2)

    info_dict = {"loss": model.saved_loss, "L2": model.saved_l2}
    df = pd.DataFrame(data=info_dict, index=None)
    df.to_excel(
        "Heat_FBSNN_PINN_"+str(args.dim)+"_"+str(args.SEED)+".xlsx",
        index=False
    )