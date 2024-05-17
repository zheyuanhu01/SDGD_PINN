import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import argparse
import sympy as sy
from tqdm import tqdm
import time
import pandas as pd

parser = argparse.ArgumentParser(description='PINN Project')
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument('--dim', type=int, default=100 + 1) # dimension of the problem.
parser.add_argument('--dataset', type=str, default="HJB_Rosenbrock")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--epochs', type=int, default=10000) # Adam epochs
parser.add_argument('--lr', type=float, default=1e-3) # Adam lr
parser.add_argument('--PINN_h', type=int, default=1024) # width of PINN
parser.add_argument('--PINN_L', type=int, default=4) # depth of PINN
parser.add_argument('--save_loss', type=bool, default=True) # save the trajectory or not
parser.add_argument('--use_sch', type=int, default=1) # use scheduler?
parser.add_argument('--batch_size', type=int, default=100) # batch size over PDE terms
parser.add_argument('--batch_size_pgd', type=int, default=10) # batch size over PDE terms
parser.add_argument('--N_f', type=int, default=int(100)) # num of residual points
parser.add_argument('--N_test', type=int, default=int(20000)) # num of residual points
parser.add_argument('--method', type=float, default=3) # method for PINN
parser.add_argument('--method_adv', type=float, default=3) # method for the loss in adv training

args = parser.parse_args()
print(args)
device = torch.device(args.device)
torch.manual_seed(args.SEED)
torch.cuda.manual_seed(args.SEED)
np.random.seed(args.SEED)
assert args.dataset == "HJB_Rosenbrock"

coeffs = np.loadtxt("HJB_Rosenbrock_Data/coeffs" + str(args.dim-1) + ".txt")# (np.random.rand(2, args.dim - 2)) + 0.5
def load_data_HJB_Rosenbrock(d):
    args.input_dim = d
    args.output_dim = 1

    N_f = args.N_f # Number of collocation points
    
    xf = np.concatenate([np.random.randn(N_f, d - 1), np.random.rand(N_f, 1)], axis=-1)
    ff = np.zeros(N_f)

    x = np.loadtxt("HJB_Rosenbrock_Data/x" + str(args.dim-1) + ".txt")
    u = np.loadtxt("HJB_Rosenbrock_Data/u" + str(args.dim-1) + ".txt")
    return x, u, xf, ff

x, u, xf, ff = load_data_HJB_Rosenbrock(d=args.dim)
print(x.shape, u.shape, xf.shape, ff.shape)

coeffs = torch.FloatTensor(coeffs).to(device)
class MLP(nn.Module):
    def __init__(self, layers:list):
        super(MLP, self).__init__()
        models = []
        for i in range(len(layers)-1):
            models.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                models.append(nn.Tanh())
        self.nn = nn.Sequential(*models)
    def forward(self, x):
        X = x[:, :-1]
        temp = coeffs[0:1, :] * (X[:, :-1] - X[:, 1:])**2 + coeffs[1:2, :] * X[:, 1:]**2
        return self.nn(x) * (1 - x[:, -1:]) + \
            torch.log(1 + torch.sum(temp, 1, keepdims=True))

class PINN:
    def __init__(self):
        self.epoch = args.epochs
        self.verbose = 1
        self.adam_lr = args.lr
        self.dim = args.dim
        self.batch_size = args.batch_size
        self.batch_size_pgd = args.batch_size_pgd
        self.x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
        self.u = torch.tensor(u, dtype=torch.float32, requires_grad=True).to(device).reshape(-1, 1)
        self.xf = torch.tensor(xf, dtype=torch.float32, requires_grad=True).to(device)
        self.ff = torch.tensor(ff, dtype=torch.float32, requires_grad=False).to(device).reshape(-1, 1)
        # Initalize Neural Networks
        layers = [args.input_dim] + [args.PINN_h] * (args.PINN_L - 1) + [args.output_dim]
        self.u_net = MLP(layers).to(device)  
        self.net_params_pinn = list(self.u_net.parameters())
        self.saved_loss = []
        self.saved_l2 = []

    def Resample_HJB_Rosenbrock(self):
        self.xf = torch.cat([torch.randn(args.N_f, args.dim - 1), torch.rand(args.N_f, 1)], dim=-1).to(device).requires_grad_()

    def HJB_Rosenbrock(self): 
        x, t = self.xf[:, :-1], self.xf[:, -1:]
        t.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.u_net(X)
        u_t = torch.autograd.grad(f.sum(), t, create_graph=True)[0]
        x.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.u_net(X)
        u_x = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        u_xx = []
        for i in range(self.dim - 1):
            d2f_dxidxi = torch.autograd.grad(u_x[:, i].sum(), x, create_graph=True)[0][:, i:i+1]
            u_xx.append(d2f_dxidxi)
        # (batch_size, x_dim)
        u_xx = torch.concat(u_xx, dim=1)
        residual_pred = u_t.reshape(-1) + torch.sum(u_xx, dim=1) - torch.sum(u_x**2, dim=1)
        return residual_pred

    def pgd(self, step_cnt=20, step_size=0.05, t_lower_bound=0.0, t_upper_bound=1.0):
        for _ in range(step_cnt):
            self.xf.requires_grad_()
            if args.method_adv == 0:
                loss = self.get_loss_pinn()
            elif args.method_adv == 1:
                loss, _ = self.get_loss_pinn_stochastic_HJB_Rosenbrock(self.batch_size_pgd)#self.get_loss_pinn()
            elif args.method_adv == 2:
                loss, _ = self.get_loss_pinn_stochastic2_HJB_Rosenbrock(self.batch_size_pgd)
            elif args.method_adv == 3:
                loss, _ = self.get_loss_pinn_stochastic3_HJB_Rosenbrock(self.batch_size_pgd)
            grad = torch.autograd.grad(loss, [self.xf])[0]
            self.xf = self.xf.detach() + step_size * torch.sign(grad.detach())
            self.xf[:,-1] = torch.clamp(self.xf[:,-1], t_lower_bound, t_upper_bound)
        return

    def get_loss_pinn_stochastic_HJB_Rosenbrock(self, bs):
        x, t = self.xf[:, :-1], self.xf[:, -1:]
        t.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.u_net(X)
        u_t = torch.autograd.grad(f.sum(), t, create_graph=True)[0]

        x.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.u_net(X)
        u_x = torch.autograd.grad(f.sum(), x, create_graph=True)[0]

        residual_pred = (u_t.reshape(-1) - torch.sum(u_x**2, dim=1)).detach()
        gradient_j = u_t.reshape(-1) - torch.sum(u_x**2, dim=1)

        idx = np.random.choice(self.dim - 1, bs, replace=False)
        for i in range(self.dim - 1):
            d2f_dxidxi = torch.autograd.grad(u_x[:, i].sum(), x, create_graph=True)[0][:, i]
            residual_pred += d2f_dxidxi.detach()
            if i in idx:
                gradient_j += (args.dim - 1) / bs * d2f_dxidxi

        loss = (2 * residual_pred.detach() * gradient_j).mean()
        saeved_loss = (residual_pred - self.ff).square().mean()
        return loss, saeved_loss

    def get_loss_pinn_stochastic2_HJB_Rosenbrock(self, bs):
        x, t = self.xf[:, :-1], self.xf[:, -1:]
        t.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.u_net(X)
        u_t = torch.autograd.grad(f.sum(), t, create_graph=True)[0]

        x.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.u_net(X)
        u_x = torch.autograd.grad(f.sum(), x, create_graph=True)[0]

        residual_pred = (u_t.reshape(-1) - torch.sum(u_x**2, dim=1)).detach()
        gradient_j = u_t.reshape(-1) - torch.sum(u_x**2, dim=1)

        idx = np.random.choice(self.dim - 1, bs, replace=False)
        for i in idx:
            d2f_dxidxi = torch.autograd.grad(u_x[:, i].sum(), x, create_graph=True)[0][:, i]
            residual_pred += (args.dim - 1) / bs * d2f_dxidxi.detach()
        idx = np.random.choice(self.dim - 1, bs, replace=False)
        for i in idx:
            d2f_dxidxi = torch.autograd.grad(u_x[:, i].sum(), x, create_graph=True)[0][:, i]
            gradient_j += (args.dim - 1) / bs * d2f_dxidxi

        loss = (2 * residual_pred.detach() * gradient_j).mean()
        saeved_loss = (residual_pred - self.ff).square().mean()

        return loss, saeved_loss
    
    def get_loss_pinn_stochastic3_HJB_Rosenbrock(self, bs):
        x, t = self.xf[:, :-1], self.xf[:, -1:]
        t.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.u_net(X)
        u_t = torch.autograd.grad(f.sum(), t, create_graph=True)[0]

        x.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.u_net(X)
        u_x = torch.autograd.grad(f.sum(), x, create_graph=True)[0]

        gradient_j = u_t.reshape(-1) - torch.sum(u_x**2, dim=1)

        idx = np.random.choice(self.dim - 1, bs, replace=False)
        for i in idx:
            d2f_dxidxi = torch.autograd.grad(u_x[:, i].sum(), x, create_graph=True)[0][:, i]
            gradient_j += (args.dim - 1) / bs * d2f_dxidxi

        loss = (gradient_j - self.ff).square().mean()

        return loss, loss

    def num_params(self):
        num_pinn = 0
        for p in self.net_params_pinn:
            num_pinn += len(p.reshape(-1))
        return num_pinn

    def get_loss_pinn(self):
        f = getattr(self, args.dataset)()
        mse_f = (f - self.ff).square().mean()
        return mse_f
    
    def train_adam(self):
        optimizer = torch.optim.Adam(self.net_params_pinn, lr=self.adam_lr)
        lr_lambda = lambda epoch: 1-epoch/args.epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        for n in tqdm(range(self.epoch)):
            self.pgd()#(step_cnt=100, step_size=0.01)
            if args.method == 0:
                assert args.batch_size == args.dim
                loss = self.get_loss_pinn()
                saved_loss = loss
            elif args.method == 1:
                assert args.batch_size < args.dim
                loss, saved_loss = getattr(self, "get_loss_pinn_stochastic_" + args.dataset)(args.batch_size)
            elif args.method == 2:
                assert args.batch_size < args.dim
                loss, saved_loss = getattr(self, "get_loss_pinn_stochastic2_" + args.dataset)(args.batch_size)
            elif args.method == 3:
                assert args.batch_size < args.dim
                loss, saved_loss = getattr(self, "get_loss_pinn_stochastic3_" + args.dataset)(args.batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.use_sch:
                scheduler.step()
            current_loss = saved_loss.item()
            self.Resample_HJB_Rosenbrock()
            if n%100==0:
                current_L2 = self.L2_pinn()
                print('epoch %d, loss: %e, l2: %e'%(n, current_loss, current_L2))
            if args.save_loss:
                self.saved_loss.append(current_loss)
                if n % 100 != 0:
                    current_L2 = self.L2_pinn()
                self.saved_l2.append(current_L2)
       
    def predict_pinn(self):
        with torch.no_grad():
            u_pred = self.u_net(self.x)
        return u_pred
    
    def L2_pinn(self):
        pinn_u_pred_20 = (self.u - self.predict_pinn())#.detach().cpu().numpy()
        pinn_error_u_total_20 = torch.norm(pinn_u_pred_20) / torch.norm(self.u)
        return pinn_error_u_total_20.item()

model = PINN()
print("Num params:", model.num_params())
model.train_adam()

if args.save_loss:
    model.saved_loss = np.asarray(model.saved_loss)
    model.saved_l2 = np.asarray(model.saved_l2)

    info_dict = {"loss": model.saved_loss, "L2": model.saved_l2}
    df = pd.DataFrame(data=info_dict, index=None)
    df.to_excel(
        "saved_loss_l2/"+args.dataset+"_Linf_dim="+str(args.dim)+\
            "_batch_pgd="+str(args.batch_size_pgd)+\
            "_method_pgd="+str(args.method_adv)+\
            "_batch="+str(args.batch_size)+\
            "_method="+str(args.method)+\
            "_SEED="+str(args.SEED)+".xlsx",
        index=False
    )