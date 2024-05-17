import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PINN Project')
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument('--dim', type=int, default=100 + 1) # dimension of the problem.
parser.add_argument('--dataset', type=str, default="HJB_Rosenbrock")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--epochs', type=int, default=10000) # Adam epochs
parser.add_argument('--lr', type=float, default=1e-3) # Adam lr
parser.add_argument('--PINN_h', type=int, default=1024) # width of PINN
parser.add_argument('--K', type=int, default=10) # depth of PINN
parser.add_argument('--save_loss', type=bool, default=True) # save the trajectory or not
parser.add_argument('--MC', type=int, default=100) # use scheduler?
parser.add_argument('--batch_size', type=int, default=100) # batch size over PDE terms
parser.add_argument('--batch_size_pgd', type=int, default=10) # batch size over PDE terms
parser.add_argument('--N_f', type=int, default=int(100)) # num of residual points
parser.add_argument('--N_test', type=int, default=int(20000)) # num of residual points
parser.add_argument('--method', type=float, default=3) # method for PINN
parser.add_argument('--method_adv', type=float, default=3) # method for the loss in adv training

args = parser.parse_args()
print(args)
np.random.seed(args.SEED)
assert args.dataset == "HJB_Rosenbrock"

coeffs = (np.random.rand(2, args.dim - 2)) + 0.5
def load_data_HJB_Rosenbrock(d):
    args.input_dim = d
    args.output_dim = 1

    N_test = args.N_test

    MC = args.MC
    W = np.random.randn(MC, 1, args.dim - 1) # MC x NC x D
    def g(X):
        temp = coeffs[None, 0:1, :] * (X[:, :, :-1] - X[:, :, 1:])**2 + coeffs[None, 1:2, :] * X[:, :, 1:]**2
        return np.log(1 + np.sum(temp, -1))
    def exp_u_exact(t, x):
        T = 1
        return (np.mean(np.exp(-g(x + np.sqrt(2.0*np.abs(T-t))*W)),0))
    def u_exact(t, x): # NC x 1, NC x D
        T = 1
        return -np.log(np.mean(np.exp(-g(x + np.sqrt(2.0*np.abs(T-t))*W)),0))
    K = args.K

    t = np.random.rand(1, N_test, 1)
    x = np.random.randn(1, N_test, d - 1)
    x = np.concatenate([x, t], axis=-1)
    u = 0
    for _ in tqdm(range(K)):
        u += exp_u_exact(x[:, :, -1:], x[:, :, :-1]) / K
    u = -np.log(u)
    x = x.squeeze()
    return x, u

x, u = load_data_HJB_Rosenbrock(d=args.dim)
print(x.shape, u.shape, coeffs.shape)
np.savetxt("HJB_Rosenbrock_Data/x" + str(args.dim-1) + ".txt", x)
np.savetxt("HJB_Rosenbrock_Data/u" + str(args.dim-1) + ".txt", u)
np.savetxt("HJB_Rosenbrock_Data/coeffs" + str(args.dim-1) + ".txt", coeffs)
