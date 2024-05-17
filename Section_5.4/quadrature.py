import torch
import numpy as np

from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss


def quadrature_1d(N, dtype=torch.double, device='cpu'):
    """
    Quadrature points and weights for one-dimensional Gauss-Legendre quadrature rules in computational domain [-1,1].
    
    Parameters:
        N: number of quadrature points in domain [1-,1]
        dtype, device
    Returns:
        X: quadrature points size([N])
        W: quadrature weights size([N])
    """
    if N == 1:
        coord = torch.tensor([[0, 2]],dtype=dtype,device=device)
    elif N == 2:
        coord = torch.tensor([[-np.sqrt(3) / 3, 1],
                            [np.sqrt(3) / 3, 1]],dtype=dtype,device=device)
    elif N == 3:
        coord = torch.tensor([[-np.sqrt(15) / 5, 5 / 9],
                            [0, 8 / 9],
                            [np.sqrt(15) / 5, 5 / 9]],dtype=dtype,device=device)
    elif N == 4:
        coord = torch.tensor([[-np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7), (18 - np.sqrt(30)) / 36],
                            [-np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7), (18 + np.sqrt(30)) / 36],
                            [np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7), (18 + np.sqrt(30)) / 36],
                            [np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7), (18 - np.sqrt(30)) / 36]],dtype=dtype,device=device)
    elif N == 5:
        coord = torch.tensor([[-1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7)), (322 - 13 * np.sqrt(70)) / 900],
                            [- 1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)), (322 + 13 * np.sqrt(70)) / 900],
                            [0, 128 / 225],
                            [1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)), (322 + 13 * np.sqrt(70)) / 900],
                            [1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7)), (322 - 13 * np.sqrt(70)) / 900]],dtype=dtype,device=device)
    elif N == 6:
        coord = torch.tensor([[-0.932469514203152, 0.171324492379170],
                            [-0.661209386466264, 0.360761573048139],
                            [-0.238619186083197, 0.467913934572691],
                            [0.238619186083197, 0.467913934572691],
                            [0.661209386466264, 0.360761573048139],
                            [0.932469514203152, 0.171324492379170]],dtype=dtype,device=device)
    elif N == 7:
        coord = torch.tensor([[-0.949107912342758, 0.129484966168870],
                            [-0.741531185599394, 0.279705391489277],
                            [-0.405845151377397, 0.381830050505119],
                            [0, 0.417959183673469],
                            [0.405845151377397, 0.381830050505119],
                            [0.741531185599394, 0.279705391489277],
                            [0.949107912342758, 0.129484966168870]],dtype=dtype,device=device)
    elif N == 8:
        coord = torch.tensor([[-0.960289856497536, 0.101228536290377],
                            [-0.796666477413627, 0.222381034453374],
                            [-0.525532409916329, 0.313706645877887],
                            [-0.183434642495650, 0.362683783378362],
                            [0.183434642495650, 0.362683783378362],
                            [0.525532409916329, 0.313706645877887],
                            [0.796666477413627, 0.222381034453374],
                            [0.960289856497536, 0.101228536290377]],dtype=dtype,device=device)
    elif N == 9:
        coord = torch.tensor([[-0.968160239507626, 0.0812743883615744],
                            [-0.836031107326636, 0.180648160694858],
                            [-0.613371432700590, 0.260610696402936],
                            [-0.324253423403809, 0.312347077040003],
                            [0.0, 0.330239355001260],
                            [0.324253423403809, 0.312347077040003],
                            [0.613371432700590, 0.260610696402936],
                            [0.836031107326636, 0.180648160694858],
                            [0.968160239507626, 0.0812743883615744]],dtype=dtype,device=device)
    elif N == 10:
        coord = torch.tensor([[-0.973906528517172, 0.0666713443086881],
                            [-0.865063366688985, 0.149451349150581],
                            [-0.679409568299024, 0.219086362515982],
                            [-0.433395394129247, 0.269266719309997],
                            [-0.148874338981631, 0.295524224714753],
                            [0.148874338981631, 0.295524224714753],
                            [0.433395394129247, 0.269266719309997],
                            [0.679409568299024, 0.219086362515982],
                            [0.865063366688985, 0.149451349150581],
                            [0.973906528517172, 0.0666713443086881]],dtype=dtype,device=device)
    elif N == 11:
        coord = torch.tensor([[-0.978228658146057, 0.0556685671161737],
                            [-0.887062599768095, 0.125580369464904],
                            [-0.730152005574049, 0.186290210927734],
                            [-0.519096129206812, 0.233193764591991],
                            [-0.269543155952345, 0.262804544510247],
                            [0.0, 0.272925086777901],
                            [0.269543155952345, 0.262804544510247],
                            [0.519096129206812, 0.233193764591991],
                            [0.730152005574049, 0.186290210927734],
                            [0.887062599768095, 0.125580369464904],
                            [0.978228658146057, 0.0556685671161737]],dtype=dtype,device=device)
    elif N == 12:
        coord = torch.tensor([[-0.981560634246719, 0.0471753363865118],
                            [-0.904117256370475, 0.106939325995318],
                            [-0.769902674194305, 0.160078328543345],
                            [-0.587317954286617, 0.203167426723066],
                            [-0.367831498998180, 0.233492536538356],
                            [-0.125233408511469, 0.249147045813403],
                            [0.125233408511469, 0.249147045813403],
                            [0.367831498998180, 0.233492536538356],
                            [0.587317954286617, 0.203167426723066],
                            [0.769902674194305, 0.160078328543345],
                            [0.904117256370475, 0.106939325995318],
                            [0.981560634246719, 0.0471753363865118]],dtype=dtype,device=device)
    elif N == 13:
        coord = torch.tensor([[-0.984183054718588, 0.0404840047653159],
                            [-0.917598399222978, 0.0921214998377285],
                            [-0.801578090733310, 0.138873510219789],
                            [-0.642349339440340, 0.178145980761946],
                            [-0.448492751036447, 0.207816047536889],
                            [-0.230458315955135, 0.226283180262898],
                            [0.0, 0.232551553230874],
                            [0.230458315955135, 0.226283180262898],
                            [0.448492751036447, 0.207816047536889],
                            [0.642349339440340, 0.178145980761946],
                            [0.801578090733310, 0.138873510219789],
                            [0.917598399222978, 0.0921214998377285],
                            [0.984183054718588, 0.0404840047653159]],dtype=dtype,device=device)
    elif N == 14:
        coord = torch.tensor([[-0.986283808696812, 0.0351194603317519],
                            [-0.928434883663574, 0.0801580871597603],
                            [-0.827201315069765, 0.121518570687902],
                            [-0.687292904811685, 0.157203167158193],
                            [-0.515248636358154, 0.185538397477937],
                            [-0.319112368927890, 0.205198463721295],
                            [-0.108054948707344, 0.215263853463158],
                            [0.108054948707344, 0.215263853463158],
                            [0.319112368927890, 0.205198463721295],
                            [0.515248636358154, 0.185538397477937],
                            [0.687292904811685, 0.157203167158193],
                            [0.827201315069765, 0.121518570687902],
                            [0.928434883663574, 0.0801580871597603],
                            [0.986283808696812, 0.0351194603317519]],dtype=dtype,device=device)
    elif N == 15:
        coord = torch.tensor([[-0.987992518020485, 0.0307532419961174],
                            [-0.937273392400706, 0.0703660474881081],
                            [-0.848206583410427, 0.107159220467172],
                            [-0.724417731360170, 0.139570677926155],
                            [-0.570972172608539, 0.166269205816993],
                            [-0.394151347077563, 0.186161000015562],
                            [-0.201194093997435, 0.198431485327112],
                            [0.0, 0.202578241925561],
                            [0.201194093997435, 0.198431485327112],
                            [0.394151347077563, 0.186161000015562],
                            [0.570972172608539, 0.166269205816993],
                            [0.724417731360170, 0.139570677926155],
                            [0.848206583410427, 0.107159220467172],
                            [0.937273392400706, 0.0703660474881081],
                            [0.987992518020485, 0.0307532419961174]],dtype=dtype,device=device)
    elif N == 16:
        coord = torch.tensor([[-0.989400934991650, 0.0271524594117540],
                            [-0.944575023073233, 0.0622535239386481],
                            [-0.865631202387832, 0.0951585116824914],
                            [-0.755404408355003, 0.124628971255535],
                            [-0.617876244402644, 0.149595988816578],
                            [-0.458016777657227, 0.169156519395002],
                            [-0.281603550779259, 0.182603415044923],
                            [-0.0950125098376374, 0.189450610455069],
                            [0.0950125098376374, 0.189450610455069],
                            [0.281603550779259, 0.182603415044923],
                            [0.458016777657227, 0.169156519395002],
                            [0.617876244402644, 0.149595988816578],
                            [0.755404408355003, 0.124628971255535],
                            [0.865631202387832, 0.0951585116824914],
                            [0.944575023073233, 0.0622535239386481],
                            [0.989400934991650, 0.0271524594117540]],dtype=dtype,device=device)
    else:
        raise ValueError('This quadrature scheme is not implemented now!')
    return coord[:,0], coord[:,1]


def composite_quadrature_1d(N, a, b, M,dtype=torch.double, device='cpu'):
    """
    Quadrature points and quadrature weights for one-dimensional Gauss-Legendre quadrature rules,
    mesh domain [a,b] into M equal subintervels and use N quadrature points in each subinterval.

    Parameters:
        N: number of quadrature points in each subintervals
        a,b: computational domain [a,b]
        M: number of subintervals of [a,b] meshed to
        dtype,device
    Returns:
        X: quadrature points size([N*M])
        W: quadrature weights size([N*M])
    """
    h = (b-a)/M
    x, w = quadrature_1d(N,dtype,device)
    x = ((x+1)/2).repeat(M)*h+torch.linspace(a,b,M+1,dtype=dtype, device=device)[:-1].repeat_interleave(N)
    w = (w/2).repeat(M)*h
    return x, w


def composite_quadrature_2d(N,a1,b1,a2,b2,M1,M2,dtype=torch.double,device='cpu'):
    """
    Quadrature points and quadrature weights for two-dimensional tensor Gauss-Legendre quadrature rules,
    for [a1,b1]*[a2,b2], mesh domain [a1,b1] and [a2,b2] into M1 and M2 equal subintervels respectively,
    and use N quadrature points in one-dimensional subintervals to get tensor quadrature rules.

    Parameters:
        N: number of quadrature points in one dimension
        a1,b1,a2,b2: computational domain [a1,b1]*[a2,b2]
        M1: number of subintervals of [a1,b1] meshed to
        M2: number of subintervals of [a2,b2] meshed to
    Returns:
        X: quadrature points size([N^2*M1*M2,2])
        W: quadrature weights size([N^2*M1*M2])
    """
    p1, w1 = composite_quadrature_1d(N,a1,b1,M1,dtype=dtype,device=device)
    p2, w2 = composite_quadrature_1d(N,a2,b2,M2,dtype=dtype,device=device)
    p1 = p1.repeat(N*M2)
    p2 = p2.repeat_interleave(N*M1)
    w1 = w1.repeat(N*M2)
    w2 = w2.repeat_interleave(N*M1)
    return torch.stack((p2,p1),dim=1), w2*w1


# ******************** Bounded Domain ********************







# ******************** Unbounded Domain ********************
# Hermite-Gasuss Rule
def Hermite_Gauss_Quad(k,dtype=torch.double,device='cpu', modified=True):
    x, w = hermgauss(k)
    if modified:
        w = w*np.exp(x**2)
    return torch.from_numpy(x).to(dtype).to(device), torch.from_numpy(w).to(dtype).to(device)


# Laguerre-Gauss Rule
def Laguerre_Gauss_Quad(k,dtype=torch.double,device='cpu', modified=True):
    x, w = laggauss(k)
    if modified:
        w = w*np.exp(x)
    return torch.from_numpy(x).to(dtype).to(device), torch.from_numpy(w).to(dtype).to(device)






def main():
    # precision of prints
    torch.set_printoptions(precision=20)

    k = 3
    a = -5
    b = 5
    n = 3

    # test 1-dimensional composite quadrature rules
    # x, w = composite_quadrature_1d(k,a,b,n)
    # print(x)
    # print(w)

    # test 2-dimensional composite quadrature rules
    # x2d, w2d = composite_quadrature_2d(k,a,b,a,b,n,n)
    # print(x2d)
    # print(w2d)

    x, w = Laguerre_Gauss_Quad(3)


if __name__ == '__main__':
    main()
