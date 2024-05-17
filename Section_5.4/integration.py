import torch.nn as nn
import torch


# Integration operations for TNN
def normalization(w, phi):
    return torch.prod(torch.sqrt(torch.sum(w*phi**2,dim=2)),dim=0), phi / torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1)


def Int1TNN(w, alpha, phi, if_sum=True):
    """
    Integration for one TNN.

    Paramters:
        w: quad weights [N]
        alpha: scaling parameters [p]
        phi: values of TNN on quad points [dim, p, N]
    Return:
        [1] if_sum=True
        [p] if_sum=False
    """
    if if_sum:
        return torch.sum(alpha*torch.prod(torch.sum(w*phi,dim=2),dim=0))
    else:
        return alpha*torch.prod(torch.sum(w*phi,dim=2),dim=0)


def Int2TNN(w, alpha1, phi1, alpha2, phi2, if_sum=True):
    """
    Integration of prod of two TNNs (L2 inner product of two TNNs).

    Paramters:
        w: quad weights [N]
        alpha1: scaling parameters of TNN1 [p1]
        phi1: values of TNN1 on quad points [dim, p1, N]
        alpha2: scaling parameters of TNN2 [p2]
        phi2: values of TNN2 on quad points [dim, p2, N]
    Return:
        [1] if_sum=True
        [p1,p2] if_sum=False
    """
    if if_sum:
        return torch.sum(torch.outer(alpha1,alpha2)*torch.prod((w*phi1)@phi2.transpose(1,2),dim=0))
    else:
        return torch.outer(alpha1,alpha2)*torch.prod((w*phi1)@phi2.transpose(1,2),dim=0)


def Int2TNN_amend_1d(w1, w2, alpha1, phi1, alpha2, phi2, grad_phi1, grad_phi2, if_sum=True):
    """
    Integration of prod of two TNNs and amend each dimension respectively (H1 inner product of two TNNs).

    Paramters:
        w: quad weights [N]
        alpha1: scaling parameters of TNN1 [p1]
        phi1: values of TNN1 on quad points [dim,p1,N]
        alpha2: scaling parameters of TNN2 [p2]
        phi2: values of TNN2 on quad points [dim,p2,N]
        grad_phi1: gradient values of TNN1 [dim,p1,N]
        grad_phi2: gradient values of TNN2 [dim,p2,N]
    Return:
        [1] if_sum=True
        [p1,p2] if_sum=False
    """

    # Efficient calculation method, but may have numerical instability
    if if_sum:
        return torch.sum(torch.sum(((w2*grad_phi1)@grad_phi2.transpose(1,2)),dim=0))
        #return torch.sum(Int2TNN(w1, alpha1, phi1, alpha2, phi2, if_sum=False) * torch.sum(((w2*grad_phi1)@grad_phi2.transpose(1,2))/((w1*phi1)@phi2.transpose(1,2)),dim=0))
    else:
        return torch.sum(((w2*grad_phi1)@grad_phi2.transpose(1,2)),dim=0)
        #return Int2TNN(w1, alpha1, phi1, alpha2, phi2, if_sum=False) * torch.sum(((w2*grad_phi1)@grad_phi2.transpose(1,2))/((w1*phi1)@phi2.transpose(1,2)),dim=0)


    # Numerically stable, but expensive to store
    # if if_sum:
    #     dim = phi1.size(0)
    #     a = (w1*phi1)@phi2.transpose(1,2).unsqueeze(dim=0).expand(dim,-1,-1,-1)
    #     b = (w2*grad_phi1)@grad_phi2.transpose(1,2)
    #     a[torch.arange(dim),torch.arange(dim),:,:] = b
    #     # print(torch.sum(Int2TNN(w1, alpha1, phi1, alpha2, phi2, if_sum=False) * torch.sum(((w2*grad_phi1)@grad_phi2.transpose(1,2))/((w1*phi1)@phi2.transpose(1,2)),dim=0))-torch.sum(torch.outer(alpha1,alpha2)*torch.prod(a,dim=1)))
    #     return torch.sum(torch.outer(alpha1,alpha2)*torch.prod(a,dim=1))
    # else:
    #     dim = phi1.size(0)
    #     a = (w1*phi1)@phi2.transpose(1,2).unsqueeze(dim=0).expand(dim,-1,-1,-1)
    #     b = (w2*grad_phi1)@grad_phi2.transpose(1,2)
    #     a[torch.arange(dim),torch.arange(dim),:,:] = b
    #     # print(Int2TNN(w1, alpha1, phi1, alpha2, phi2, if_sum=False) * torch.sum(((w2*grad_phi1)@grad_phi2.transpose(1,2))/((w1*phi1)@phi2.transpose(1,2)),dim=0)-torch.sum(torch.outer(alpha1,alpha2)*torch.prod(a,dim=1),dim=0))
    #     return torch.sum(torch.outer(alpha1,alpha2)*torch.prod(a,dim=1),dim=0)


def Int3TNN(w, alpha1, phi1, alpha2, phi2, alpha3, phi3, if_sum=True):
    """
    Integration of prod of three TNNs.

    Paramters:
        w: quad weights [N]
        alpha1: scaling parameters of TNN1 [p1]
        phi1: values of TNN1 on quad points [dim,p1,N]
        alpha2: scaling parameters of TNN2 [p2]
        phi2: values of TNN2 on quad points [dim,p2,N]
        alpha3: scaling parameters of TNN3 [p3]
        phi3: values of TNN3 on quad points [dim,p3,N]
    Return:
        [1] if_sum=True
        [p1,p2,p3] if_sum=False
    """
    if if_sum:
        return torch.sum(torch.einsum('i,j,k->ijk',alpha1,alpha2,alpha3)*torch.prod(torch.einsum('din,djn,dkn->dijk',w*phi1,phi2,phi3),dim=0))
    else:
        return torch.einsum('i,j,k->ijk',alpha1,alpha2,alpha3)*torch.prod(torch.einsum('din,djn,dkn->dijk',w*phi1,phi2,phi3),dim=0)


def Int4TNN(w, alpha1, phi1, alpha2, phi2, alpha3, phi3, alpha4, phi4, if_sum=True):
    """
    Integration of prod of four TNNs.

    Paramters:
        w: quad weights [N]
        alpha1: scaling parameters of TNN1 [p1]
        phi1: values of TNN1 on quad points [dim,p1,N]
        alpha2: scaling parameters of TNN2 [p2]
        phi2: values of TNN2 on quad points [dim,p2,N]
        alpha3: scaling parameters of TNN3 [p3]
        phi3: values of TNN3 on quad points [dim,p3,N]
        alpha4: scaling parameters of TNN4 [p4]
        phi4: values of TNN4 on quad points [dim,p4,N]
    Return:
        [1] if_sum=True
        [p1,p2,p3,p4] if_sum=False
    """
    if if_sum:
        return torch.sum(torch.einsum('i,j,k,l->ijkl',alpha1,alpha2,alpha3,alpha4)*torch.prod(torch.einsum('din,djn,dkn,dln->dijkl',w*phi1,phi2,phi3,phi4),dim=0))
    else:
        return torch.einsum('i,j,k,l->ijkl',alpha1,alpha2,alpha3,alpha4)*torch.prod(torch.einsum('din,djn,dkn,dln->dijkl',w*phi1,phi2,phi3,phi4),dim=0)



def InnerH1_with_coefficients(w, alpha_A, A, alpah1, phi1, alpha2, phi2, if_sum=True):
    if if_sum:
        return
    else:
        return


# Integration operations for Multi_TNN
def Int1MultiTNN(w, alpha, phi, if_sum=True):
    """
    Integration of one Multi_TNNs.

    Paramters:
        w: quad weights [N]
        alpha1: scaling parameters of TNN1 [k,p]
        phi1: values of Multi_TNN1 on quad points [k,dim,p,N]
    Return:
        [k] if_sum=True
        [k,p] if_sum=False
    """
    if if_sum:
        return torch.sum(alpha*torch.prod(torch.sum(w*phi,dim=-1),dim=1),dim=-1)
    else:
        return alpha*torch.prod(torch.sum(w*phi,dim=-1),dim=1)


def Int2MultiTNN(w, alpha1, phi1, alpha2, phi2, if_sum=True):
    """
    Integration of prod of two Multi_TNNs (assemble mass matrix).

    Paramters:
        w: quad weights [N]
        alpha1: scaling parameters of TNN1 [k1,p1]
        phi1: values of Multi_TNN1 on quad points [k1,dim,p1,N]
        alpha2: scaling parameters of TNN2 [k2,p2]
        phi2: values of Multi_TNN2 on quad points [k2,dim,p2,N]
        grad_phi1: gradient Multi_values of TNN1 [k1,dim,p1,N]
        grad_phi2: gradient Multi_values of TNN2 [k2,dim,p2,N]
    Return:
        [k1,k2] if_sum=True
        [k1,k2,p1,p2] if_sum=False
    """
    if if_sum:
        return torch.sum(torch.einsum('ai,bj->abij',alpha1,alpha2)*torch.prod(torch.einsum('adin,bdjn->abdij',w*phi1,phi2),dim=2),dim=(-1,-2))
    else:
        return torch.einsum('ai,bj->abij',alpha1,alpha2) * torch.prod(torch.einsum('adin,bdjn->abdij',w*phi1,phi2),dim=2)


def Int2MultiTNN_amend_1d(w1, w2, alpha1, phi1, alpha2, phi2, grad_phi1, grad_phi2, if_sum=True):
    """
    Integration of prod of two Multi_TNNs and amend each dimension (assemble stiffness matrix).

    Paramters:
        w: quad weights [N]
        alpha1: scaling parameters of TNN1 [k1,p1]
        phi1: values of Multi_TNN1 on quad points [k1,dim,p1,N]
        alpha2: scaling parameters of TNN2 [k2,p2]
        phi2: values of Multi_TNN2 on quad points [k2,dim,p2,N]
    Return:
        [k1,k2] if_sum=True
        [k1,k2,p1,p2] if_sum=False
    """
    # if if_sum:
    #     return torch.sum(Int2MultiTNN(w1, alpha1, phi1, alpha2, phi2, if_sum=False) * torch.sum(torch.einsum('adin,bdjn->abdij',w2*grad_phi1,grad_phi2)/torch.einsum('adin,bdjn->abdij',w1*phi1,phi2),dim=2),dim=(-1,-2))
    # else:
    #     return Int2MultiTNN(w1, alpha1, phi1, alpha2, phi2, if_sum=False) * torch.sum(torch.einsum('adin,bdjn->abdij',w2*grad_phi1,grad_phi2)/torch.einsum('adin,bdjn->abdij',w1*phi1,phi2),dim=2)
    
    # if if_sum:
    #     dim = phi1.size(1)
    #     a = torch.einsum('adin,bdjn->abdij',w1*phi1,phi2).unsqueeze(dim=2).expand(-1,-1,dim,-1,-1,-1)
    #     b = torch.einsum('adin,bdjn->abdij',w2*grad_phi1,grad_phi2)
    #     print(a[:,:,torch.arange(dim),torch.arange(dim),:,:].size())
    #     print(b.size())
    #     a[:,:,torch.arange(dim),torch.arange(dim),:,:] = b
    #     # print(a.diagonal().diag_embed().size())
    #     # a = a - a.diagonal().diag_embed(dim1=0,dim2=1) + b.diag_embed(dim1=0,dim2=1)
    #     return torch.sum(torch.einsum('ai,bj->abij',alpha1,alpha2).unsqueeze(dim=2)*torch.prod(a,dim=2))
    # else:
    #     dim = phi1.size(1)
    #     a = torch.einsum('adin,bdjn->abdij',w1*phi1,phi2).unsqueeze(dim=2).expand(-1,-1,dim,-1,-1,-1)
    #     b = torch.einsum('adin,bdjn->abdij',w2*grad_phi1,grad_phi2)
    #     a[:,:,torch.arange(dim),torch.arange(dim),:,:] = b
    #     return torch.sum(torch.einsum('ai,bj->abij',alpha1,alpha2).unsqueeze(2)*torch.prod(a,dim=3),dim=2)

    if if_sum:
        dim = phi1.size(1)
        a = torch.einsum('adin,bdjn->abdij',w1*phi1,phi2).unsqueeze(dim=2).expand(-1,-1,dim,-1,-1,-1)
        b = torch.einsum('adin,bdjn->abdij',w2*grad_phi1,grad_phi2).unsqueeze(dim=2).expand(-1,-1,dim,-1,-1,-1)
        # print(a[:,:,torch.arange(dim),torch.arange(dim),:,:].size())
        # print(a.size())
        # print(a.diagonal(dim1=2,dim2=3).size())
        # print(a.diagonal(dim1=2,dim2=3).diag_embed(dim1=2,dim2=3).size())

        a = a - a.diagonal(dim1=2,dim2=3).diag_embed(dim1=2,dim2=3) + b.diagonal(dim1=2,dim2=3).diag_embed(dim1=2,dim2=3)
        # print(a.diagonal().diag_embed().size())
        # a = a - a.diagonal().diag_embed(dim1=0,dim2=1) + b.diag_embed(dim1=0,dim2=1)
        return torch.sum(torch.einsum('ai,bj->abij',alpha1,alpha2).unsqueeze(dim=2)*torch.prod(a,dim=3),dim=(-1,-2,-3))
    else:
        dim = phi1.size(1)
        a = torch.einsum('adin,bdjn->abdij',w1*phi1,phi2).unsqueeze(dim=2).expand(-1,-1,dim,-1,-1,-1)
        b = torch.einsum('adin,bdjn->abdij',w2*grad_phi1,grad_phi2).unsqueeze(dim=2).expand(-1,-1,dim,-1,-1,-1)

        a = a - a.diagonal(dim1=2,dim2=3).diag_embed(dim1=2,dim2=3) + b.diagonal(dim1=2,dim2=3).diag_embed(dim1=2,dim2=3)
        return torch.sum(torch.einsum('ai,bj->abij',alpha1,alpha2).unsqueeze(dim=2)*torch.prod(a,dim=3),dim=2)


def Int3MultiTNN(w, alpha1, phi1, alpha2, phi2, alpha3, phi3, if_sum=True):
    """
    Integration of prod of three Multi_TNNs.

    Paramters:
        w: quad weights [N]
        alpha1: scaling parameters of TNN1 [k1,p1]
        phi1: values of Multi_TNN1 on quad points [k1,dim,p1,N]
        alpha2: scaling parameters of TNN2 [k2,p2]
        phi2: values of Multi_TNN2 on quad points [k2,dim,p2,N]
        alpha3: scaling parameters of TNN3 [k3,p3]
        phi3: values of Multi_TNN3 on quad points [k3,dim,p3,N]
    Return:
        [k1,k2,k3] if_sum=True
        [k1,k2,k3,p1,p2,p3] if_sum=False
    """
    if if_sum:
        return torch.sum(torch.einsum('ai,bj,ck->abcijk',alpha1,alpha2,alpha3)*torch.prod(torch.einsum('adin,bdjn,cdkn->abcdijk',w*phi1,phi2,phi3),dim=3),dim=(-1,-2,-3))
    else:
        return torch.einsum('ai,bj,ck->abcijk',alpha1,alpha2,alpha3)*torch.prod(torch.einsum('adin,bdjn,cdkn->abcdijk',w*phi1,phi2,phi3),dim=3)






# Integration operations for TNN_List


# Integration operations for Multi_TNN_List



# ********** error estimator **********
def error0_estimate(w, alpha_F, F, alpha, phi, projection=True):
    inner0_phi_phi = Int2TNN(w, alpha, phi, alpha, phi)
    inner0_F_phi = Int2TNN(w, alpha_F, F, alpha, phi)
    inner0_F_F = Int2TNN(w, alpha_F, F, alpha_F, F)
    if projection:
        return torch.sqrt(1 - torch.sum(inner0_F_phi)**2 / (torch.sum(inner0_phi_phi)*torch.sum(inner0_F_F)))
    else:
        return torch.sqrt(torch.sum(inner0_phi_phi) - 2 * torch.sum(inner0_F_phi) + torch.sum(inner0_F_F))


def error1_estimate(w, alpha_F, F, alpha, phi, grad_F, grad_phi, projection=True):
    inner1_phi_phi = Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, grad_phi, grad_phi)
    inner1_F_phi = Int2TNN_amend_1d(w, w, alpha_F, F, alpha, phi, grad_F, grad_phi)
    inner1_F_F = Int2TNN_amend_1d(w, w, alpha_F, F, alpha_F, F, grad_F, grad_F)
    if projection:
        return torch.sqrt(1 - torch.sum(inner1_F_phi)**2 / (torch.sum(inner1_phi_phi)*torch.sum(inner1_F_F)))
    else:
        return torch.sqrt(torch.sum(inner1_phi_phi) - 2 * torch.sum(inner1_F_phi) + torch.sum(inner1_F_F))

