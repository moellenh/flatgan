import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
from math import sqrt 

class BatchDeterminant(autograd.Function):
    """
    batched computation of determinant
    """

    @staticmethod
    def forward(ctx, input):
        # compute determinant based on LU decomposition
        LU, pivot = F.btrifact(input, pivot = False)
        dets = torch.diagonal(LU, dim1 = 1, dim2 = 2).prod(dim = 1)

        # save LU decomposition for backward pass when inverse is required
        ctx.save_for_backward(LU, pivot, dets)
        
        return dets

    @staticmethod
    def backward(ctx, grad_output):
        LU, pivot, dets = ctx.saved_tensors

        LUinv_t = torch.zeros_like(LU)
        unit_vec = torch.zeros(LU.shape[0], LU.shape[1], dtype = LU.dtype).to(LU.device)
        
        for i in range(LU.shape[1]):
            unit_vec[:, i] = 1
            LUinv_t[:, i, :] = torch.btrisolve(unit_vec, LU, pivot)
            unit_vec[:, i] = 0            

        return grad_output.view(-1, 1, 1) * dets.view(-1, 1, 1) * LUinv_t

def haar_measure(out):
    """
    samples a k-dimensional subspace in R^d distributed according to the Haar measure on the Grassmannian Gr(k, d)
    out is assumed to be a tensor of dimension (bs, d, k)
    """

    bs, d, k = out.shape 
    z = torch.randn(bs, k, d) / sqrt(2)
    for i in range(bs):
        _, _, out[i, :, :] = torch.svd(z[i, :, :], some = True)

def rop(y, x, v):
    """
    Compute Jacobian vector product \nabla f(x) v with y = f(x).
    """
    
    if isinstance(y, tuple):
        w = [Variable(torch.zeros_like(y_), requires_grad=True) for y_ in y]
    else:
        w = Variable(torch.zeros_like(y), requires_grad=True)

    g = torch.autograd.grad(y, x, grad_outputs=w, create_graph=True, retain_graph=True, allow_unused=True)    
    r = torch.autograd.grad(g, w, grad_outputs=v, create_graph=True, retain_graph=True, allow_unused=True)
    
    return r 


def pushforward_latent(gen, z, Sz):
    """
    compute pushforward of latent current S sampled at (z, Sz)
    """

    zp = gen(z)
    d = zp.shape[1]

    if Sz is not None:
        bs, l, k = Sz.shape
    
        Dg = torch.Tensor(bs, d, k).to(z.device)
        for i in range(k):
            Dg[:, :, i] = rop(zp, z, Variable(Sz[:, :, i], requires_grad = True))[0]
            
    else:
        Dg = None

    return zp, Dg

def sample_latent_uniform(out_z, out_S, latent_scale):
    bs, l, k = out_S.shape 

    out_z.uniform_()
    out_z -= 0.5
    out_z *= latent_scale

    if k > 0:
        out_S.zero_()
        for i in range(k):
            out_S[:, i, i] = 1

def sample_latent_mixed(out_z, out_S, normal_scale, uniform_scale):
    bs, l, k = out_S.shape 

    # gaussian noise
    out_z.normal_(mean = 0, std = normal_scale) 

    # uniform noise 
    out_z[:, 0:k].uniform_()
    out_z[:, 0:k] -= 0.5

    for i in range(k):
        out_z[:, i] *= uniform_scale[i]

    if k > 0:
        out_S.zero_()
        for i in range(k):
            out_S[:, i, i] = 1
