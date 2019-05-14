import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
from torch.autograd import Variable
import torch.functional as F
import numpy as np
import copy
from util import pushforward_latent, haar_measure, rop, BatchDeterminant 

class DifferentialForm(nn.Module):
    
    def __init__(self, k, d, zero_form, one_form, alpha = 0.1, downsampler = None, d_small = -1, use_bound = False):
        """
        given a 0-form and a 1-form in \R^d, this adds and stacks them together 
        to implement a k-form in \R^d
        """
        
        super(DifferentialForm, self).__init__()

        self.k = k
        self.d = d
        
        if d_small == -1 or downsampler == None:
            self.d_small = d
        else:
            self.d_small = d_small
        
        self.nets = nn.ModuleList([copy.deepcopy(one_form) for i in range(k)])
        self.critic = zero_form
        self.alpha = alpha

        self.downsampler = downsampler
        self.det = BatchDeterminant.apply
        self.use_bound = use_bound 
        
    def b_network(self, x, v):
        """
        evaluate pairing of the form at point x with simple k-vector v 
        """
        
        if self.k == 0:
            return 0
        
        batch_size = x.shape[0]

        omega = torch.Tensor(batch_size, self.d_small, self.k).to(x.device)

        for idx, net in enumerate(self.nets):
            omega[:, :, idx] = net(x)

        V = v.view(batch_size, self.d, self.k)
        V = V.permute(0, 2, 1)

        if self.downsampler is not None:
            V = (self.downsampler(V)).view(batch_size, self.k, self.d_small)
            
        mm_prods = torch.bmm(V, omega)

        if self.k == 1:
            prod = mm_prods.view(-1, 1)

        elif self.k == 2:
            prod = mm_prods[:, 0, 0] * mm_prods[:, 1, 1] - mm_prods[:, 0, 1] * mm_prods[:, 1, 0]

        elif self.k == 3:
            a = mm_prods[:, 0, 0]
            b = mm_prods[:, 0, 1]
            c = mm_prods[:, 0, 2]
            d = mm_prods[:, 1, 0]
            e = mm_prods[:, 1, 1]
            f = mm_prods[:, 1, 2]
            g = mm_prods[:, 2, 0]
            h = mm_prods[:, 2, 1]
            i = mm_prods[:, 2, 2]

            prod = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
            
        else:
            prod = self.det(mm_prods) # use batched LU decomposition for determinants bigger than 3x3
            
        return self.alpha * prod.view(-1, 1)
        
    def forward(self, x, v):
        """
        evaluate full differential form omega 
        """

        return self.critic(x) + self.b_network(x, v)  

    def penalty_comass(self, x, v, lam = 1.0, penalty_pow = 2.0):
        """
        creates a penalty term to bound comass of the form at samples x, v
        """

        return (torch.clamp(torch.abs(self.forward(x, v)) - lam, min = 0.) ** penalty_pow).mean()

    def penalty_comass_exterior(self, x, v, lam = 1.0, penalty_pow = 2.0):
        """
        creates a penalty term to bound the comass of exterior derivative at samples x, v
        """

        bs = x.shape[0]
        d_omega = 0
        omega_b = [] 

        if self.use_bound == 1: 
            # use upper bound on exterior derivative (closer to WGAN-GP)            
            omega_a = self.critic(x)
            d_omega_a = autograd.grad(outputs = omega_a,
                                      inputs = x,
                                      grad_outputs = torch.ones_like(omega_a),
                                      create_graph = True,
                                      retain_graph = True,
                                      only_inputs = True)[0]

            if self.k > 0:
                for i in range(0, self.k + 1):
                    skip_i = [j for j in range(0, self.k + 1) if j != i]
                
                    omega_b.append(self.b_network(x, v[:, :, skip_i]))
                    d_omega += ((-1)**i) * rop(omega_b[-1], x, v[:, :, i].view(-1, self.d))[0]

                d_omega.abs_()

            d_omega += (self.k + 1) * d_omega_a.norm(2, dim = 1).mean()
            
        else:
            # "exact" exterior derivative (up to sampling)
            for i in range(0, self.k + 1):
                skip_i = [j for j in range(0, self.k + 1) if j != i]

                omega_b.append(self.forward(x, v[:, :, skip_i]))
                d_omega += ((-1)**i) * rop(omega_b[-1], x, v[:, :, i].view(-1, self.d))[0]

            d_omega.abs_()
            
        return (torch.clamp(d_omega - lam, min = 0) ** penalty_pow).mean(), omega_b

class FlatGAN:

    def __init__(self, opt, gen, omega, data, latent_dim, sample_latent):
        """
        """

        self.opt = opt
        self.gen = gen
        self.omega = omega
        self.data = data

        self.l = latent_dim
        self.k = omega.k
        self.d = omega.d
        
        self.optim_G = optim.Adam(gen.parameters(), lr = opt.lr_gen, betas = (opt.b1, opt.b2))
        self.optim_D = optim.Adam(omega.parameters(), lr = opt.lr_critic, betas = (opt.b1, opt.b2))

        self.dev = torch.device(opt.device)

        self.Gr1 = torch.Tensor(opt.bs, self.d, self.k)
        self.Gr2 = torch.Tensor(opt.bs, self.d, self.k + 1)
        self.z = torch.Tensor(opt.bs, self.l).to(self.dev)
        self.vecS = torch.Tensor(opt.bs, self.l, self.k).to(self.dev)
        self.sample_latent = sample_latent

    def step_critic(self, X, T):
        """
        updates the dual variable 
        """
        
        # update critic (differential form omega)
        for p in self.omega.parameters(): 
            p.requires_grad = True 

        for p in self.gen.parameters(): 
            p.requires_grad = False

        # don't learn downsampler parameters
        if self.omega.downsampler is not None:
            for p in self.omega.downsampler.parameters():
                p.requires_grad = False
            
        self.optim_D.zero_grad()
        
        # get fake examples
        self.sample_latent(self.z, self.vecS)
            
        zp, Dg = pushforward_latent(self.gen,
                                    Variable(self.z, requires_grad = True),
                                    self.vecS)
        
        # build loss
        loss = self.omega(X, T).mean() - self.omega(zp, Dg).mean()

        # add penalty terms        
        if self.k > 0:
            haar_measure(out = self.Gr1)
            loss += self.opt.rho * self.omega.penalty_comass(
               X,
               self.Gr1.to(self.dev),
               lam = self.opt.lmb,
               penalty_pow = 2.)
        else:
            loss += self.opt.rho * self.omega.penalty_comass(
                X,
                None,
                lam = self.opt.lmb,
                penalty_pow = 2.)
                        
        haar_measure(out = self.Gr2)

        penalty, omega_b = self.omega.penalty_comass_exterior(
            X,
            self.Gr2.to(self.dev),
            lam = 1.0,
            penalty_pow = 2.)
        
        loss += self.opt.rho * penalty 

        loss.backward(retain_graph = False)
        self.optim_D.step()
        
    def step_gen(self):
        """
        updates the generator 
        """

        for p in self.omega.parameters(): 
            p.requires_grad = False 

        for p in self.gen.parameters(): 
            p.requires_grad = True

        self.optim_G.zero_grad()

        # get fake samples and build loss for generator
        self.sample_latent(self.z, self.vecS)
        
        zp, Dg = pushforward_latent(self.gen,
                                    Variable(self.z, requires_grad = True),
                                    self.vecS)
        
        loss = self.omega(zp, Dg).mean()
        loss.backward(retain_graph = False)
        
        self.optim_G.step()

    def push_latent(self, z, s):
        return pushforward_latent(self.gen, Variable(z, requires_grad = True), s)
