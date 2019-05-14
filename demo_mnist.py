import torch
import torch.nn as nn
import os
from torchvision.utils import save_image, make_grid
from util import sample_latent_mixed
from models import MNIST_Generator, MNIST_Discriminator, Downsampler
from flatgan import DifferentialForm, FlatGAN
import datasets
import numpy as np
import argparse 

# default parameters 
n_iter_default = 50000
bs_default = 50
lr_gen_default = 1e-4
lr_critic_default = 1e-4
b1_default = 0.5
b2_default = 0.9
n_critic_default = 1
device_default = 'cuda'
latent_dim_default = 128
latent_scale_uniform1_default = 15.0  #rotation
latent_scale_uniform2_default = 1.0   #thickness
latent_scale_normal_default = 1.
rho_default = 5.
rho_pow_default = 2.
alpha_default = 1e-5
lmb_default = 1.
k_default = 2
vis_iter_default = 500
tan_scale1_default = 0.05
tan_scale2_default = 10.

parser = argparse.ArgumentParser()
parser.add_argument("--n_iter", type=int, default=n_iter_default, help="number of generator updates")
parser.add_argument("--bs", type=int, default=bs_default, help="size of the batches")
parser.add_argument("--lr_gen", type=float, default=lr_gen_default, help="adam: learning rate of generator")
parser.add_argument("--lr_critic", type=float, default=lr_critic_default, help="adam: learning rate of critic")
parser.add_argument("--b1", type=float, default=b1_default, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=b2_default, help="adam: decay of second order momentum of gradient")
parser.add_argument("--n_critic", type=int, default=n_critic_default, help="number of training steps for discriminator per iter")
parser.add_argument("--device", default=device_default, help="device to run on")
parser.add_argument("--latent_dim", type=int, default=latent_dim_default, help="dimensionality of the latent space")
parser.add_argument("--latent_scale_uniform1", type=float, default=latent_scale_uniform1_default, help="scaling of the latent space")
parser.add_argument("--latent_scale_uniform2", type=float, default=latent_scale_uniform2_default, help="scaling of the latent space")
parser.add_argument("--latent_scale_normal", type=float, default=latent_scale_normal_default, help="scaling of the latent space")
parser.add_argument("--alpha", type=float, default=alpha_default, help="scaling of discriminator")
parser.add_argument("--rho", type=float, default=rho_default, help="penalty parameter")
parser.add_argument("--lmb", type=float, default=lmb_default, help="flat norm scale")
parser.add_argument("--k", type=int, default=k_default, help="number of invariances")
parser.add_argument("--vis_iter", type=int, default=vis_iter_default, help="iterations between visualizations")
parser.add_argument("--tan_scale1", type=float, default=tan_scale1_default, help="scaling of tangent vector1")
parser.add_argument("--tan_scale2", type=float, default=tan_scale2_default, help="scaling of tangent vector2")

opt = parser.parse_args()
params = vars(opt)

print('Training parameters:')
for k, v in params.items():
    print(k, '=', v)

dev = torch.device(opt.device)

if 'cuda' in opt.device:
    print('running on %s' % torch.cuda.get_device_name(dev))

data, num_tangents = datasets.get_MNIST_data(batch_size = opt.bs, scale = [opt.tan_scale1, opt.tan_scale2])

gen = MNIST_Generator(latent_dim = opt.latent_dim).to(dev)
disc = MNIST_Discriminator(tangent = False, dim = 32).to(dev)
disc_tan = MNIST_Discriminator(tangent = True, dim = 8).to(dev)

omega = DifferentialForm(opt.k,
                         32 * 32,
                         zero_form = disc,
                         one_form = disc_tan, 
                         alpha = opt.alpha,
                         downsampler = Downsampler(k = opt.k, factor = 4, sigma = 2.),
                         d_small = 8 * 8,
                         use_bound = True).to(dev)

latent_sampler = lambda z, s : sample_latent_mixed(z, s, normal_scale = opt.latent_scale_normal, uniform_scale = [opt.latent_scale_uniform1, opt.latent_scale_uniform2])

fg = FlatGAN(opt,
             gen,
             omega,
             data,
             opt.latent_dim, 
             sample_latent = latent_sampler)

# prepare latent space for visualization
z_vis = torch.Tensor(20 * 10, opt.latent_dim).to(dev)
s_vis = torch.Tensor(20 * 10, opt.latent_dim, opt.k).to(dev)
latent_sampler(z_vis, s_vis)
z_vis = z_vis.view(20, 10, -1)
z_vis[0:10, 1:10, :] = z_vis[0:10, 0:1, :].repeat(1,9,1)
z_vis[10:, 0:10, :] = z_vis[0:10, :, :]
z_vis[:, :, 0:2] = 0
z_vis[0:10, :, 0:1] = torch.linspace(-opt.latent_scale_uniform1 / 2, opt.latent_scale_uniform1 / 2, 10).view(1,10,1).repeat(10,1,1)
z_vis[10:20, :, 1:2] = torch.linspace(-opt.latent_scale_uniform2 / 2, opt.latent_scale_uniform2 / 2, 10).view(1,10,1).repeat(10,1,1)
z_vis = z_vis.view(20 * 10, -1)

for it in range(opt.n_iter + 1):
    if (it + 1) % 100 == 0:
        print('Iteration %5d' % (it + 1))

    for d_it in range(opt.n_critic):   
        try:
            (X, T) = next(data_iter)
        except:
            data_iter = iter(data)
            (X, T) = next(data_iter)

        if X.shape[0] != opt.bs:
            continue

        T = T[:, :, 0:opt.k]
        X = X.to(dev)
        T = T.to(dev)
        
        X.requires_grad = True
        fg.step_critic(X, T)
        
    fg.step_gen() 

    # result visualizations
    if it % opt.vis_iter == 0:
        
        # visualize latent walk 
        zp, _ = fg.push_latent(z_vis, s_vis) 
        zp = zp.detach().cpu().view(-1, 1, 32, 32)
        zp = (zp + 1.) / 2

        zp = zp.view(20, 10, 32, 32)
        zp2 = torch.zeros(10, 10, 32, 32)
        zp3 = torch.zeros(10, 10, 32, 32)
        zp2[:, 0:10, :, :] = zp[0:10, :, :, :]
        zp3[:, 0:10, :, :] = zp[10:20, :, :, :]
        zp2 = zp2.view(-1, 1, 32, 32)
        zp3 = zp3.view(-1, 1, 32, 32)                
        save_image(make_grid(1. - zp2, nrow = 10, range = (0, 1)), 'results/mnist/rotate_it_%05d.png' % it)
        save_image(make_grid(1. - zp3, nrow = 10, range = (0, 1)), 'results/mnist/dilate_it_%05d.png' % it)
