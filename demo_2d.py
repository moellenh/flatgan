import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import argparse 
import torch
import torch.nn as nn
from util import sample_latent_uniform
from models import FullyConnectedNet
from flatgan import DifferentialForm, FlatGAN
import datasets
import os

# plotting parameters
manifold_color = (255. / 255., 200. / 255., 0. / 255., 0.004)
n_vis = 250000
DPI = 100
output_path = 'results/2d/'

# default parameters 
bs_default = 5
lr_gen_default = 1e-4
lr_critic_default = 1e-4
b1_default = 0.5
b2_default = 0.9
n_critic_default = 5
device_default = 'cpu'
latent_dim_default = 5
latent_scale_default = 2 * np.pi
rho_default = 10.
lmb_default = 1.
k_default = 1
vis_iters = [250, 500, 1000, 2000]
n_iter_default = vis_iters[-1]
scale = 1.75

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
parser.add_argument("--latent_scale", type=float, default=latent_scale_default, help="scaling of the latent space")
parser.add_argument("--rho", type=float, default=rho_default, help="penalty parameter for GP")
parser.add_argument("--lmb", type=float, default=lmb_default, help="flat norm scale")
parser.add_argument("--k", type=int, default=k_default, help="number of invariances")
opt = parser.parse_args()
params = vars(opt)

print('Training parameters:')
for q, v in params.items():
    print(q, '=', v)

dev = torch.device(opt.device)

if 'cuda' in opt.device:
    print('running on %s' % torch.cuda.get_device_name(dev))

data, num_tangents = datasets.get_circle_data(batch_size = opt.bs, N = 5)

# plot the data
(X, T) = next(iter(data))

outfile = 'k=%d_data_distribution.png' % opt.k
fig = plt.figure(frameon = False)
fig.set_size_inches(3, 3)    
ax = plt.Axes(fig, [0, 0, 1, 1], )
ax.set_xlim([-scale, scale])
ax.set_ylim([-scale, scale])
ax.set_axis_off()
fig.add_axes(ax)
    
plt.scatter(X[:,0], X[:,1], s = 50, c='black')

if opt.k == 1:
    plt.quiver(X[:,0], X[:,1], T[:,0], T[:,1], scale = 7.5, color='black')

plt.savefig(os.path.join(output_path, outfile), dpi = DPI)

gen = FullyConnectedNet(opt.latent_dim,
                        2,
                        hidden_dim = 250,
                        num_hidden = 3,
                        z1_circle_topology = True).to(dev)

disc = FullyConnectedNet(2,
                         1,
                         hidden_dim = 100,
                         num_hidden = 3)

disc_tan = FullyConnectedNet(2,
                             2,
                             hidden_dim = 50,
                             num_hidden = 2)
    
omega = DifferentialForm(opt.k,
                         2,
                         alpha = 0.1,
                         zero_form = disc, 
                         one_form = disc_tan,
                         downsampler = None,
                         d_small = 2,
                         use_bound = True).to(dev)

fg = FlatGAN(opt,
             gen,
             omega,
             data,
             opt.latent_dim, 
             sample_latent = lambda z, s : sample_latent_uniform(z, s, opt.latent_scale))

for it in range(opt.n_iter):
    if (it + 1) % 100 == 0:
        print('Iteration %5d' % (it + 1))

    for d_it in range(opt.n_critic):   
        try:
            (X, T) = next(data_iter)
        except:
            data_iter = iter(data)
            (X, T) = next(data_iter)

        X = X.to(dev)
        T = T.to(dev)    
        X.requires_grad = True
        fg.step_critic(X, T)
        
    fg.step_gen() 

    if (it + 1) in vis_iters:
        z_vis = torch.Tensor(n_vis, opt.latent_dim).to(dev)
        s_vis = torch.Tensor(n_vis, opt.latent_dim, opt.k).to(dev)
        sample_latent_uniform(out_z = z_vis, out_S = s_vis, latent_scale = opt.latent_scale)

        z_vis.requires_grad = True
        s_vis.requires_grad = True
        zp, Dg = fg.push_latent(z_vis, s_vis)

        zcurve = torch.zeros(100, opt.latent_dim).to(dev)
        zcurve[:, :] = torch.rand(1, opt.latent_dim).to(dev)
        zcurve[:,0] = torch.linspace(-opt.latent_scale/2, opt.latent_scale/2, 100)
        zpcurve = gen(zcurve)

        zp = zp.detach().cpu().numpy()
        zpcurve = zpcurve.detach().cpu().numpy()
        Dg = Dg.detach().cpu().numpy()
        
        outfile = 'k=%d_epoch=%05d.png' % (opt.k, it + 1)
        fig = plt.figure(frameon = False)
        fig.set_size_inches(3,3)
        ax = plt.Axes(fig, [0, 0, 1, 1], )
        ax.set_xlim([-scale, scale])
        ax.set_ylim([-scale, scale])
        ax.set_axis_off()
        fig.add_axes(ax)

        plt.scatter(zp[:,0], zp[:,1], s=1, c=manifold_color)
        plt.plot(zpcurve[:,0].squeeze(), zpcurve[:,1].squeeze(), 'k', linewidth = 2)
        plt.savefig(os.path.join(output_path, outfile), dpi = DPI)

        plt.close()
