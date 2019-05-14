import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage.filters import gaussian_filter

DIM_gen = 32
DIM_enc = 32
    
class MNIST_Generator(nn.Module):
    def __init__(self, latent_dim, dim = DIM_gen):
        super(MNIST_Generator, self).__init__()

        main = nn.Sequential(
            
            # latent vector
            nn.ConvTranspose2d(in_channels = latent_dim, out_channels=dim*32, kernel_size=4, stride=1, padding=0),
            nn.ELU(inplace = True),

            # 32F x 4 x 4
            nn.ConvTranspose2d(in_channels = dim*32, out_channels=dim*16, kernel_size=4, stride=2, padding=1),
            nn.ELU(inplace = True),

            # 16F x 8 x 8
            nn.ConvTranspose2d(in_channels = dim*16, out_channels=dim*4, kernel_size=4, stride=2, padding=1),
            nn.ELU(inplace = True),

            # 4F x 16 x 16
            nn.ConvTranspose2d(in_channels = dim*4, out_channels=1, kernel_size=4, stride=2, padding=1),

            # output: 1 x 32 x 32
        )

        self.main = main
        self.output = nn.Tanh()
        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.output(self.main(x.view(-1, self.latent_dim, 1, 1)))
        
        return x.view(-1, 32 * 32)

    
class MNIST_Discriminator(nn.Module):
    def __init__(self, tangent = False, dim = DIM_enc):
        super(MNIST_Discriminator, self).__init__()

        self.main_module = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=dim * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 2F x 16 x 16 
            nn.Conv2d(in_channels=dim * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 4F x 8 x 8
            nn.Conv2d(in_channels=dim * 4, out_channels=dim * 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)

            # output: 32F x 4 x 4
        )

        if tangent == False:
            self.last_layer = nn.Conv2d(in_channels = dim * 32, out_channels = 1, kernel_size = 4, stride = 1, padding = 0)
        else:
            self.last_layer = nn.ConvTranspose2d(in_channels = dim * 32, out_channels = 1, kernel_size = 4, stride = 2, padding = 1)

        self.tangent = tangent            
        self.dim = dim

    def forward(self, x):
        x = self.main_module(x.view(-1, 1, 32, 32))
        x = self.last_layer(x)

        if self.tangent == True:
            return x.view(-1, 8 * 8)
        else:
            return x.view(-1, 1)

        
class Downsampler(nn.Module):

    """
    downsamples by specified factor, combined with gaussian smoothing
    """
    
    def __init__(self, k = 1, factor = 8, sz = 32, sigma = 2, nc = 1):
        super(Downsampler, self).__init__()

        self.pooling = nn.AvgPool2d(factor, stride = factor)
        self.gaussian = nn.Conv2d(1, 1, kernel_size = (17, 17), bias = False, padding = 8)
        dirac = np.zeros((17, 17))
        dirac[8, 8] = 1
        self.gaussian.weight.data = torch.FloatTensor(gaussian_filter(dirac, sigma = sigma)).view(1, 1, 17, 17)
        self.k = k
        self.factor = factor
        self.size = sz // factor
        self.sz = sz
        self.nc = nc

        self.gaussian.requires_grad = False
        self.pooling.requires_grad = False

        if self.k == 0:
            raise("k should not be zero for Downsampler")

    def forward(self, x):
        x = x.view(-1, self.k, self.nc, self.sz, self.sz)

        output = torch.FloatTensor(x.shape[0], self.k, self.nc, (self.size ** 2)).to(x.device)
        for ch in range(self.k):
            for cc in range(self.nc):
                output[:, ch, cc, :] = self.pooling(self.gaussian(x[:, ch, cc, :, :].view(-1, 1, self.sz, self.sz))).view(-1, self.size ** 2)

        return output.view(-1, self.k, self.nc * (self.size ** 2))

    
class FullyConnectedNet(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim = 100, num_hidden = 1, z1_circle_topology = False):
        super(FullyConnectedNet, self).__init__()

        if z1_circle_topology == True:
            inp_dim += 1
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(inp_dim, hidden_dim))
        
        for i in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.z1_circle_topology = z1_circle_topology

    def forward(self, x):
        if self.z1_circle_topology == True:
            lifted_coords = torch.Tensor(x.shape[0], x.shape[1] + 1).to(x.device)
            lifted_coords[:, 2:] = x[:, 1:]
            lifted_coords[:, 0] = torch.cos(x[:, 0])
            lifted_coords[:, 1] = torch.sin(x[:, 0])
            x = lifted_coords
        
        for i in range(len(self.layers)-1):
            x = F.leaky_relu(self.layers[i](x), 0.2)

        return self.layers[-1](x)
