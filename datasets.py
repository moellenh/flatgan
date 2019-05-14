import torch
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import torchvision.transforms.functional as TF
import PIL
import PIL.ImageFilter as IF
import numpy as np

def get_circle_data(batch_size = 5, N = 5):
    """
    returns data loader for N points on unit circle 
    """
    
    num_tangents = 1
    
    def circle(z):
        x = torch.cos(z * 2 * np.pi)
        y = torch.sin(z * 2 * np.pi)
        
        return torch.stack([x, y], dim = 1)

    z = torch.linspace(0, 1 - 1. / float(N), N) 
    h = 1e-5
    
    X = circle(z)
    T = (circle(z + h) - circle(z)) / h 
    T = T / (torch.norm(T, dim=1).repeat(2, 1).t())

    dataset = TensorDataset(X, T)

    return DataLoader(dataset, batch_size = min(batch_size, N), shuffle = True), num_tangents 

def get_MNIST_data(batch_size = 100, scale = [1, 1]):
    """
    returns data loader for MNIST with two tangent vectors
    """
    
    num_tangents = 2
    angle = 1.0
    filename = 'data/MNIST_scale_%.2f_%.2f.pt' % (scale[0], scale[1])
    
    try:
        X, T = torch.load(open(filename, 'rb'))
        
    except:
        print('temporary file "%s" not found -- processing data...' % filename)
        train_data = datasets.MNIST('data/', train=True, download=True)
        test_data = datasets.MNIST('data/', train=False, download=True)
        data = ConcatDataset([train_data, test_data])

        temp = torch.Tensor(len(data), 32 * 32, 4)
        X = torch.Tensor(len(data), 32 * 32)
        T = torch.Tensor(len(data), 32 * 32, 2)
        
        for i in range(0, len(data)):
            temp[i, :, 0] = TF.to_tensor(TF.resize(data[i][0], (32, 32))).view(-1)
            temp[i, :, 1] = TF.to_tensor(TF.rotate(TF.resize(data[i][0], (32, 32)), angle, resample = PIL.Image.BILINEAR)).view(-1)
            temp[i, :, 2] = TF.to_tensor(TF.rotate(TF.resize(data[i][0], (32, 32)), -angle, resample = PIL.Image.BILINEAR)).view(-1)
            temp[i, :, 3] = TF.to_tensor(TF.resize(data[i][0], (32, 32)).filter(IF.MaxFilter(3))).view(-1)
            
        # scale to [-1, 1]
        temp -= 0.5
        temp *= 2.

        # compute tangent vectors 
        X[:, :] = temp[:, :, 0]
        T[:, :, 0] = (temp[:, :, 0] - temp[:, :, 3]) * scale[0] # thickness
        T[:, :, 1] = (temp[:, :, 2] - temp[:, :, 1]) * scale[1] # rotation (central differences)
        
        torch.save((X, T), open(filename, 'wb'))
        
    dataset = TensorDataset(X.squeeze(),
                            T.squeeze())
        
    return DataLoader(dataset, batch_size = batch_size, shuffle = True), num_tangents
