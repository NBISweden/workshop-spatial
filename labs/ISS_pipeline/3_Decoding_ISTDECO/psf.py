import torch
import copy
import numpy as np
from typing import Union, Tuple

class PSF:
    '''
        Point spread function - Helper class for doing arithmetics with the point spread function

        Parameters
        ----------
        shape: tuple(int,int)
            size of the images that are to be blurred by the psf.

        sigma : tuple(float,float)
            Tuple with standard deviations for the point spread functions

        upscale : int

    '''
    
    def __init__(self, shape: Tuple[int,int], sigma: Tuple[float,float], upscale: float = 1):
        self.ndim = np.sum(np.array(shape) > 1)
        if sigma[0] > 0:
            self.By = load_B(shape[0], sigma=sigma[0], s=upscale).t()
        else:
            self.By = 1
        self.Bx = load_B(shape[1], sigma=sigma[1], s=upscale)

    def to(self, device):
        '''
            Put tensors on a device.
            See Pytorch doc.
        '''
        if self.ndim > 1:
            self.By = self.By.to(device)
        self.Bx = self.Bx.to(device)
        return self

    def matmul(self, tensor):
        '''
            Multiplies a (rc,y,x) shaped tensor with the psf
        '''

        # If two dimensions, blurr in x and y
        if self.ndim > 1:
            return self.By @ tensor @ self.Bx
        return tensor @ self.Bx

    def matmul_t(self, tensor):
        '''
            Multiplies a (rc,sy,sx) shaped tensor with the transposed psf.
            Here, s is the upscale factor.
        '''
        if self.ndim > 1:
            return self.By.t() @ tensor @ self.Bx.t()
        return tensor @ self.Bx.t()


    def down_pool(self, tensor):
        '''
            Convolves an image with constant filter with same 
            support as the PSF (2*(3sigma)+1)
        '''

        if self.ndim > 1:
            return (self.By != 0).float() @ tensor @ (self.Bx != 0).float()
        return tensor @ (self.Bx != 0).float()

    def up_pool(self, tensor):
        '''
            Transpose. Convolves an image with constant filter with same 
            support as the PSF (2*(3sigma)+1)
        '''
        if self.ndim > 1:
            return (self.By.t() != 0).float() @ tensor @ (self.Bx.t() != 0).float()
        return tensor @ (self.Bx.t() != 0).float()

def load_B(n:int, sigma:float=1.0,s:float=1.0):
    '''
        Function used for creating a 'convolution matrix'.
        
        Parameters 
        ----------
            n : int 
                Image width
            sigma : float
                The std of the Gaussian shaped psf
            s : int
                The upscale factor. 
    '''
    if sigma == 0:
        return 1

    ns = int(np.ceil(n*s)); sigmas = sigma*s
    x = torch.linspace(0,ns-1,n); xs = torch.arange(ns)
    b = torch.exp(-0.5*(xs.reshape((-1,1)) - x)**2/sigmas**2).float()
    b = b / b.sum(axis=1,keepdim=True)

    bclosed = torch.zeros_like(b)
    f = 2*np.ceil(3*sigmas)+1
    w = int(f // 2)
    for i in range(x.shape[0]):
        xi = int(x[i])
        start = max(0,xi-w); stop = min(xs.shape[0],xi+w+1) 
        slize = slice(start,stop)
        bclosed[slize,i] = b[slize,i]
    bclosed = bclosed / b.sum(axis=0,keepdim=True) 
    return bclosed