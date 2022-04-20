import torch
import copy
import numpy as np
from typing import Union, Tuple
from codebook import Codebook
from psf import PSF

class ISTDeco:
    '''
    ISTDECO - Deconvovles 1D or 2D spatial transcriptomic data

    Parameters
    ----------
    Y : float
        Input image data of shape (rounds, channels, height, width)
    D : float
        Codebook of shape (ncodes, rounds, channels)
    sigma : tuple(float,float)
        Tuple of values corresponding to the standard deviation
        of the gaussian shaped psf. 
    b : float
        Background offset parameter. Can be a constant or same shape as Y.
        Must be positive. Default: 1e-5
    scale : float
        We can deconvolve the image data to higher/lower spatial resolution.
        Scale determines the scale factor. Defalut: 1.0 (no upscaling)

    Example
    ----------
        model = ISTDeco(Y, D, sigma, b=1e-5, upscale=1)
        model = model.to('cuda') # If we want GPU
        X, Q, loss = model.run(niter=100)
    '''

    def __init__(self, Y, D, sigma, b=1e-8, scale=1):

        self.input_shape = Y.shape
        self.sigma = sigma
        self.scale = scale

        m, r, c = D.shape
        _,_,self.h,self.w = Y.shape
        self.hx = int(np.ceil(self.h * scale)) if self.h > 1 else self.h
        self.wx = int(np.ceil(self.w * scale)) if self.w > 1 else self.w

        bh = int(2*np.ceil(3*sigma[0]*scale)+1) if self.h > 1 else 1
        bw =  int(2*np.ceil(3*sigma[1]*scale)+1) if self.w > 1 else 1
        self.psf_support_scaled = (bh, bw)

        bh = int(2*np.ceil(3*sigma[0])+1) if self.h > 1 else 1
        bw =  int(2*np.ceil(3*sigma[1])+1) if self.w > 1 else 1
        self.psf_support = (bh, bw)

        # Set up X
        x_shape = (m, self.hx, self.wx)
        self.X = torch.ones(x_shape).float()

        # Set up Y
        self.Y = torch.tensor(Y).float().flatten(start_dim=0, end_dim=1)

        # Set up b
        self.b = torch.tensor(b).float()
        if self.b.ndim == 4:
            self.b = self.b.flatten(start_dim=0, end_dim=1)
        
        # Set up codebook
        self.codebook = Codebook(D)

        # Set up spatial blurr
        self.psf = PSF((self.h,self.w),sigma,scale)

        # Prepare constant
        ones_channels = torch.ones((r*c, 1, 1))
        ones_space = torch.ones((1, self.h, self.w))
        self.denominator =  self.codebook.matmul_t(ones_channels) * \
                            self.psf.matmul_t(ones_space)

        # Compute Yhat = DXG
        self.Yhat = self.__forward(self.X)

    def to(self, device):
        '''
            Puts tensors on a device. See pytorch doc for more info.
            Useful for moving tensors to cuda.

            Example
            ----------
                model = ISTDECO(Y,D,sigma)
                model.to('cuda')    # Put tensors on GPU
                model.to('cpu')     # Put tensors on CPU    

            Parameters
            ----------
                device : str
                    The device, for instance 'cpu' or 'cuda'

        '''
        self.Y = self.Y.to(device)
        self.Yhat = self.Yhat.to(device)
        self.X = self.X.to(device)
        self.denominator = self.denominator.to(device)
        self.codebook = self.codebook.to(device)
        self.psf = self.psf.to(device)
        self.b = self.b.to(device)
        return self        

    def __forward(self, tensor):
        return self.psf.matmul(self.codebook.matmul(tensor)) + self.b

    def __compute_quality(self, tensor):
        # Pool intensities spatially
        
        tensor_blurr = torch.nn.functional.avg_pool2d(tensor, \
            self.psf_support_scaled,\
            stride=1,\
            divisor_override=1,\
            padding=tuple(t//2 for t in self.psf_support_scaled))


        tensor_blurr2 = self.psf.up_pool(torch.nn.functional.relu(self.Y - self.b))

        # Compute quality feature
        Q = tensor_blurr / self.codebook.matmul_t(tensor_blurr2)
        Q[torch.isnan(Q)] = 0
        return Q
        

    def run(self, niter=100, acc=1.0, suppress_radius=1):
        '''
            Run the optimization
            
            Parameters
            ----------
                niter : int 
                    Number of iterations

                acc : float
                    Factor for acceleration the multiplicative updates
                    If too large, a convergence may be unstalbe. Usually a 
                    value between 1.0 and 1.5 is fine. Default: 1.0
                suppress_width : int
                    Integer indicating the radius of the non-maxima supression footprint.
                    Default: 1. 

            Outputs
            ---------
                X : numpy array 
                    A multi-channel image of shape (m, sy, sx) where
                    m is the number of barcodes, sy and sx are the scaled height and width.
                    The values in X corresponds to the intensity of different barcodes.
                    For instance, X_kij is the intensity of the barcode with code k, localized
                    at i and j.

                Q : numpy array
                    A multi-channel image of shape (m, sy, sx) where
                    m is the number of barcodes, sy and sx are the scaled height and width.
                    The values in Q are useful for elimintaing false-positive detections during
                    pos-processing.

                loss : numpy array
                    An (niter,) shaped array with values 
        '''
        loss = torch.zeros((niter,))
        for i in range(niter):
            loss[i] = torch.sum(self.Yhat) - \
                torch.sum(self.Y*torch.log(self.Yhat+1e-9))
            self.X = self.X * (self.codebook.matmul_t(self.psf.matmul_t(self.Y / self.Yhat)) / self.denominator)**acc
            self.Yhat = self.__forward(self.X)
        Q = self.__compute_quality(self.X)
        if suppress_radius is not None:
            mask = self.__nonmaxsuppress(suppress_radius, self.X)
            self.X = self.X * mask
            Q = Q * mask
        return self.X.cpu().numpy(), Q.cpu().numpy(), loss


    def __nonmaxsuppress(self, radius, tensor):
        padd = [radius if self.h > 1 else 0, radius]
        kernel_sz = (2*radius+1 if self.h > 1 else 1, 2*radius+1)
        mask = torch.nn.functional.max_pool2d(tensor, kernel_sz, stride=1, padding=padd) == tensor
        #ints = torch.nn.functional.avg_pool2d(tensor, kernel_sz, stride=1, padding=padd, divisor_override=1)
        return mask




