import numpy as np
import torch
import copy
class Codebook:
    '''
    Codebook - Helper class for doing arithmetics with the codebook

    Parameters
    ----------
    codebook : numpy array
        A ndarray of shape (m, rounds, channels)     
    '''

    def __init__(self, codebook):
        self.codebook = torch.from_numpy(codebook).float().flatten(start_dim=1).T
        self.codebook = self.codebook / self.codebook.sum(axis=0, keepdim=True)            

    def to(self, device):
        '''
            Put the tensors on a device.
            See PyTorch doc for more info.
        '''
        self.codebook = self.codebook.to(device)
        return self

    def matmul(self, tensor):
        '''
            Multiplies a (m,y,x) shaped tensor with the codebook.
            If the codebook is of shape (rc, m), then the output is of shape
            (rc,y,x)
        '''
        m, ny, nx = tensor.shape
        return (self.codebook @ tensor.view((m,ny*nx))).view((-1,ny,nx))

    def matmul_t(self,tensor):
        '''
            Multiploes a (rc,y,x) shaped tensor with the codebook.
            If the codebook is of shape (rc, m), then the output
            is of (m,y,x).
        '''
        m, ny, nx = tensor.shape
        return (self.codebook.t() @ tensor.view((m,ny*nx))).view((-1,ny,nx))
