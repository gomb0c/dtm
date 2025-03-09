"""
Library functions to perform circular convolution operations.
"""

__author__ = "Ashwinkumar Ganesan, Sunil Gandhi, Hang Gao"
__email__ = "gashwin1@umbc.edu,sunilga1@umbc.edu,hanggao@umbc.edu"

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
np.set_printoptions(threshold=sys.maxsize)

"""
Pytorch functions.
"""
def complex_multiplication(left, right):
    """
    Multiply two vectors in complex domain.
    """
    left_real, left_complex = left[..., 0], left[..., 1]
    right_real, right_complex = right[..., 0], right[..., 1]

    output_real = left_real * right_real - left_complex * right_complex
    output_complex = left_real * right_complex + left_complex * right_real
    return torch.stack([output_real, output_complex], dim=-1)

def complex_division(left, right):
    """
    Divide two vectors in complex domain.
    """
    left_real, left_complex = left[..., 0], left[..., 1]
    right_real, right_complex = right[..., 0], right[..., 1]

    output_real = torch.div((left_real * right_real + left_complex * right_complex),(right_real**2 + right_complex**2))
    output_complex = torch.div((left_complex * right_real - left_real * right_complex ),(right_real**2 + right_complex**2))
    return torch.stack([output_real, output_complex], dim=-1)

def circular_conv(a, b):
    """ Defines the circular convolution operation
    a: tensor of shape (batch, D)
    b: tensor of shape (batch, D)
    Assumes a and b are real-valued tensors
    """
    left = torch.fft.fft(a, dim=1)
    right = torch.fft.fft(b, dim=1)
    output = left * right # complex multiply in frequency domain
    output = torch.fft.ifft(output, n=a.shape[-1], dim=-1)
    return output.real

def circular_corr(a, b): 
      # Compute the FFT of both inputs along the last dimension.
    A = torch.fft.rfft(a, n=a.size(1))
    B = torch.fft.rfft(b, n=b.size(1))
    
    # Multiply A by the complex conjugate of B.
    prod = A * torch.conj(B)
    
    # Compute the inverse FFT to obtain the circular correlation in the time domain.
    y = torch.fft.irfft(prod, n=a.size(1))
    D = a.size(1)
    idxs = (-torch.arange(D)) % D
    return y[...,idxs] # cyclic shift


def get_appx_inv(a):
    """
    Compute approximate inverse of vector a.
    """
    return torch.roll(torch.flip(a, dims=[-1]), 1,-1)

def get_inv(a: torch.DoubleTensor, eps: float=1e-8):
    """
    Compute exact inverse of vector a.
    Old method from authors:
    left = torch.rfft(a, 1, onesided=False)
    complex_1 = np.zeros(left.shape)
    complex_1[...,0] = 1
    op = complex_division(typ(complex_1),left)
    return torch.irfft(op,1,onesided=False)
    """
    A = torch.fft.fft(a, dim=-1) # assume A is real
    A_inv = 1/ (A + eps) 
    a_inv = torch.fft.ifft(A_inv, dim=-1)
    return a_inv.real


def complexMagProj(x):
    """
    Normalize a vector x in complex domain.
    Assumes x real-valued tensor with last dimension as the signal domain
    """
    # 1. Forward real-to-complex FFT along the last dimension
    X_fft = torch.fft.rfft(x, dim=-1) 
    
    # 2. Normalize magnitudes
    X_fft_unit = X_fft / (X_fft.abs() + 1e-12)
    
    # Ensure that each element has unit magnitude \sqrt(Re^{2} + Im^{2}) \approx 1
    # np.testing.assert_array_almost_equal(torch.abs(c_norm), np.ones_like(c_norm))
    
    # 3. Inverse complex-to-real FFT, specifying the original signal length
    n = x.shape[-1]
    x_unitary = torch.fft.irfft(X_fft_unit, n=n, dim=-1)
    
    return x_unitary

def normalize(x):
    return x/torch.norm(x)

def _generate_vectors(num_vectors, dims):
    """
    Generate n vectors of size dims that are orthogonal to each other.
    """
    if num_vectors > dims:
        raise ValueError("num_vectors cannot be greater than dims!")

    # Intializing class vectors.
    vecs = torch.randn(dims, num_vectors, dtype=torch.float)

    # Using QR decomposition to get orthogonal vectors.
    vecs, _ = torch.linalg.qr(vecs)
    vecs = vecs.t()
    vecs = vecs / torch.norm(vecs, dim=-1, keepdim=True)
    return vecs

""" 
Procedure as outlined in https://arxiv.org/pdf/2109.02157, code from authors
"""
def generate_seed_vecs(n_vecs: int, dims: int, strict_orth: bool=False) -> torch.Tensor: 
    if strict_orth: 
        orth_randn_vecs = _generate_vectors(n_vecs, dims)
        #print(f'Orth randn vecs is {orth_randn_vecs.numpy()}')
        #print(f'After complex magnitude projecting, they are {complexMagProj(orth_randn_vecs).numpy()}')
        return complexMagProj(orth_randn_vecs)
    else: 
        randn_vecs = torch.randn(n_vecs, dims) / math.sqrt(dims)
        #print(f'Randn vecs is {randn_vecs.numpy()}')
        #print(f'After complex magnitude projecting, they are {complexMagProj(randn_vecs).numpy()}')
        return complexMagProj(randn_vecs)
        
        