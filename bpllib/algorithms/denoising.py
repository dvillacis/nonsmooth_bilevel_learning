import numpy as np
import pylops
from pyproximal import L1, L2, L21
from pyproximal.optimization.primal import LinearizedADMM, ProximalGradient
from pyproximal.optimization.primaldual import PrimalDual

from bpllib.operators.FirstDerivative import FirstDerivative
from bpllib.operators.Gradient import Gradient
from bpllib.operators.Patch import Patch
from bpllib.operators.SDL2 import SDL2
from bpllib.operators.SDL21 import SDL21

def denoise_1d_ladmm(signal,sigma=5.0,tau=1.0):
    l2 = L2(b=signal)
    l1 = L1(sigma=5.)
    Dop = FirstDerivative(len(signal))
    
    L = np.real((Dop.H * Dop).eigs(neigs=1, which='LM')[0])
    mu = 0.99 * tau / L
    xladmm, _ = LinearizedADMM(l2,l1,Dop,tau=tau,mu=mu,x0=np.zeros_like(signal),niter=200)
    return xladmm

def denoise_tv_2d_pdhg(noisy,sigma1=1.0,sigma2=0.1):
    nx,ny = noisy.shape
    # Gop = pylops.Gradient(dims=(nx,ny),sampling=1.0, edge=False, kind='forward',dtype=np.float64)
    Gop = Gradient(dims=(ny,nx),kind='forward')
    # L = np.real((Gop.H * Gop).eigs(neigs=1, which='LM')[0])
    L = 8.0
    # Data Fidelity term
    l2 = L2(b=noisy.ravel(),sigma=sigma1)
    # Regularization term
    l21 = L21(ndim=2,sigma=sigma2)
    # Meta-parameters
    tau = 1.0 / np.sqrt(L)
    mu = 1.0 / (tau*L)
    # Solving 
    imtv = PrimalDual(l2,l21,Gop,tau=tau,mu=mu,theta=1.0,x0=np.zeros_like(noisy.ravel()),niter=200,show=True)
    return imtv.reshape((nx,ny))

def denoise_tv_2d_pdhg_patch(noisy,data_parameter:Patch,reg_parameter:Patch,show=False):
    '''
    Patch Denoising using Primal-Dual Hibrid Gradient Algorithm
    
    Input
    -----
    noisy: 2D ndarray: Image corrupted with noise
    
    Output
    -----
    imtv: 2D ndarray: Reconstructed image
    '''
    nx,ny = noisy.shape
    # Getting mapped parameters
    data_parameter = data_parameter.map_to_img(noisy)
    reg_parameter = reg_parameter.map_to_img(noisy[:-1,:-1])
    #print(f'param_shapes:{reg_parameter.shape},{data_parameter.shape}')
    # Generating components
    Gop = Gradient(dims=(ny,nx))
    l2 = SDL2(b=noisy.ravel(),sigma=data_parameter.ravel())
    l21 = SDL21(ndim=2,sigma=reg_parameter.ravel())
    L = 8.0
    tau = 1.0 / np.sqrt(L)
    mu = 1.0 / (tau*L)
    # Solving
    imtv = PrimalDual(l2,l21,Gop,tau=tau,mu=mu,theta=1.0,x0=np.zeros_like(noisy.ravel()),niter=200,show=show)
    return imtv.reshape((nx,ny))
    
