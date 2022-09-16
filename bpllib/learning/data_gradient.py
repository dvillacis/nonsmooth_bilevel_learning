import numpy as np
from bpllib.operators.ActiveOp import ActiveOp
from bpllib.operators.Gradient import Gradient
from bpllib.operators.InactiveOp import InactiveOp
from bpllib.operators.TOp import TOp
from bpllib.operators.Tgamma import Tgamma
from pylops import Diagonal, Identity, Block
from pylops.optimization.solver import cgls, lsqr, cg
import scipy.sparse.linalg as spla

def data_adjoint_smooth(parameter,original,denoised):
    parameter = parameter.map_to_img(original)
    nx,ny = original.shape
    n = nx*ny
    K = Gradient(dims=(nx,ny))
    nkx,nky = K.shape
    L = Diagonal(parameter)
    T = Tgamma(denoised)
    Id = Identity(nkx)
    # print(L.shape,K.adjoint().shape,T.shape,Id.shape)
    A = Block([[L,K.adjoint()],[-T,Id]])
    b = np.concatenate((denoised.ravel()-original.ravel(),np.zeros(nkx)))
    p = spla.spsolve(A.tosparse(),b)
    return p[:n]

def data_gradient_smooth(parameter,original,noisy,denoised):
    p = data_adjoint_smooth(parameter,original,denoised)
    L = Diagonal(p)
    grad = L*(denoised.ravel()-noisy.ravel())
    grad = parameter.reduce_from_img(grad.reshape(original.shape))
    return -grad
   
def data_adjoint_nonsmooth(parameter,original,denoised,show=False):
    parameter = parameter.map_to_img(original)
    nx,ny = original.shape
    n = nx*ny
    K = Gradient(dims=(nx,ny))
    nkx,nky = K.shape
    L = Diagonal(parameter)
    T = TOp(denoised)
    Act = ActiveOp(denoised)
    Inact = InactiveOp(denoised)
    A = Block([[L,K.adjoint()],[Act*K-Inact*T,Inact-1e-12*Act]])
    b = np.concatenate((denoised.ravel()-original.ravel(),np.zeros(nkx)))
    p = spla.spsolve(A.tosparse(),b)
    return p[:n]
     
def data_gradient_nonsmooth(parameter,original,noisy,denoised):
    p = data_adjoint_nonsmooth(parameter,original,denoised)
    L = Diagonal(p)
    grad = L*(denoised.ravel()-noisy.ravel())
    grad = parameter.reduce_from_img(grad.reshape(original.shape))
    return -grad