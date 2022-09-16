import numpy as np
import scipy.sparse.linalg as spla
from pylops import Diagonal, Identity, Block, Zero

from bpllib.operators.FirstDerivative import FirstDerivative
from bpllib.operators.Gradient import Gradient
from bpllib.operators.ActiveOp import ActiveOp
from bpllib.operators.InactiveOp import InactiveOp
from bpllib.operators.TOp import TOp
from bpllib._utils import tv_smooth_subdiff
from bpllib.operators.Tgamma import Tgamma
from bpllib._utils import spai

def reg_adjoint_smooth(parameter,original,denoised):
    parameter = parameter.map_to_img(original[:-1,:-1])
    nx,ny = original.shape
    n = nx*ny
    K = Gradient(dims=(nx,ny))
    nkx,nky = K.shape
    L = Diagonal(np.concatenate((parameter,parameter)))
    T = Tgamma(denoised)
    Id = Identity(n)
    Id2 = Identity(nkx)
    A = Block([[Id,K.adjoint()],[-L*T,Id2]])
    b = np.concatenate((denoised.ravel()-original.ravel(),np.zeros(nkx)))
    p = spla.spsolve(A.tosparse(),b)
    # print(f'cond:{np.linalg.cond(A.todense())}')
    # print(f'res:{np.linalg.norm(A*p-b)}')
    return p[:n]

def reg_gradient_smooth(parameter,original,noisy,denoised):
    p = reg_adjoint_smooth(parameter,original,denoised)
    nx,ny = original.shape
    n = nx*ny
    Kx = FirstDerivative(n,dims=(nx,ny),dir=0)
    Ky = FirstDerivative(n,dims=(nx,ny),dir=1)
    Kxp = Kx*p
    Kyp = Ky*p
    hx,hy = tv_smooth_subdiff(denoised,gamma=1000)
    #grad = L*(reconstruction.ravel()-noisy.ravel())
    grad = -(hx*Kxp + hy*Kyp)
    grad = grad.reshape((nx-1,ny-1))
    grad = np.pad(grad,[(0,1),(0,1)],mode='edge')
    grad = parameter.reduce_from_img(grad)
    return grad
    
def reg_adjoint_nonsmooth(parameter,original,denoised):
    parameter = parameter.map_to_img(original[:-1,:-1])
    nx,ny = original.shape
    n = nx*ny
    K = Gradient(dims=(nx,ny))
    nkx,nky = K.shape
    Id = Identity(n)
    L = Diagonal(np.concatenate((parameter,parameter)))
    Linv = L = Diagonal(np.concatenate(((1/parameter),(1/parameter))))
    Act = ActiveOp(denoised)
    T = TOp(denoised)
    Inact = InactiveOp(denoised)
    Z = Zero(nkx,n)
    Z2 = Zero(n,nkx)
    A = Block([[Id,K.adjoint()],[-T,Linv*(Inact+1e-12*Act)]]).tosparse()
    # print(Z.shape,Linv.shape,(K*K.adjoint()).shape)
    M = Block([[Id,Z2],[Z,Linv*(Inact+1e-12*Act)+K*K.adjoint()]]).tosparse()
    # M = spai(A.tosparse(),1000)
    # print(f'schur cn:{np.linalg.cond((Linv+K*K.adjoint()).todense())}')
    b = np.concatenate((denoised.ravel()-original.ravel(),np.zeros(nkx)))
    # A = M.multiply(A)
    # b = M.dot(b)
    p= spla.lsmr(A,b,atol=0)[0]
    # p = spla.spsolve(A,b)
    # print(f'cond:{np.linalg.cond(A.todense())}')
    # print(f'res:{np.linalg.norm(A*p-b)}')
    return p[:n]
    
def reg_gradient_nonsmooth(parameter,original,noisy,denoised,tol=1e-6):
    p = reg_adjoint_nonsmooth(parameter,original,denoised)
    nx,ny = original.shape
    n = nx*ny
    Kx = FirstDerivative(n,dims=(nx,ny),dir=0)
    Ky = FirstDerivative(n,dims=(nx,ny),dir=1)
    Kxu = Kx*denoised.ravel()
    Kyu = Ky*denoised.ravel()
    Kxp = Kx*p
    Kyp = Ky*p
    nKu = np.linalg.norm(np.vstack((Kxu,Kyu)).T,axis=1)
    mul = np.where(nKu<tol,0,-1/nKu)
    grad = mul * (Kxu * Kxp + Kyu * Kyp)
    grad = grad.reshape((nx-1,ny-1))
    grad = np.pad(grad,[(0,1),(0,1)],mode='edge')
    grad = parameter.reduce_from_img(grad)
    return grad