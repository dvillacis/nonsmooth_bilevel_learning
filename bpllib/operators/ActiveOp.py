import numpy as np
from pylops import LinearOperator
from pylops.utils.backend import get_array_module
from bpllib.operators.Gradient import Gradient
from bpllib._utils import pointwise_euclidean_norm


class ActiveOp(LinearOperator):
    def __init__(self, u, tol=1e-5, dtype='float64'):
        nx,ny = u.shape
        K = Gradient(dims=(nx,ny))
        Ku = K*u.ravel()
        nKu = pointwise_euclidean_norm(Ku)
        d = np.where(nKu < tol,1,0)
        self.diag = np.concatenate((d,d),axis=0)
        self.shape = (len(self.diag),len(self.diag))
        self.dtype = np.dtype(dtype)

    def _matvec(self, x):
        y = self.diag * x
        return y

    def _rmatvec(self, x):
        y = self.diag * x
        return y

    def matrix(self):
        ncp = get_array_module(self.diag)
        densemat = ncp.diag(self.diag.squeeze())
        return densemat

    def todense(self):
        return self.matrix()