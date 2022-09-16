import numpy as np
import pytest

from bpllib.operators.FirstDerivative import FirstDerivative
from bpllib.operators.Gradient import Gradient

def test_FirstDerivative_1D():
    # One dimensional FD test
    Kx = FirstDerivative(9,dir=1)
    print(Kx.shape)
    print(Kx.todense())
    print(np.linalg.cond(Kx.todense()))
    

def test_FirstDerivative_2D():
    # Two dimensional FD test
    Kx = FirstDerivative(9, dims=(3,3), dir=0)
    print(Kx.shape)
    print(f'Kx:\n{Kx.todense()}')
    print(np.linalg.cond(Kx.todense()))
    
    Ky = FirstDerivative(9, dims=(3,3), dir=1)
    print(Ky.shape)
    print(f'Ky:\n{Ky.todense()}')
    print(np.linalg.cond(Ky.todense()))
    
def test_Gradient():
    # Test Gradient operator
    Gop = Gradient(dims=(3,3))
    print(Gop.shape)
    print(f'G:{Gop.todense()}')
    print(np.linalg.cond(Gop.todense()))
    
def test_Gradient_img():
    img = np.array([[1,2,3],[1,2,3],[1,2,3]])
    Gop = Gradient(dims=(3,3))
    print(Gop.shape)
    Gimg = Gop*img.ravel()
    print(Gimg)
    
    
