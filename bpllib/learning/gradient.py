import numpy as np
from bpllib._utils import _to_array
from bpllib.operators.Patch import Patch
from bpllib.algorithms.denoising import denoise_tv_2d_pdhg_patch
from bpllib.learning.data_gradient import data_gradient_nonsmooth, data_gradient_smooth
from bpllib.learning.reg_gradient import reg_gradient_nonsmooth, reg_gradient_smooth

def gradient(parameter,original,noisy,is_data_parameter=False,is_smooth=False,px=1,py=1):
    parameter = _to_array(parameter,'parameter')
    if len(parameter) != px*py:
        raise ValueError('Parameter has incorrect dimensions')
    
    # Setting up parameters
    if is_data_parameter:
        data_parameter = Patch(parameter,px,py)
        reg_parameter = Patch(np.ones_like(parameter),px,py)
    else:
        data_parameter = Patch(np.ones_like(parameter),px,py)
        reg_parameter = Patch(parameter,px,py)
        
    # Denoising
    denoised = denoise_tv_2d_pdhg_patch(noisy=noisy,data_parameter=data_parameter,reg_parameter=reg_parameter)
    
    if is_data_parameter and is_smooth:
        return data_gradient_smooth(data_parameter,original,noisy,denoised)
    elif is_data_parameter and not is_smooth:
        return data_gradient_nonsmooth(data_parameter,original,noisy,denoised)
    elif not is_data_parameter and is_smooth:
        return reg_gradient_smooth(reg_parameter,original,noisy,denoised)
    else:
        return reg_gradient_nonsmooth(reg_parameter,original,noisy,denoised)