import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from bpllib.operators.Patch import Patch
from bpllib.algorithms.denoising import denoise_tv_2d_pdhg_patch
from bpllib._utils import _to_array



def l2_cost(parameter,original,noisy,is_data_parameter=False,px=1,py=1):
    
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
    
    return 0.5*np.linalg.norm(original.ravel()-denoised.ravel())**2

def ssim_cost(parameter,original,noisy,is_data_parameter=False,px=1,py=1):
    
    parameter = _to_array(parameter,'parameter')
    
    # Setting up parameters
    if is_data_parameter:
        data_parameter = Patch(parameter,px,py)
        reg_parameter = Patch(np.ones_like(parameter),px,py)
    else:
        data_parameter = Patch(np.ones_like(parameter),px,py)
        reg_parameter = Patch(parameter,px,py)
        
    # Denoising
    denoised = denoise_tv_2d_pdhg_patch(noisy=noisy,data_parameter=data_parameter,reg_parameter=reg_parameter)
    
    return ssim(original,denoised)

def psnr_cost(parameter,original,noisy,is_data_parameter=False,px=1,py=1):
    
    parameter = _to_array(parameter,'parameter')
    
    # Setting up parameters
    if is_data_parameter:
        data_parameter = Patch(parameter,px,py)
        reg_parameter = Patch(np.ones_like(parameter),px,py)
    else:
        data_parameter = Patch(np.ones_like(parameter),px,py)
        reg_parameter = Patch(parameter,px,py)
        
    # Denoising
    denoised = denoise_tv_2d_pdhg_patch(noisy=noisy,data_parameter=data_parameter,reg_parameter=reg_parameter)
    
    return psnr(original,denoised)