from functools import partial
import numpy as np
import pytest
from scipy.optimize import rosen, rosen_der
import matplotlib.pyplot as plt

from bpllib.learning.cost_functions import l2_cost, ssim_cost, psnr_cost
from bpllib.learning.gradient import gradient
from bpllib.algorithms.trustregion_box import trustregion_box
from bpllib.algorithms.denoising import denoise_tv_2d_pdhg_patch
from bpllib.operators.Patch import Patch

def test_tr_rosen():
    # Trust region test in rosenbrock function
    x0 = np.array([1.1,2.5])
    x_opt = trustregion_box(rosen,rosen_der,rosen_der,x0)
    print(x_opt)
    
def test_tr_bilevel_data():
    # Trust region test in rosenbrock function
    
    img = np.tril(np.ones((8,8)))
    # add noise
    sigman = 0.5
    n = sigman * np.max(abs(img.ravel())) * np.random.uniform(-1,1,img.shape)
    noisy = img + n
    
    px=1
    py=1
    x0 = 0.5*np.ones(px*py)
    x_opt = trustregion_box(
        partial(l2_cost,original=img,noisy=noisy,is_data_parameter=True,px=px,py=py),
        partial(gradient,original=img,noisy=noisy,is_data_parameter=True,px=px,py=py),
        partial(gradient,original=img,noisy=noisy,is_data_parameter=True,is_smooth=True,px=px,py=py),
        x0
    )
    print(x_opt)
    par = Patch(x_opt.x,px,py)
    par_img = par.map_to_img(img).reshape(img.shape)
    # par_img = par_img / np.max(par_img)
    ones_par = Patch(np.ones_like(x_opt.x),px,py)
    imtv = denoise_tv_2d_pdhg_patch(noisy=noisy,data_parameter=par,reg_parameter=ones_par)
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    axs[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[0].axis('tight')
    axs[1].imshow(noisy, cmap='gray', vmin=0, vmax=1)
    axs[1].set_title('Noisy')
    axs[1].axis('off')
    axs[1].axis('tight')
    axs[2].imshow(imtv, cmap='gray', vmin=0, vmax=1)
    axs[2].set_title('TViso')
    axs[2].axis('off')
    axs[2].axis('tight')
    pcm = axs[3].imshow(par_img, cmap='gray')
    axs[3].set_title('Patch')
    axs[3].axis('off')
    axs[3].axis('tight')
    fig.colorbar(pcm,ax=axs[3])
    plt.tight_layout()
    plt.show()
    
def test_tr_bilevel_reg():
    # Trust region test in rosenbrock function
    np.random.seed(0)
    n = 128
    img = np.tril(np.ones((n,n)))
    # add noise
    noisy = np.abs(img + 0.5*np.tril((np.random.rand(n,n)-0.5)))
    
    px=32
    py=32
    x0 = 0.01*np.ones(px*py)
    x_opt = trustregion_box(
        partial(l2_cost,original=img,noisy=noisy,is_data_parameter=False,px=px,py=py),
        partial(gradient,original=img,noisy=noisy,is_data_parameter=False,is_smooth=False,px=px,py=py),
        partial(gradient,original=img,noisy=noisy,is_data_parameter=False,is_smooth=True,px=px,py=py),
        x0
    )
    print(x_opt)
    par = Patch(x_opt.x,px,py)
    print(par)
    par_img = par.map_to_img(img).reshape(img.shape)
    # par_img = par_img / np.max(par_img)
    ones_par = Patch(np.ones_like(x_opt.x),px,py)
    imtv = denoise_tv_2d_pdhg_patch(noisy=noisy,data_parameter=ones_par,reg_parameter=par)
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    axs[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[0].axis('tight')
    axs[1].imshow(noisy, cmap='gray', vmin=0, vmax=1)
    axs[1].set_title('Noisy')
    axs[1].axis('off')
    axs[1].axis('tight')
    axs[2].imshow(imtv, cmap='gray', vmin=0, vmax=1)
    axs[2].set_title('TViso')
    axs[2].axis('off')
    axs[2].axis('tight')
    pcm = axs[3].imshow(par_img, cmap='gray')
    axs[3].set_title('Patch')
    axs[3].axis('off')
    axs[3].axis('tight')
    fig.colorbar(pcm,ax=axs[3])
    plt.tight_layout()
    plt.show()
    
def test_bilevel_cost_function_reg():
    # Trust region with custom gradient and cost function
    img = np.tril(np.ones((128,128)))
    # add noise
    sigman = 0.5
    n = sigman * np.max(abs(img.ravel())) * np.random.uniform(-1,1,img.shape)
    noisy = img + n
    
    costs = []
    ssims = []
    psnrs = []
    
    ran = np.arange(1e-5,4.0,0.01)
    for x in ran:
        cost = l2_cost(x,img,noisy)
        ssim = ssim_cost(x,img,noisy)
        psnr = psnr_cost(x,img,noisy)
        print(cost)
        costs.append(cost)
        ssims.append(ssim)
        psnrs.append(psnr)
        
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(ran,costs)
    axs[0].set_title('L2')
    axs[0].grid()
    axs[0].axis('tight')
    axs[1].plot(ran,ssims)
    axs[1].set_title('SSIM')
    axs[1].grid()
    axs[1].axis('tight')
    axs[2].plot(ran,psnrs)
    axs[2].set_title('PSNR')
    axs[2].grid()
    axs[2].axis('tight')
    plt.tight_layout()
    plt.show()
    # grad = 
    # reg_grad=
    # x_opt = trustregion_box(cost,grad,reg_grad,x0)
    
def test_bilevel_cost_function_data():
    # Trust region with custom gradient and cost function
    img = np.tril(np.ones((128,128)))
    # add noise
    sigman = 0.5
    n = sigman * np.max(abs(img.ravel())) * np.random.uniform(-1,1,img.shape)
    noisy = img + n
    
    costs = []
    ssims = []
    psnrs = []
    
    ran = np.arange(0.1,20.0,0.1)
    for x in ran:
        cost = l2_cost(x,img,noisy,is_data_parameter=True)
        ssim = ssim_cost(x,img,noisy,is_data_parameter=True)
        psnr = psnr_cost(x,img,noisy,is_data_parameter=True)
        print(cost)
        costs.append(cost)
        ssims.append(ssim)
        psnrs.append(psnr)
        
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(ran,costs)
    axs[0].set_title('L2')
    axs[0].grid()
    axs[0].axis('tight')
    axs[1].plot(ran,ssims)
    axs[1].set_title('SSIM')
    axs[1].grid()
    axs[1].axis('tight')
    axs[2].plot(ran,psnrs)
    axs[2].set_title('PSNR')
    axs[2].grid()
    axs[2].axis('tight')
    plt.tight_layout()
    plt.show()
    
def test_bilevel_gradient_data():
    np.random.seed(0)
    img = np.tril(np.ones((16,16)))
    # add noise
    sigman = 0.5
    n = sigman * np.max(abs(img.ravel())) * np.random.uniform(-1,1,img.shape)
    noisy = img + n
    
    ran = np.arange(1,20,0.1)
    grads = []
    costs = []
    for x in ran:
        cost = l2_cost(x,img,noisy,is_data_parameter=True)
        g = gradient(x,img,noisy,is_data_parameter=True,is_smooth=False)
        grads.append(g)
        costs.append(cost)
        print(x,cost,g)
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].plot(ran,costs)
    axs[0].set_title('L2')
    axs[0].grid()
    axs[0].axis('tight')
    axs[1].plot(ran,grads)
    axs[1].set_title('grad')
    axs[1].grid()
    axs[1].axis('tight')
    plt.show()
    

def test_bilevel_smooth_gradient_data():
    np.random.seed(0)
    img = np.tril(np.ones((64,64)))
    # add noise
    sigman = 0.1
    n = sigman * np.max(abs(img.ravel())) * np.random.uniform(-1,1,img.shape)
    noisy = img + n
    
    ran = np.arange(1,20.0,0.1)
    grads = []
    costs = []
    for x in ran:
        cost = l2_cost(x,img,noisy,is_data_parameter=True)
        g = gradient(x,img,noisy,is_data_parameter=True,is_smooth=True)
        grads.append(g)
        costs.append(cost)
        print(x,cost,g)
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].plot(ran,costs)
    axs[0].set_title('L2')
    axs[0].grid()
    axs[0].axis('tight')
    axs[1].plot(ran,grads)
    axs[1].set_title('grad')
    axs[1].grid()
    axs[1].axis('tight')
    plt.show()

def test_bilevel_gradient_reg():
    np.random.seed(0)
    n = 8
    img = np.tril(np.ones((8,8)))
    # add noise
    sigman = 0.1
    noisy = np.abs(img + 0.5*np.tril((np.random.rand(n,n)-0.5)))
    
    ran = np.arange(1e-9,2.0,0.01)
    grads = []
    costs = []
    for x in ran:
        cost = l2_cost(x,img,noisy,is_data_parameter=False)
        g = gradient(x,img,noisy,is_data_parameter=False,is_smooth=False)
        grads.append(g)
        costs.append(cost)
        print(x,cost,g)
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].plot(ran,costs)
    axs[0].set_title('L2')
    axs[0].grid()
    axs[0].axis('tight')
    axs[1].plot(ran,grads)
    axs[1].set_title('grad')
    axs[1].grid()
    axs[1].axis('tight')
    plt.show()
    
def test_bilevel_smooth_gradient_reg():
    np.random.seed(0)
    n = 8
    img = np.tril(np.ones((n,n)))
    # add noise
    sigman = 0.1
    noisy = np.abs(img + 0.5*np.tril((np.random.rand(n,n)-0.5)))
    
    ran = np.arange(1e-9,2.0,0.01)
    grads = []
    costs = []
    for x in ran:
        cost = l2_cost(x,img,noisy,is_data_parameter=False)
        g = gradient(x,img,noisy,is_data_parameter=False,is_smooth=True)
        grads.append(g)
        costs.append(cost)
        print(x,cost,g)
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].plot(ran,costs)
    axs[0].set_title('L2')
    axs[0].grid()
    axs[0].axis('tight')
    axs[1].plot(ran,grads)
    axs[1].set_title('grad')
    axs[1].grid()
    axs[1].axis('tight')
    plt.show()
    