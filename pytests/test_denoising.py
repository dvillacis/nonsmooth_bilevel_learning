import numpy as np
import pytest
import matplotlib.pyplot as plt
from scipy import misc
from skimage import data

from bpllib.algorithms.denoising import denoise_1d_ladmm, denoise_tv_2d_pdhg, denoise_tv_2d_pdhg_patch
from bpllib.operators.Patch import Patch

def test_ladmm_1d():
    nx = 101
    x = np.zeros(nx)
    x[:nx//2] = 10
    x[nx//2:3*nx//4] = -5
    n = np.random.normal(0, 2, nx)
    y = x + n
    den = denoise_1d_ladmm(y)
    print(den,len(den))
    # plt.plot(range(nx),x)
    # plt.plot(range(nx),y)
    # plt.plot(range(nx),den)
    # plt.show()
    
def test_tv_2d():
    img = misc.ascent()
    img = img / np.max(img)
    nx,ny = img.shape
    # add noise
    sigman = 0.2
    n = sigman * np.max(abs(img.ravel())) * np.random.uniform(-1,1,img.shape)
    noisy = img + n
    imtv = denoise_tv_2d_pdhg(noisy)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
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
    plt.tight_layout()
    plt.show()
    
def test_tv_2d_moon():
    img = data.moon()
    img = img / np.max(img)
    nx,ny = img.shape
    # add noise
    sigman = 0.3
    n = sigman * np.max(abs(img.ravel())) * np.random.uniform(-1,1,img.shape)
    noisy = img + n
    imtv = denoise_tv_2d_pdhg(noisy,sigma2=0.15)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
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
    plt.tight_layout()
    plt.show()
    
def test_tv_2d_pd_ascent():
    img = misc.ascent()
    img = img / np.max(img)
    nx,ny = img.shape
    # add noise
    sigman = 0.2
    n = sigman * np.max(abs(img.ravel())) * np.random.uniform(-1,1,img.shape)
    noisy = img + n
    data_parameter = Patch(np.array([1,1,1,1]),2,2)
    reg_parameter = Patch(np.array([0.9,0.1,0.05,0.001]),2,2)
    print(data_parameter,reg_parameter)
    imtv = denoise_tv_2d_pdhg_patch(noisy=noisy,data_parameter=data_parameter,reg_parameter=reg_parameter)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
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
    plt.tight_layout()
    plt.show()
    
def test_tv_2d_pd_moon():
    img = data.moon()
    img = img / np.max(img)
    nx,ny = img.shape
    # add noise
    sigman = 0.2
    n = sigman * np.max(abs(img.ravel())) * np.random.uniform(-1,1,img.shape)
    noisy = img + n
    data_parameter = Patch(np.array([1,1,1,1]),2,2)
    reg_parameter = Patch(np.linspace(1e-5,0.1,128*128),128,128)
    print(data_parameter,reg_parameter)
    imtv = denoise_tv_2d_pdhg_patch(noisy=noisy,data_parameter=data_parameter,reg_parameter=reg_parameter)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
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
    plt.tight_layout()
    plt.show()
    
def test_tv_2d_pd_moon_scalar():
    img = data.moon()
    img = img / np.max(img)
    nx,ny = img.shape
    # add noise
    sigman = 0.2
    n = sigman * np.max(abs(img.ravel())) * np.random.uniform(-1,1,img.shape)
    noisy = img + n
    data_parameter = Patch(np.array([1]),1,1)
    reg_parameter = Patch(np.array([0.1]),1,1)
    print(data_parameter,reg_parameter)
    imtv = denoise_tv_2d_pdhg_patch(noisy=noisy,data_parameter=data_parameter,reg_parameter=reg_parameter,show=True)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
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
    plt.tight_layout()
    plt.show()