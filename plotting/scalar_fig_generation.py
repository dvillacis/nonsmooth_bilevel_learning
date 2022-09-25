import sys,os
from typing import ChainMap
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def generate_scalar_figure(setting_file,outpath):
    
    setting_file_basename = setting_file.split(os.path.sep)[-1].replace('.json', '')
    resultdir = os.path.join('raw_results',setting_file_basename)
    if not os.path.isfile(setting_file):
        raise RuntimeError('Cannot find results folder: %s' % resultdir)
    recons = np.load(os.path.join(resultdir,'%s_recons.npy' % (setting_file_basename)))
    true_imgs = np.load(os.path.join(resultdir,'%s_true_imgs.npy' % (setting_file_basename)))
    noisy_imgs = np.load(os.path.join(resultdir,'%s_noisy_imgs.npy' % (setting_file_basename)))
    
    num_imgs = recons.shape[2]
    
    if num_imgs > 1:
        fig,axs = plt.subplots(2,num_imgs)
        for i in range(num_imgs):
            l2_rec = 0.5 * np.linalg.norm(true_imgs[:,:,i].ravel() - recons[:,:,i].ravel())**2
            ssim_rec = ssim(true_imgs[:,:,i],recons[:,:,i])
            psnr_rec = psnr(true_imgs[:,:,i],recons[:,:,i])
            l2_noisy = 0.5 * np.linalg.norm(true_imgs[:,:,i].ravel() - noisy_imgs[:,:,i].ravel())**2
            ssim_noisy = ssim(true_imgs[:,:,i],noisy_imgs[:,:,i])
            psnr_noisy = psnr(true_imgs[:,:,i],noisy_imgs[:,:,i])
            axs[0,i].imshow(noisy_imgs[:,:,i],cmap='gray',vmin=0,vmax=1)
            axs[0,i].set_xticklabels([])
            axs[0,i].set_xticks([])
            axs[0,i].set_yticks([])
            axs[0,i].set_yticklabels([])
            axs[0,i].set_xlabel(f'PSNR={np.mean(psnr_noisy):.4f}\nSSIM={np.mean(ssim_noisy):.4f}')
            axs[1,i].imshow(recons[:,:,i],cmap='gray',vmin=0,vmax=1)
            axs[1,i].set_xticklabels([])
            axs[1,i].set_xticks([])
            axs[1,i].set_yticks([])
            axs[1,i].set_yticklabels([])
            axs[1,i].set_xlabel(f'PSNR={np.mean(psnr_rec):.4f}\nSSIM={np.mean(ssim_rec):.4f}')
    else:
        fig,axs = plt.subplots(3,num_imgs)
        l2_rec = 0.5 * np.linalg.norm(true_imgs[:,:,0].ravel() - recons[:,:,0].ravel())**2
        ssim_rec = ssim(true_imgs[:,:,0],recons[:,:,0])
        psnr_rec = psnr(true_imgs[:,:,0],recons[:,:,0])
        l2_noisy = 0.5 * np.linalg.norm(true_imgs[:,:,0].ravel() - noisy_imgs[:,:,0].ravel())**2
        ssim_noisy = ssim(true_imgs[:,:,0],noisy_imgs[:,:,0])
        psnr_noisy = psnr(true_imgs[:,:,0],noisy_imgs[:,:,0])
        axs[0].imshow(true_imgs[:,:,0],cmap='gray',vmin=0,vmax=1)
        axs[0].set_xticklabels([])
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_yticklabels([])
        axs[1].imshow(noisy_imgs[:,:,0],cmap='gray',vmin=0,vmax=1)
        axs[1].set_xticklabels([])
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_yticklabels([])
        axs[1].set_xlabel(f'PSNR={np.mean(psnr_noisy):.4f}\nSSIM={np.mean(ssim_noisy):.4f}')
        axs[2].imshow(recons[:,:,0],cmap='gray',vmin=0,vmax=1)
        axs[2].set_xticklabels([])
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[2].set_yticklabels([])
        axs[2].set_xlabel(f'PSNR={np.mean(psnr_rec):.4f}\nSSIM={np.mean(ssim_rec):.4f}')
    plt.show()
    # tikzplotlib.save(os.path.join(outpath,'scalar_comparison_plot.tex'))

def main():
    if len(sys.argv) < 3:
        print("Usage: python %s settings_file outfolder" % sys.argv[0])
        print("where")
        print("    settings_files = json file containing run details")
        print("    outfolder = folder to save plots to")
        exit()
    
    # List setting files
    num_setting_files = len(sys.argv)-2
    setting_files = [sys.argv[i] for i in range(1,1+num_setting_files)]
    print(f'Generating patch comparison plot with:\n{setting_files}')
    for setting_file in setting_files:
        if not os.path.isfile(setting_file):
            raise RuntimeError('Settings file does not exist: %s' % setting_file)
    outfolder = sys.argv[-1]
    outpath = os.path.join('output_table',outfolder)
    if not os.path.isdir(outpath):
        os.makedirs(outpath, exist_ok=True)

    generate_scalar_figure(setting_files[0],outpath)

if __name__ == '__main__':
    main()