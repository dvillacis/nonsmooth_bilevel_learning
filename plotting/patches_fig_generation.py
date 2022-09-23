import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def read_json(infile):
    with open(infile, 'r') as ifile:
        mydict = json.load(ifile)
    return mydict

def generate_patches_figure(setting_files,outpath,img_to_show=0):
    fig, axs = plt.subplots(2, len(setting_files)+1)
    axs[1,0].axis('off')
    for index,setting_file in enumerate(setting_files):
        setting = read_json(setting_file)
        px = int(setting["problem"]["px"])
        py = int(setting["problem"]["px"])
        setting_file_basename = setting_file.split(os.path.sep)[-1].replace('.json', '')
        resultdir = os.path.join('raw_results',setting_file_basename)
        if not os.path.isfile(setting_file):
            raise RuntimeError('Cannot find results folder: %s' % resultdir)
        recons = np.load(os.path.join(resultdir,'%s_recons.npy' % (setting_file_basename)))
        true_imgs = np.load(os.path.join(resultdir,'%s_true_imgs.npy' % (setting_file_basename)))
        noisy_imgs = np.load(os.path.join(resultdir,'%s_noisy_imgs.npy' % (setting_file_basename)))
        param = np.load(os.path.join(resultdir,'%s_optimal_par.npy' % (setting_file_basename)))
        l2_recs = []
        ssim_recs = []
        psnr_recs = []
        l2_noisy = []
        ssim_noisy = []
        psnr_noisy = []
        for i in range(recons.shape[2]):
            l2_recs.append(0.5 * np.linalg.norm(true_imgs[:,:,i].ravel() - recons[:,:,i].ravel())**2)
            ssim_recs.append(ssim(true_imgs[:,:,i],recons[:,:,i]))
            psnr_recs.append(psnr(true_imgs[:,:,i],recons[:,:,i]))
            l2_noisy.append(0.5 * np.linalg.norm(true_imgs[:,:,i].ravel() - noisy_imgs[:,:,i].ravel())**2)
            ssim_noisy.append(ssim(true_imgs[:,:,i],noisy_imgs[:,:,i]))
            psnr_noisy.append(psnr(true_imgs[:,:,i],noisy_imgs[:,:,i]))

        axs[0,0].imshow(noisy_imgs[:,:,img_to_show], cmap='gray', vmin=0, vmax=1)
        axs[0][0].set_xticklabels([])
        axs[0][0].set_xticks([])
        axs[0][0].set_yticks([])
        axs[0][0].set_yticklabels([])
        axs[0,0].set_xlabel(f'PSNR={np.mean(psnr_noisy):.4f}\nSSIM={np.mean(ssim_noisy):.4f}')

        axs[0,index+1].set_title(f'{px}x{py}')
        axs[0,index+1].imshow(recons[:,:,img_to_show], cmap='gray', vmin=0, vmax=1)
        axs[0,index+1].set_xlabel(f'PSNR={np.mean(psnr_recs):.4f}\nSSIM={np.mean(ssim_recs):.4f}')
        axs[0][index+1].set_xticklabels([])
        axs[0][index+1].set_xticks([])
        axs[0][index+1].set_yticks([])
        axs[0][index+1].set_yticklabels([])
        pcm = axs[1,index+1].imshow(param.reshape(px,py))
        axs[1][index+1].axis('off')
        fig.colorbar(pcm,ax=axs[1][index+1],orientation='horizontal')
    # plt.show()
    tikzplotlib.save(os.path.join(outpath,'patch_comparison_plot.tex'))

def main():
    if len(sys.argv) < 3:
        print("Usage: python %s settings_files outfolder" % sys.argv[0])
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

    generate_patches_figure(setting_files,outpath)

if __name__ == '__main__':
    main()