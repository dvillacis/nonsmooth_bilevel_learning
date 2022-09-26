from argparse import ArgumentError
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

def generate_patches_comparison(data_setting_files,reg_setting_files,outpath):
    fig,axs = plt.subplots(2,len(data_setting_files))
    for index,setting_file in enumerate(data_setting_files):
        setting_data = read_json(setting_file)
        setting_reg = read_json(reg_setting_files[index])
        px = int(setting_data["problem"]["px"])
        py = int(setting_data["problem"]["px"])
        setting_file_basename_data = setting_file.split(os.path.sep)[-1].replace('.json', '')
        setting_file_basename_reg = reg_setting_files[index].split(os.path.sep)[-1].replace('.json', '')
        resultdir_data = os.path.join('raw_results',setting_file_basename_data)
        if not os.path.isfile(setting_file):
            raise RuntimeError('Cannot find results folder: %s' % resultdir_data)
        resultdir_reg = os.path.join('raw_results',setting_file_basename_reg)
        if not os.path.isfile(setting_file):
            raise RuntimeError('Cannot find results folder: %s' % resultdir_reg)
        
        # loading results
        recons_data = np.load(os.path.join(resultdir_data,'%s_recons.npy' % (setting_file_basename_data)))
        true_imgs_data = np.load(os.path.join(resultdir_data,'%s_true_imgs.npy' % (setting_file_basename_data)))
        noisy_imgs_data = np.load(os.path.join(resultdir_data,'%s_noisy_imgs.npy' % (setting_file_basename_data)))
        param_data = np.load(os.path.join(resultdir_data,'%s_optimal_par.npy' % (setting_file_basename_data)))
        
        recons_reg = np.load(os.path.join(resultdir_reg,'%s_recons.npy' % (setting_file_basename_reg)))
        true_imgs_reg = np.load(os.path.join(resultdir_reg,'%s_true_imgs.npy' % (setting_file_basename_reg)))
        noisy_imgs_reg = np.load(os.path.join(resultdir_reg,'%s_noisy_imgs.npy' % (setting_file_basename_reg)))
        param_reg = np.load(os.path.join(resultdir_reg,'%s_optimal_par.npy' % (setting_file_basename_reg)))
        
        l2_recs_data = []
        ssim_recs_data = []
        psnr_recs_data = []
        l2_recs_reg = []
        ssim_recs_reg = []
        psnr_recs_reg = []
        for i in range(recons_data.shape[2]):
            l2_recs_data.append(0.5 * np.linalg.norm(true_imgs_data[:,:,i].ravel() - recons_data[:,:,i].ravel())**2)
            ssim_recs_data.append(ssim(true_imgs_data[:,:,i],recons_data[:,:,i]))
            psnr_recs_data.append(psnr(true_imgs_data[:,:,i],recons_data[:,:,i]))
            l2_recs_reg.append(0.5 * np.linalg.norm(true_imgs_reg[:,:,i].ravel() - recons_reg[:,:,i].ravel())**2)
            ssim_recs_reg.append(ssim(true_imgs_reg[:,:,i],recons_reg[:,:,i]))
            psnr_recs_reg.append(psnr(true_imgs_reg[:,:,i],recons_reg[:,:,i]))
        
        axs[0,index].set_title(f'{px}x{py}')
        pcm_data =axs[0,index].imshow(param_data.reshape(px,py))
        axs[0][index].set_xticklabels([])
        axs[0][index].set_xticks([])
        axs[0][index].set_yticks([])
        axs[0][index].set_yticklabels([])
        fig.colorbar(pcm_data,ax=axs[0][index],orientation='horizontal')
        axs[0,index].set_xlabel(f'PSNR={np.mean(psnr_recs_data):.4f}\nSSIM={np.mean(ssim_recs_data):.4f}')
        pcm_reg=axs[1,index].imshow(param_reg.reshape(px,py))
        axs[1][index].set_xticklabels([])
        axs[1][index].set_xticks([])
        axs[1][index].set_yticks([])
        axs[1][index].set_yticklabels([])
        fig.colorbar(pcm_reg,ax=axs[1][index],orientation='horizontal')
        axs[1,index].set_xlabel(f'PSNR={np.mean(psnr_recs_reg):.4f}\nSSIM={np.mean(ssim_recs_reg):.4f}')
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
    if len(setting_files)%2 != 0:
        raise ArgumentError('Cannot compare not even setting files')
    print(f'Generating patch comparison plot with:\n{setting_files}')
    for setting_file in setting_files:
        if not os.path.isfile(setting_file):
            raise RuntimeError('Settings file does not exist: %s' % setting_file)
    outfolder = sys.argv[-1]
    outpath = os.path.join('output_table',outfolder)
    if not os.path.isdir(outpath):
        os.makedirs(outpath, exist_ok=True)
        
    data_sf = [sf for i,sf in enumerate(setting_files) if i%2==0]
    reg_sf = [sf for i,sf in enumerate(setting_files) if i%2!=0]
    
    print(f'data_sf:\n{data_sf}\nreg_sf:\n{reg_sf}')

    generate_patches_comparison(data_sf,reg_sf,outpath)

if __name__ == '__main__':
    main()