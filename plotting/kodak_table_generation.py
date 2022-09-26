import sys,os
import pandas as pd
import numpy as np
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def read_json(infile):
    with open(infile, 'r') as ifile:
        mydict = json.load(ifile)
    return mydict

def generate_patch_table(nstr_settings_file,outfolder):
    summary_table = pd.DataFrame(columns=['img_num','l2_nstr','l2_dynamic','l2_fixed','psnr_nstr','psnr_dynamic','psnr_fixed','ssim_nstr','ssim_dynamic','ssim_fixed'])
    # NSTR experiment
    setting_nstr = read_json(nstr_settings_file)
    setting_file_basename_nstr = nstr_settings_file.split(os.path.sep)[-1].replace('.json', '')
    resultdir_nstr = os.path.join('raw_results',setting_file_basename_nstr)
    # if not os.path.isfile(resultdir_nstr):
    #     raise RuntimeError('Cannot find results folder: %s' % resultdir_nstr)
    recons_nstr = np.load(os.path.join(resultdir_nstr,'%s_recons.npy' % (setting_file_basename_nstr)))
    true_imgs = np.load(os.path.join(resultdir_nstr,'%s_true_imgs.npy' % (setting_file_basename_nstr)))

    # DYNAMIC experiment
    setting_file_basename_dynamic = '3param2d_demo'
    resultdir_dynamic = os.path.join('raw_results',setting_file_basename_dynamic)
    recons_dynamic = np.load(os.path.join(resultdir_dynamic,'%s_fista_dynamic_recons.npy' % (setting_file_basename_dynamic)))

    # FIXED experiment
    setting_file_basename_fixed = '3param2d_demo'
    resultdir_fixed = os.path.join('raw_results',setting_file_basename_fixed)
    recons_fixed = np.load(os.path.join(resultdir_fixed,'%s_fista_fixed_niters_1000_recons.npy' % (setting_file_basename_fixed)))

    for index in range(true_imgs.shape[2]):
        print(index,true_imgs[:,:,index].shape)
        l2_nstr = 0.5 * np.linalg.norm(true_imgs[:,:,index].ravel() - recons_nstr[:,:,index].ravel())**2
        psnr_nstr = psnr(true_imgs[:,:,index],recons_nstr[:,:,index])
        ssim_nstr = ssim(true_imgs[:,:,index],recons_nstr[:,:,index])
        l2_dynamic = 0.5 * np.linalg.norm(true_imgs[:,:,index].ravel() - recons_dynamic[:,:,index].ravel())**2
        psnr_dynamic = psnr(true_imgs[:,:,index],recons_dynamic[:,:,index])
        ssim_dynamic = ssim(true_imgs[:,:,index],recons_dynamic[:,:,index])
        l2_fixed = 0.5 * np.linalg.norm(true_imgs[:,:,index].ravel() - recons_fixed[:,:,index].ravel())**2
        psnr_fixed = psnr(true_imgs[:,:,index],recons_fixed[:,:,index])
        ssim_fixed = ssim(true_imgs[:,:,index],recons_fixed[:,:,index])
        summary_table.loc[index+1] = [index+1,l2_nstr,l2_dynamic,l2_fixed,psnr_nstr,psnr_dynamic,psnr_fixed,ssim_nstr,ssim_dynamic,ssim_fixed]
    summary_table.loc['mean'] = summary_table.mean()
    print(summary_table)
    with open(os.path.join(outfolder,'summary_table.tex'),'w') as f:
        print(summary_table.style.to_latex(),file=f)

def main():
    if len(sys.argv) < 3:
        print("Usage: python %s nstr_settings_file outfolder" % sys.argv[0])
        print("where")
        print("    settings_files = json file containing run details")
        print("    outfolder = folder to save plots to")
        exit()
    
    # List setting files
    nstr_settings_file = sys.argv[1]
    print(f'Generating summary table with:\n{nstr_settings_file}')
    if not os.path.isfile(nstr_settings_file):
        raise RuntimeError('Settings file does not exist: %s' % nstr_settings_file)
    outfolder = sys.argv[-1]
    outpath = os.path.join('output_table',outfolder)
    if not os.path.isdir(outpath):
        os.makedirs(outpath, exist_ok=True)

    generate_patch_table(nstr_settings_file,outpath)

if __name__ == '__main__':
    main()