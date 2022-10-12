import sys, os, json
import numpy as np
import pandas as pd

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def read_json(infile):
    with open(infile, 'r') as ifile:
        mydict = json.load(ifile)
    return mydict

def generate_comparison_table(data_setting_files,reg_setting_files, outpath):
    summary = pd.DataFrame(columns=['dataset','dataparam','dataL2','dataPSNR','dataSSIM','regparam','regL2','regPSNR','regSSIM'])
    for index,setting_file in enumerate(data_setting_files):
        setting = read_json(setting_file)
        data_setting_file_basename = setting_file.split(os.path.sep)[-1].replace('.json', '')
        resultdir_data = os.path.join('../raw_results',data_setting_file_basename)
        if not os.path.isfile(setting_file):
            raise RuntimeError('Cannot find results folder: %s' % data_setting_file_basename)
        data_param = np.load(os.path.join(resultdir_data,'%s_optimal_par.npy' % data_setting_file_basename))
        
        reg_setting_file_basename = reg_setting_files[index].split(os.path.sep)[-1].replace('.json', '')
        resultdir_reg = os.path.join('../raw_results',reg_setting_file_basename)
        if not os.path.isfile(reg_setting_files[index]):
            raise RuntimeError('Cannot find results folder: %s' % reg_setting_file_basename)
        reg_param = np.load(os.path.join(resultdir_reg,'%s_optimal_par.npy' % reg_setting_file_basename))
        
        # Loading Results
        recons_data = np.load(os.path.join(resultdir_data,'%s_recons.npy' % (data_setting_file_basename)))
        true_imgs_data = np.load(os.path.join(resultdir_data,'%s_true_imgs.npy' % (data_setting_file_basename)))
        noisy_imgs_data = np.load(os.path.join(resultdir_data,'%s_noisy_imgs.npy' % (data_setting_file_basename)))
        param_data = np.load(os.path.join(resultdir_data,'%s_optimal_par.npy' % (data_setting_file_basename)))
        
        recons_reg = np.load(os.path.join(resultdir_reg,'%s_recons.npy' % (reg_setting_file_basename)))
        true_imgs_reg = np.load(os.path.join(resultdir_reg,'%s_true_imgs.npy' % (reg_setting_file_basename)))
        noisy_imgs_reg = np.load(os.path.join(resultdir_reg,'%s_noisy_imgs.npy' % (reg_setting_file_basename)))
        param_reg = np.load(os.path.join(resultdir_reg,'%s_optimal_par.npy' % (reg_setting_file_basename)))

        l2_data_recs = []
        psnr_data_recs = []
        ssim_data_recs = []
        l2_reg_recs = []
        psnr_reg_recs = []
        ssim_reg_recs = []
        for i in range(recons_data.shape[2]):
            l2_data_recs.append(0.5 * np.linalg.norm(true_imgs_data[:,:,i].ravel() - recons_data[:,:,i].ravel())**2)
            ssim_data_recs.append(ssim(true_imgs_data[:,:,i],recons_data[:,:,i]))
            psnr_data_recs.append(psnr(true_imgs_data[:,:,i],recons_data[:,:,i]))
            l2_reg_recs.append(0.5 * np.linalg.norm(true_imgs_reg[:,:,i].ravel() - recons_reg[:,:,i].ravel())**2)
            ssim_reg_recs.append(ssim(true_imgs_reg[:,:,i],recons_reg[:,:,i]))
            psnr_reg_recs.append(psnr(true_imgs_reg[:,:,i],recons_reg[:,:,i]))
        
        summary.loc[index+1] = [index+1,param_data[0],np.mean(l2_data_recs),np.mean(psnr_data_recs),np.mean(ssim_data_recs),param_reg[0],np.mean(l2_reg_recs),np.mean(psnr_reg_recs),np.mean(ssim_reg_recs)]
    print(summary)
    with open(os.path.join(outpath,'summary_table.tex'),'w') as f:
        print(summary.style.to_latex(),file=f)

def main():
    sys.path.append(os.path.pardir)  # for upper_level_problem
    if len(sys.argv) < 3:
        print("Usage: python %s settings_files outfolder" % sys.argv[0])
        print("where")
        print("    settings_files = json file containing run details")
        print("    outfolder = folder to save plots to")
        exit()
    
    # List setting files
    num_setting_files = len(sys.argv)-2
    setting_files = [sys.argv[i] for i in range(1,1+num_setting_files)]
    print(f'Generating summary table with:\n{setting_files}')
    for setting_file in setting_files:
        if not os.path.isfile(setting_file):
            raise RuntimeError('Settings file does not exist: %s' % setting_file)
        
    data_setting_files = [s for i,s in enumerate(setting_files) if i%2 == 0 ]
    reg_setting_files = [s for i,s in enumerate(setting_files) if i%2 != 0]
    print(data_setting_files)
    print(reg_setting_files)
    outfolder = sys.argv[-1]
    outpath = os.path.join('output_table',outfolder)
    if not os.path.isdir(outpath):
        os.makedirs(outpath, exist_ok=True)

    generate_comparison_table(data_setting_files,reg_setting_files,outpath)

if __name__ == '__main__':
    main()