import sys,os,json
import pandas as pd
import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def read_json(infile):
    with open(infile, 'r') as ifile:
        mydict = json.load(ifile)
    return mydict

def read_stats_file(stats_path):
    keywords = ['nfev','nit','njev','n_reg_jev']
    if not os.path.isfile(stats_path):
            raise RuntimeError('Cannot find stats: %s' % stats_path)
    with open(stats_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'n_reg_jev' in line:
                n_reg_jev = int(line.split(':')[1].strip())
            elif 'nfev' in line:
                n_fev = int(line.split(':')[1].strip())
            elif 'nit' in line:
                nit = int(line.split(':')[1].strip())
            elif 'njev' in line:
                njev = int(line.split(':')[1].strip())
            else:
                print('Skipping line...')
        return n_reg_jev,n_fev,nit,njev

def generate_param_init_table(setting_files,outpath):
    summary_table = pd.DataFrame(columns=['initpar','nit','nfev','ngev','nreggev','COST','PSNR','SSIM'])
    for index,setting_file in enumerate(setting_files):
        setting = read_json(setting_file)
        setting_file_basename = setting_file.split(os.path.sep)[-1].replace('.json', '')
        resultdir = os.path.join('../problem_settings/cameraman_alpha_data_init_experiment/alpha_init_experiment',setting_file_basename)
        if not os.path.isfile(setting_file):
            raise RuntimeError('Cannot find results folder: %s' % setting_file_basename)
        recons = np.load(os.path.join(resultdir,'%s_recons.npy' % (setting_file_basename)))
        true_imgs = np.load(os.path.join(resultdir,'%s_true_imgs.npy' % (setting_file_basename)))
        n_reg_jev,n_fev,nit,njev = read_stats_file(os.path.join(resultdir,'%s_stats.txt' % (setting_file_basename))) 
        l2_recs = []
        ssim_recs = []
        psnr_recs = []
        for i in range(recons.shape[2]):
            l2_recs.append(0.5 * np.linalg.norm(true_imgs[:,:,i].ravel() - recons[:,:,i].ravel())**2)
            ssim_recs.append(ssim(true_imgs[:,:,i],recons[:,:,i]))
            psnr_recs.append(psnr(true_imgs[:,:,i],recons[:,:,i]))
        # print(f'{setting_file_basename}:\nl2={np.mean(l2_recs)} psnr={np.mean(psnr_recs)} ssim={np.mean(ssim_recs)}')
        summary_table.loc[index] = [f'{setting["problem"]["start_parameter"]}',nit,n_fev,njev,n_reg_jev,np.mean(l2_recs),np.mean(psnr_recs),np.mean(ssim_recs)]
    print(summary_table)
    with open(os.path.join(outpath,'summary_table.tex'),'w') as f:
        print(summary_table.style.to_latex(),file=f)

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
    print(f'Generating summary table with:\n{setting_files}')
    for setting_file in setting_files:
        if not os.path.isfile(setting_file):
            raise RuntimeError('Settings file does not exist: %s' % setting_file)
    outfolder = sys.argv[-1]
    outpath = os.path.join('output_table',outfolder)
    if not os.path.isdir(outpath):
        os.makedirs(outpath, exist_ok=True)

    generate_param_init_table(setting_files,outpath)

if __name__ == '__main__':
    main()