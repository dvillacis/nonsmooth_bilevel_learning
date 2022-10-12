import sys, os, json
import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def read_json(infile):
    with open(infile, 'r') as ifile:
        mydict = json.load(ifile)
    return mydict

def denoise(noisy_imgs,param,px,py):
    from problems.solvers.rof.solver import ROFSolver_2D
    from problems.operators.Gradient import Gradient
    from problems.operators.Patch import Patch
    nx,ny = noisy_imgs[0].shape
    recons = np.zeros(shape=(nx,ny,len(noisy_imgs)))
    K = Gradient(dims=((nx,ny)))
    for i,noisy_img in enumerate(noisy_imgs):
        solver = ROFSolver_2D(noisy_img,K)
        reg_par = Patch(param,px,py)
        data_par = Patch(np.ones(px*py),px,py)
        recons[:,:,i] = solver.solve(data_par=data_par,reg_par=reg_par)
    return recons

def generate_validation_table(setting_files, validation_dataset, outpath):
    from problems.data_generation import load_training_data
    for index,setting_file in enumerate(setting_files):
        setting = read_json(setting_file)
        px = int(setting["problem"]["px"])
        py = int(setting["problem"]["px"])
        setting_file_basename = setting_file.split(os.path.sep)[-1].replace('.json', '')
        resultdir = os.path.join('../raw_results',setting_file_basename)
        if not os.path.isfile(setting_file):
            raise RuntimeError('Cannot find results folder: %s' % setting_file_basename)
        param = np.load(os.path.join(resultdir,'%s_optimal_par.npy' % setting_file_basename))
        num_val_imgs, true_imgs, noisy_imgs = load_training_data(validation_dataset)
        recons = denoise(noisy_imgs,param,px,py)
        l2_recs = []
        psnr_recs = []
        ssim_recs = []
        for i in range(recons.shape[2]):
            l2_recs.append(0.5 * np.linalg.norm(true_imgs[i].ravel() - recons[:,:,i].ravel())**2)
            ssim_recs.append(ssim(true_imgs[i],recons[:,:,i]))
            psnr_recs.append(psnr(true_imgs[i],recons[:,:,i]))
        print(f'{np.mean(l2_recs)} - {np.mean(psnr_recs)} - {np.mean(ssim_recs)}')

def main():
    sys.path.append(os.path.pardir)  # for upper_level_problem
    if len(sys.argv) < 4:
        print("Usage: python %s settings_files validation_dataset outfolder" % sys.argv[0])
        print("where")
        print("    settings_files = json file containing run details")
        print("    outfolder = folder to save plots to")
        exit()
    
    # List setting files
    num_setting_files = len(sys.argv)-3
    setting_files = [sys.argv[i] for i in range(1,1+num_setting_files)]
    print(f'Generating summary table with:\n{setting_files}')
    for setting_file in setting_files:
        if not os.path.isfile(setting_file):
            raise RuntimeError('Settings file does not exist: %s' % setting_file)
    outfolder = sys.argv[-1]
    validation_dataset = sys.argv[-2]
    outpath = os.path.join('output_table',outfolder)
    if not os.path.isdir(outpath):
        os.makedirs(outpath, exist_ok=True)

    generate_validation_table(setting_files,validation_dataset,outpath)

if __name__ == '__main__':
    main()