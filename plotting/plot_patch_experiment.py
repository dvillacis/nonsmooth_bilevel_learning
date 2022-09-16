import sys,os,json
import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def read_json(infile):
    with open(infile, 'r') as ifile:
        mydict = json.load(ifile)
    return mydict

def plot_recons(param,true_imgs,noisy_imgs,recons,px,py):
    n_samples = true_imgs.shape[2]
    if n_samples > 1:
        nfigs = 4
        fig, axs = plt.subplots(n_samples, nfigs, figsize=(14, 8))
        for i in range(n_samples):
            axs[i][1].imshow(true_imgs[:,:,i], cmap='gray', vmin=0, vmax=1)
            axs[i][1].set_title('Original')
            axs[i][1].axis('off')
            axs[i][1].axis('tight')
            axs[i][2].imshow(noisy_imgs[:,:,i], cmap='gray', vmin=0, vmax=1)
            axs[i][2].set_title('Noisy')
            axs[i][2].axis('off')
            axs[i][2].axis('tight')
            axs[i][3].imshow(recons[:,:,i], cmap='gray', vmin=0, vmax=1)
            axs[i][3].set_title('Optimal TViso Reconstruction')
            axs[i][3].axis('off')
            axs[i][3].axis('tight')
            
        if len(param) > 1:
            pcm = axs[0][0].imshow(param.reshape((px,py)), cmap='gray')
            axs[0][0].set_title('Optimal Parameter')
            axs[0][0].axis('off')
            axs[0][0].axis('tight')
            fig.colorbar(pcm,ax=axs[0][0],orientation='horizontal')
    else:
        nfigs = 3
        if len(param) > 1:
            nfigs = 4
        fig, axs = plt.subplots(n_samples, nfigs, figsize=(12, 4))
        axs[0].imshow(true_imgs[:,:,0], cmap='gray', vmin=0, vmax=1)
        axs[0].set_title('Original')
        axs[0].axis('off')
        axs[0].axis('tight')
        axs[1].imshow(noisy_imgs[:,:,0], cmap='gray', vmin=0, vmax=1)
        axs[1].set_title('Noisy')
        axs[1].axis('off')
        axs[1].axis('tight')
        axs[2].imshow(recons[:,:,0], cmap='gray', vmin=0, vmax=1)
        axs[2].set_title('Optimal TViso')
        axs[2].axis('off')
        axs[2].axis('tight')
        
        if len(param) > 1:
            pcm = axs[3].imshow(param, cmap='gray')
            axs[3].set_title('Optimal Parameter')
            axs[3].axis('off')
            axs[3].axis('tight')
            fig.colorbar(pcm,ax=axs[3],orientation='horizontal')
        else:
            axs[0].set_xlabel(f'parameter: {param}')
            
    plt.tight_layout()
    # plt.show()
    return fig
    
def generate_recons_quality_stats(true_imgs,noisy_imgs,recons):
    n_samples = true_imgs.shape[2]
    l2_costs = []
    psnrs = []
    ssims = []
    for i in range(n_samples):
        l2_costs.append(np.linalg.norm(true_imgs[:,:,i].ravel() - recons[:,:,i].ravel())**2)
        ssims.append(ssim(true_imgs[:,:,i],recons[:,:,i]))
        psnrs.append(psnr(true_imgs[:,:,i],recons[:,:,i]))
    q_df = pd.DataFrame.from_dict({'l2':l2_costs,'psnr':psnrs,'ssim':ssims})
    q_df.loc['mean'] = q_df.mean()
    
    l2_noisy = []
    psnrs_noisy = []
    ssims_noisy = []
    for i in range(n_samples):
        l2_noisy.append(np.linalg.norm(true_imgs[:,:,i].ravel() - noisy_imgs[:,:,i].ravel())**2)
        ssims_noisy.append(ssim(true_imgs[:,:,i],noisy_imgs[:,:,i]))
        psnrs_noisy.append(psnr(true_imgs[:,:,i],noisy_imgs[:,:,i]))
    q_df_noisy = pd.DataFrame.from_dict({'l2':l2_noisy,'psnr':psnrs_noisy,'ssim':ssims_noisy})
    q_df_noisy.loc['mean'] = q_df_noisy.mean()
    return q_df,q_df_noisy

def main():
    if len(sys.argv) != 3:
        print("Usage: python %s settings_file outfolder" % sys.argv[0])
        print("where")
        print("    settings_file = json file containing run details")
        print("    outfolder = folder to save plots to")
        exit()
    
    # Specific settings
    settings_file = sys.argv[1]
    if not os.path.isfile(settings_file):
        raise RuntimeError('Settings file does not exist: %s' % settings_file)
    settings_file_basename = settings_file.split(os.path.sep)[-1].replace('.json', '')
    results_folder = os.path.join('raw_results',settings_file_basename)
    if not os.path.isdir(results_folder):
        raise RuntimeError('Results folder does not exist: %s' % results_folder)
    outfolder = os.path.join(sys.argv[2], settings_file_basename)
    settings_dict = read_json(settings_file)
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder, exist_ok=True)
        
    # Load patch size
    px = int(settings_dict['problem']['px'])
    py = int(settings_dict['problem']['py'])
        
    # Load optimal parameter
    opt_param = np.load(os.path.join(results_folder,'%s_optimal_par.npy' % (settings_file_basename)))
    
    # Load images
    q_recons_fig_outfile = os.path.join(outfolder,'recons_fig.pgf')
    true_imgs = np.load(os.path.join(results_folder,'%s_true_imgs.npy' % (settings_file_basename)))
    noisy_imgs = np.load(os.path.join(results_folder,'%s_noisy_imgs.npy' % (settings_file_basename)))
    recons = np.load(os.path.join(results_folder,'%s_recons.npy' % (settings_file_basename)))
    fig = plot_recons(opt_param,true_imgs,noisy_imgs,recons,px,py)
    plt.savefig(q_recons_fig_outfile)
    
    # Generate quality tables
    q_recons_outfile = os.path.join(outfolder,'q_recons.tex')
    q_noisy_outfile = os.path.join(outfolder,'q_noisy.tex')
    q_stats = generate_recons_quality_stats(true_imgs,noisy_imgs,recons)
    with open(q_recons_outfile,'w') as f:
        print(q_stats[0].style.to_latex(),file=f)
    with open(q_noisy_outfile,'w') as f:
        print(q_stats[1].style.to_latex(),file=f)
    
    
        
if __name__ == '__main__':
    main()