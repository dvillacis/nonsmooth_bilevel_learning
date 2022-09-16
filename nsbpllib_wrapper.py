#!/usr/bin/env python

"""
Main script to run tests
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import json
import os, sys

from problems.solvers.nstrbox.solver import solve as nstrbox_solve
from problems.upper_level_problems import UpperScalarDataLearning_2D, UpperPatchDataLearning_2D, UpperPatchRegLearning_2D, UpperScalarDataLearning, UpperPatchDataLearning

def read_json(infile):
    with open(infile, 'r') as ifile:
        mydict = json.load(ifile)
    return mydict

def _to_array(x,lbl):
    try:
        if isinstance(x,float):
            x = [x]
        return np.asarray_chkfinite(x)
    except ValueError:
        raise ValueError('%s contains Nan/Inf values' % lbl)
    
def load_start_parameter(par,px=1,py=1):
    if '.npy' in str(par):
        x0 = np.load(par)
        p = int(np.sqrt(len(x0)))
        m = px // p
        x0 = x0.reshape((p,p))
        x0 = np.kron(x0,np.ones((m,m)))
        print(f'x0:{x0}')
    else:
        x0 = float(par)
        x0 = x0 * np.ones(px*py)
    return x0.ravel()
    
def build_settings(settings_dict):
    problem_type = str(settings_dict['problem']['type'])
    if problem_type == '2D_scalar_data_learning':
        return build_settings_2d_scalar_data_learning(settings_dict)
    elif problem_type == '2D_patch_data_learning':
        return build_settings_2d_patch_data_learning(settings_dict)
    elif problem_type == '2D_patch_reg_learning':
        return build_settings_2d_patch_reg_learning(settings_dict)
    elif problem_type == 'ds_scalar_data_learning':
        return build_settings_ds_scalar_data_learning(settings_dict)
    elif problem_type == 'ds_patch_data_learning':
        return build_settings_ds_patch_data_learning(settings_dict)
    else:
        raise RuntimeError('Unknown problem type: %s' % problem_type)
    
def build_settings_2d_scalar_data_learning(settings_dict):
    num_training_data = int(settings_dict['problem']['num_training_data'])
    npixels = int(settings_dict['problem']['npixels'])
    noise_level = float(settings_dict['problem']['noise_level'])
    verbose = bool(settings_dict['problem']['verbose'])
    upper_level_problem = UpperScalarDataLearning_2D(
        num_training_data,
        int(settings_dict['seed']),
        noise_level,
        npixels,
        verbose)
    x0 = _to_array(float(settings_dict['problem']['start_parameter']),'x0')
    return upper_level_problem,x0

def build_settings_2d_patch_data_learning(settings_dict):
    num_training_data = int(settings_dict['problem']['num_training_data'])
    npixels = int(settings_dict['problem']['npixels'])
    noise_level = float(settings_dict['problem']['noise_level'])
    verbose = bool(settings_dict['problem']['verbose'])
    px = int(settings_dict['problem']['px'])
    py = int(settings_dict['problem']['py'])
    upper_level_problem = UpperPatchDataLearning_2D(
        num_training_data,
        px,
        py,
        int(settings_dict['seed']),
        noise_level,
        npixels,
        verbose)
    x0 = np.load(settings_dict['problem']['start_parameter'])
    x0 = x0*np.ones(px*py)
    # x0 = _to_array(float(settings_dict['problem']['start_parameter']),'x0')
    return upper_level_problem,x0

def build_settings_ds_scalar_data_learning(settings_dict):
    ds_dir = settings_dict['problem']['dataset_dir']
    # noise_level = float(settings_dict['problem']['noise_level'])
    verbose = bool(settings_dict['problem']['verbose'])
    px = int(settings_dict['problem']['px'])
    py = int(settings_dict['problem']['py'])
    upper_level_problem = UpperScalarDataLearning(
        ds_dir,
        seed=int(settings_dict['seed']),
        # noise_level=noise_level,
        verbose=verbose)
    x0 = load_start_parameter(settings_dict['problem']['start_parameter'])
    # x0 = _to_array(float(settings_dict['problem']['start_parameter']),'x0')
    return upper_level_problem,x0

def build_settings_ds_patch_data_learning(settings_dict):
    ds_dir = settings_dict['problem']['dataset_dir']
    # noise_level = float(settings_dict['problem']['noise_level'])
    verbose = bool(settings_dict['problem']['verbose'])
    px = int(settings_dict['problem']['px'])
    py = int(settings_dict['problem']['py'])
    upper_level_problem = UpperPatchDataLearning(
        ds_dir,
        seed=int(settings_dict['seed']),
        # noise_level=noise_level,
        px=px,
        py=py,
        verbose=verbose)
    x0 = load_start_parameter(settings_dict['problem']['start_parameter'],px,py)
    # x0 = _to_array(float(settings_dict['problem']['start_parameter']),'x0')
    return upper_level_problem,x0

def build_settings_2d_patch_reg_learning(settings_dict):
    num_training_data = int(settings_dict['problem']['num_training_data'])
    npixels = int(settings_dict['problem']['npixels'])
    noise_level = float(settings_dict['problem']['noise_level'])
    verbose = bool(settings_dict['problem']['verbose'])
    px = int(settings_dict['problem']['px'])
    py = int(settings_dict['problem']['py'])
    upper_level_problem = UpperPatchRegLearning_2D(
        num_training_data,
        px,
        py,
        int(settings_dict['seed']),
        noise_level,
        npixels,
        verbose)
    if '.npy' in str(settings_dict['problem']['start_parameter']):
        x0 = np.load(settings_dict['problem']['start_parameter'])
    else:
        x0 = float(settings_dict['problem']['start_parameter'])
        x0 = x0 * np.ones(px*py)
    x0 = x0*np.ones(px*py)
    # x0 = _to_array(float(settings_dict['problem']['start_parameter']),'x0')
    return upper_level_problem,x0

def run_nsbpl(settings_dict, outfolder, run_name):
    upl,x0 = build_settings(settings_dict)
    evals,sol = nstrbox_solve(upl,x0)
    true_imgs, noisy_imgs, recons = upl.get_training_data()
    extra_data = {'true_imgs':true_imgs,'noisy_imgs':noisy_imgs,'recons':recons}
    return evals,sol,extra_data

def test_gradient():
    up_prob = UpperPatchRegLearning_2D(1,1,1)
    for i in np.arange(1e-12,1.0,1e-2):
        i = _to_array(i,'i')
        f,g=up_prob(i,smooth=False)
        print(i,f,g) 
    up_evals = up_prob.get_evals()
    up_evals.plot(x='eval',y=['f','g'])
    plt.show()
    print(up_evals)  

def save_nsbpl_results(settings_dict,evals,sol,extra_data,outfolder,run_name):
    # Exporting evals
    evals_outfile = os.path.join(outfolder,'%s_evals.pkl' % (run_name))
    evals.to_pickle(evals_outfile)
    logging.info(f'Saved evals to: {evals_outfile}')
    
    if 'true_imgs' in extra_data:
        true_img_outfile = os.path.join(outfolder, '%s_true_imgs.npy' % (run_name))
        np.save(true_img_outfile,extra_data['true_imgs'])
        logging.info("Saved training data (true images) to: %s" % true_img_outfile)
    
    if 'noisy_imgs' in extra_data:
        true_img_outfile = os.path.join(outfolder, '%s_noisy_imgs.npy' % (run_name))
        np.save(true_img_outfile,extra_data['noisy_imgs'])
        logging.info("Saved training data (noisy images) to: %s" % true_img_outfile)
    
    if 'recons' in extra_data:
        true_img_outfile = os.path.join(outfolder, '%s_recons.npy' % (run_name))
        np.save(true_img_outfile,extra_data['recons'])
        logging.info("Saved training data (final reconstruction) to: %s" % true_img_outfile)
        
    # Write final statistics
    stats_outfile = os.path.join(outfolder, '%s_stats.txt' % (run_name))
    with open(stats_outfile,'w') as f:
        print(sol,file=f)
    logging.info("Saved experiment statistics to: %s" % stats_outfile)
        
    # Write the parameter
    opt_parameter_outfile = os.path.join(outfolder, '%s_optimal_par.npy' % (run_name))
    np.save(opt_parameter_outfile,sol.x)
    logging.info("Saved optimal parameter to: %s" % opt_parameter_outfile)
    
    

def main():
    if len(sys.argv) != 3:
        print("Usage: python %s settings_file outfolder" % sys.argv[0])
        print("where")
        print("    settings_file = json file containing run details")
        print("    outfolder = folder to save results to")
        exit()
        
    # Specific settings
    settings_file = sys.argv[1]
    if not os.path.isfile(settings_file):
        raise RuntimeError('Settings file does not exist: %s' % settings_file)
    settings_file_basename = settings_file.split(os.path.sep)[-1].replace('.json', '')
    outfolder = os.path.join(sys.argv[2], settings_file_basename)
    settings_dict = read_json(settings_file)
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder, exist_ok=True)
    logfile = os.path.join(outfolder, '%s_log.txt' % (settings_file_basename))
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=logfile, filemode='w')
    
    runname = settings_file_basename + '_' + settings_dict['name']
    
    # test_gradient()
    # run nsbpl
    evals, sol, extra_data = run_nsbpl(settings_dict,outfolder,settings_file_basename)
    print(sol)
    save_nsbpl_results(settings_dict,evals,sol,extra_data,outfolder,settings_file_basename)


if __name__ == '__main__':
    main()
