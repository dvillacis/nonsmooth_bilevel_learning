
from argparse import ArgumentError
import numpy as np

from os import listdir, pardir
from os.path import isfile, join, isdir

import imageio

from PIL import Image

def get_1D_signal(index,domain=(-2,2),step=0.1):
    t = np.arange(domain[0],domain[1],step)
    u = np.zeros_like(t)
    u[len(u)//2:-index-1] = 1.0
    return u

def get_2D_phantom(index,height=10,width=10):
    u = np.zeros((height,width))
    u[-index-1+height//2:-index-1,-index-1+width//2:-index-1] = 1.0
    return u

def load_training_data(ds_dir):
    true_imgs_dir = join(ds_dir,'true')
    noisy_imgs_dir = join(ds_dir,'noisy')
    if not isdir(true_imgs_dir):
        raise ArgumentError(f'{true_imgs_dir} does not exists...')
    
    true_imgs = []
    true_imgs_path = [f for f in listdir(true_imgs_dir) if isfile(join(true_imgs_dir, f))]
    for img_path in sorted(true_imgs_path):
        if '.png' in img_path:
            img = np.array(Image.open(join(true_imgs_dir,img_path)).convert('L'))
            img = img / np.amax(img)
            true_imgs.append(img)
        
    noisy_imgs = []
    noisy_imgs_path = [f for f in listdir(noisy_imgs_dir) if isfile(join(noisy_imgs_dir, f))]
    for img_path in sorted(noisy_imgs_path):
        if '.png' in img_path:
            img = np.array(Image.open(join(noisy_imgs_dir,img_path)).convert('L'))
            img = img / np.amax(img)
            noisy_imgs.append(img)
    return len(true_imgs), true_imgs, noisy_imgs

def get_kodak_image(img_num, ds_dir, new_height=256, new_width=256):
    infile = join(ds_dir, 'kodim%02d.png' % img_num)
    if isfile(infile):
        img = imageio.imread(infile)
    else:
        raise AttributeError(f'Image not found in {infile}')
    img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]  # to black & white
    height, width = img.shape
    img = img[height // 2 - new_height // 2:height // 2 + new_height // 2,
              width // 2 - new_width // 2:width // 2 + new_width // 2]
    return img / 255  # normalise
    