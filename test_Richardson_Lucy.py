# test on different dataset
import os 
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
from natsort import natsorted
from scipy.signal import convolve2d as conv2
from skimage import color, data, restoration

from util.utils import *
from util.metrics import *
from util.data import *

def test():

    test_dir = './data/test/'
    test_list = natsorted(os.listdir(test_dir))  
    
    test_raw = np.load(test_dir + test_list[0])  # select testset [imn, bio, storm, w_c]
    w_test_img, o_test_img, psf_test_img = test_raw['w'], test_raw['gt'], test_raw['psf']
    
    deconv_RL_test = []
    for i in range(w_test_img.shape[0]):
        RL_temp = restoration.richardson_lucy(w_test_img[i].squeeze(), psf_test_img[i].squeeze(), num_iter=30)
        deconv_RL_test.append(RL_temp)

    deconv_RL_test = np.expand_dims(rescale(np.asarray(deconv_RL_test)), axis=-1)
    # print(deconv_RL_test.shape, w_test_img.shape)
    
    # save a rnadom fig under path
    # NUM = random.randint(0, pred_test_list[0].shape[0]-1)
    NUM = 24
    fig_path = './results/RL/'
    save_svg(np.expand_dims(deconv_RL_test[NUM],axis=0), 'RL', 'pre', fig_path)
    save_svg(np.expand_dims(o_test_img[NUM],axis=0), 'RL', 'o', fig_path)
    save_svg(np.expand_dims(w_test_img[NUM],axis=0), 'RL', 'w', fig_path)
    print('test saved at:', fig_path)
    
    psnr_value, ssim_value, rmse = metrics(deconv_RL_test, o_test_img)
    print('test performance:', psnr_value.round(2), ssim_value.round(2), rmse.round(2))
    
test()