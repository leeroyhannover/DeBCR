# test on different dataset
import os 
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import random
from natsort import natsorted
from models.csbdeep.utils import axes_dict, plot_some, plot_history
from models.csbdeep.utils.tf import limit_gpu_memory
from models.csbdeep.io import load_training_data
from models.csbdeep.models import Config, CARE

from util.utils import *
from models.Unet import *
from util.loss_func import *
from util.metrics import *
from util.data import *

def test():
    
    axes = 'SYXC'
    n_channel_in, n_channel_out = 1, 1
    config = Config(axes, n_channel_in, n_channel_out, unet_kern_size=3, train_batch_size=8, train_steps_per_epoch=40)
    
    eval_model = CARE(config=None, name='my_model', basedir='models')
    
    test_dir = './data/test/'
    test_list = natsorted(os.listdir(test_dir))  
    
    test_raw = np.load(test_dir + test_list[0])  # select testset [imn, bio, storm, w_c]
    w_test_img, o_test_img = test_raw['w'], test_raw['gt']
    
    pred_test_img = model.keras_model.predict(w_test_img)
    
    # save a rnadom fig under path
    NUM = random.randint(0, pred_test_list[0].shape[0]-1)
    fig_path = './results/CARE/'
    save_svg(np.expand_dims(pred_test_img[NUM],axis=0), 'CARE', 'pre', fig_path)
    save_svg(np.expand_dims(o_test_img[NUM],axis=0), 'CARE', 'o', fig_path)
    save_svg(np.expand_dims(w_test_img[NUM],axis=0), 'CARE', 'w', fig_path)
    print('test saved at:', fig_path)
    
    psnr_value, ssim_value, rmse = metrics(rescale(pred_test_img), o_test_img)
    print('test performance:', psnr_value.round(2), ssim_value.round(2), rmse.round(2))
    
test()