# test on different dataset
import os 
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import random
from natsort import natsorted
from keras.models import load_model

from util.utils import *
from models.s_rBCR import *
from util.loss_func import *
from util.metrics import *
from util.data import *

def test(path = './weights/s-rBCR/'):
    eval_model = model_s_rBCR()
    eval_model.compile(optimizer='adam', loss=loss_function_mimo, metrics=[metrics_func_mimo]) 

    eval_model = load_model(path+'single_bcr_best.h5', compile=False)
    
    test_dir = './data/test/'
    test_list = natsorted(os.listdir(test_dir))  
    
    test_raw = np.load(test_dir + test_list[0])  # select testset [imn, bio, storm, w_c]
    w_test_img, o_test_img = test_raw['w'], test_raw['gt']
    
    pred_test_img = eval_model.predict(w_test_img)
    
    # save a rnadom fig under path
    # NUM = random.randint(0, pred_test_list[0].shape[0]-1)
    NUM = 24
    fig_path = './results/s_rBCR/'
    save_svg(np.expand_dims(pred_test_img[NUM],axis=0), 'single_rBCR', 'pre', fig_path)
    save_svg(np.expand_dims(o_test_img[NUM],axis=0), 'single_rBCR', 'o', fig_path)
    save_svg(np.expand_dims(w_test_img[NUM],axis=0), 'single_rBCR', 'w', fig_path)
    print('test saved at:', fig_path)
    
    psnr_value, ssim_value, rmse = metrics(pred_test_img, o_test_img)
    print('test performance:', psnr_value.round(2), ssim_value.round(2), rmse.round(2))
    
test()