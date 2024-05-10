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
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from util.utils import *
from models.Unet import *
from util.loss_func import *
from util.metrics import *
from util.data import *

def test(path = './weights/Unet/'):
    
    eval_model = load_model(path+'Unet_best.hdf5', compile=False, custom_objects={'InstanceNormalization':InstanceNormalization})
    
    test_dir = './data/test/'
    test_list = natsorted(os.listdir(test_dir))  
    
    test_raw = np.load(test_dir + test_list[0])  # select testset [imn, bio, storm, w_c]
    w_test_img, o_test_img = test_raw['w'], test_raw['gt']
    
    pred_test_img = eval_model.predict(w_test_img)
    
    # save a rnadom fig under path
    # NUM = random.randint(0, pred_test_list[0].shape[0]-1)
    NUM = 24
    fig_path = './results/Unet/'
    save_svg(np.expand_dims(pred_test_img[NUM],axis=0), 'Unet', 'pre', fig_path)
    save_svg(np.expand_dims(o_test_img[NUM],axis=0), 'Unet', 'o', fig_path)
    save_svg(np.expand_dims(w_test_img[NUM],axis=0), 'Unet', 'w', fig_path)
    print('test saved at:', fig_path)
    
    psnr_value, ssim_value, rmse = metrics(rescale(pred_test_img), o_test_img)
    print('test performance:', psnr_value.round(2), ssim_value.round(2), rmse.round(2))
    
test()