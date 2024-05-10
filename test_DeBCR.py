# test on different dataset
import os 
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import random
from natsort import natsorted

from util.utils import *
from models.m_rBCR import *
from util.loss_func import *
from util.metrics import *
from util.data import *

def test(path = './weights/2D_denoising/'):
    eval_model = model_m_rBCR()
    eval_model.compile(optimizer='adam', loss=loss_function_mimo, metrics=[metrics_func_mimo]) 

    print(eval_model.input_shape, eval_model.output_shape) # [(None, 128, 128, 1), (None, 64, 64, 1), (None, 32, 32, 1)] 
    print(eval_model.summary())

    # reload the check point
    checkpoint = tf.train.Checkpoint(model=eval_model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, path, max_to_keep=5)

    # Specify the checkpoint you want to restore for testing
    checkpoint_to_restore = path + '/ckpt-best'
    # Restore the model's weights
    status = checkpoint.restore(checkpoint_to_restore)
    status.assert_consumed()
    
    test_dir = './data/test/2D_denoising/'
    test_list = natsorted(os.listdir(test_dir))  
    
    test_raw = np.load(test_dir + test_list[0])

    w_test_img, o_test_img = test_raw['low'], test_raw['gt']
    w_test_img, o_test_img = np.expand_dims(w_test_img, axis=3), np.expand_dims(o_test_img, axis=3)
    w_test_img, o_test_img = rescale(w_test_img), rescale(o_test_img)

    test_w_list, test_o_list = multi_input(w_test_img, o_test_img)
    
    pred_test_list = eval_model.predict(test_w_list)
    
    # save a rnadom fig under path
    NUM = random.randint(0, pred_test_list[0].shape[0]-1)
    fig_path = './results/DeBCR/'
    pred_list_level= pred_test_list[0][NUM], pred_test_list[1][NUM], pred_test_list[2][NUM]
    test_w_list_level = test_w_list[0][NUM], test_w_list[1][NUM], test_w_list[2][NUM]
    save_3_levels(pred_list_level, fig_path, '2d_denoise', 'pre')
    save_3_levels(test_w_list_level, fig_path, '2d_denoise', 'w')
    print('test saved at:', fig_path)
    
    psnr_value, ssim_value, rmse = metrics(pred_test_list[0], test_o_list[0])
    print('test performance:', psnr_value.round(2), ssim_value.round(2), rmse.round(2))
    
test()