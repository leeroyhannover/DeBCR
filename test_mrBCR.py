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

def multi_input(w_img, o_img):
    
    w_0, o_0 = w_img, o_img
    w_2, o_2 = w_0[:, ::2, ::2, :], o_0[:, ::2, ::2, :]
    w_4, o_4 = w_0[:, ::4, ::4, :], o_0[:, ::4, ::4, :]
    
    return [w_0, w_2, w_4], [o_0, o_2, o_4]

def save_3_levels(image_list, output_path, domain, kind):
    
    for i in range(len(image_list)):
        image = image_list[i]
        plt.figure(figsize=(1.28, 1.28), dpi=300)  # Set figsize and dpi to match the 128x128 size
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        svg_filename = os.path.join(output_path, f"{str(kind)}_{i:03d}_{str(domain)}.svg")
        plt.savefig(svg_filename, format='svg', bbox_inches='tight', pad_inches=0)
    print('finish save')


def test(path = './weights/m-rBCR/'):
    eval_model = model_m_rBCR()
    eval_model.compile(optimizer='adam', loss=loss_function_mimo, metrics=[metrics_func_mimo]) 

    print(eval_model.input_shape, eval_model.output_shape) # [(None, 128, 128, 1), (None, 64, 64, 1), (None, 32, 32, 1)] 
    print(eval_model.summary())

    # reload the check point
    checkpoint = tf.train.Checkpoint(model=eval_model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, path, max_to_keep=5)

    # Specify the checkpoint you want to restore for testing
    checkpoint_to_restore = path + '/ckpt-24'
    # Restore the model's weights
    status = checkpoint.restore(checkpoint_to_restore)
    status.assert_consumed()
    
    test_dir = './data/test/'
    test_list = natsorted(os.listdir(test_dir))  
    
    test_raw = np.load(test_dir + test_list[0])  # select testset [imn, bio, storm, w_c]
    w_test_img, o_test_img = test_raw['w'], test_raw['gt']
    test_w_list, test_o_list = multi_input(w_test_img, o_test_img)
    
    pred_test_list = eval_model.predict(test_w_list)
    
    # save a rnadom fig under path
    NUM = random.randint(0, pred_test_list[0].shape[0]-1)
    fig_path = './results/m_rBCR/'
    pred_list_level= pred_test_list[0][NUM], pred_test_list[1][NUM], pred_test_list[2][NUM]
    test_w_list_level = test_w_list[0][NUM], test_w_list[1][NUM], test_w_list[2][NUM]
    save_3_levels(pred_list_level, fig_path, 'bio', 'pre')
    save_3_levels(test_w_list_level, fig_path, 'bio', 'w')
    print('test saved at:', fig_path)
    
    psnr_value, ssim_value, rmse = metrics(pred_test_list[0], test_o_list[0])
    print('test performance:', psnr_value.round(2), ssim_value.round(2), rmse.round(2))
    
test()