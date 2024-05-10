# test on different dataset
import os 
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import random
from natsort import natsorted
from tqdm import tqdm

from util.utils import *
from models.Unet import *
from util.loss_func import *
from util.metrics import *
from util.data import *
from models.ddpm import train_model
from util.ddpm_utils import variance_schedule

def ddpm_obtain_sr_img(fx, timesteps_test, p_model, SMOOTH=0.01):
    pred_sr = np.random.normal(0, 1, fx.shape)
    gamma_vec_test, alpha_vec_test = variance_schedule(timesteps_test, schedule_type='linear', min_beta=1e-4,
                                                       max_beta=3e-2) 
    smooth_factor = SMOOTH
    for t in tqdm(range(timesteps_test, 0, -1)):
        z = np.random.normal(0, 1, fx.shape)
        if t == 1:
            z = 0
        alpha_t = alpha_vec_test[t - 1]
        gamma_t = gamma_vec_test[t - 1]
        gamma_tm1 = gamma_vec_test[t - 2]
        beta_t = 1 - alpha_t
        gamma_t_inp = np.ones((fx.shape[0], 1)) * np.reshape(gamma_t, (1, 1))
        pred_noise = p_model.predict([pred_sr, fx, gamma_t_inp], verbose=False)

        alpha_factor = beta_t / np.sqrt(1 - gamma_t)
        beta_factor = (1 - gamma_tm1) * beta_t / (1 - gamma_t)
        pred_sr = 1 / np.sqrt(alpha_t) * (pred_sr - alpha_factor * pred_noise) + np.sqrt(
            beta_t) * z - smooth_factor * np.sign(pred_sr) * np.sqrt(beta_t)

    return pred_sr


def test(path = './weights/ddpm/'):
    
    timesteps_test = 200
    lr = 1e-4
    opt_m = tf.keras.optimizers.Adam(lr)
    p_model_eval, t_model_eval = train_model((128,128,1), 1)
    t_model_eval.compile(loss='mse', optimizer=opt_m)  # 用于sinus embedding的    

    t_model_eval.load_weights(path + 'model_best.h5') # for the tarining on imagenet

    test_dir = './data/test/'
    test_list = natsorted(os.listdir(test_dir))  

    test_raw = np.load(test_dir + test_list[0])  # select testset [imn, bio, storm, w_c]
    w_test_img, o_test_img = test_raw['w'], test_raw['gt']

    pred_test_img = ddpm_obtain_sr_img((w_test_img-0.5)/0.5, timesteps_test, p_model_eval, 0.01)

    # save a rnadom fig under path
    # NUM = random.randint(0, pred_test_list[0].shape[0]-1)
    NUM = 24
    fig_path = './results/ddpm/'
    save_svg(np.expand_dims(pred_test_img[NUM],axis=0), 'ddpm', 'pre', fig_path)
    save_svg(np.expand_dims(o_test_img[NUM],axis=0), 'ddpm', 'o', fig_path)
    save_svg(np.expand_dims(w_test_img[NUM],axis=0), 'ddpm', 'w', fig_path)
    print('test saved at:', fig_path)

    psnr_value, ssim_value, rmse = metrics(rescale(pred_test_img), o_test_img)
    print('test performance:', psnr_value.round(2), ssim_value.round(2), rmse.round(2))
    
test()