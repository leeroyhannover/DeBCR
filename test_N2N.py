# test on different dataset
import os 
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import random
from natsort import natsorted
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from util.utils import *
from models.N2N import *
from N2N_noise_model import get_noise_model

from util.loss_func import *
from util.metrics import *
from util.data import *
import argparse

def test(path = './weights/N2N/'):

    parser = argparse.ArgumentParser(description='N2N')

    parser.add_argument("--image_dir", type=str,
                        help="train image dir")
    parser.add_argument("--test_dir", type=str,
                        help="test image dir")
    parser.add_argument("--image_size", type=int, default=128,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=1000,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--steps", type=int, default=20,
                        help="steps per epoch")
    parser.add_argument("--loss", type=str, default="mse",
                        help="loss; mse', 'mae', or 'l0' is expected")
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default="./weights/",
                        help="checkpoint dir")
    parser.add_argument("--source_noise_model", type=str, default="gaussian,0,50",
                        help="noise model for source images")
    parser.add_argument("--target_noise_model", type=str, default="gaussian,0,50",
                        help="noise model for target images")
    parser.add_argument("--val_noise_model", type=str, default="gaussian,25,25",
                        help="noise model for validation source images")
    parser.add_argument("--val_target_noise_model", type=str, default="gaussian,0,0",
                        help="noise model for validation source images")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")

    args = parser.parse_args(args=[])
    
    loss_type = args.loss
    output_path = args.output_path
    eval_model = load_model(path + '/best.h5', custom_objects={'PSNR': PSNR})
    
    test_dir = './data/test/'
    test_list = natsorted(os.listdir(test_dir))  
    
    test_raw = np.load(test_dir + test_list[0])  # select testset [imn, bio, storm, w_c]
    w_test_img, o_test_img = test_raw['w'], test_raw['gt']
    
    w_test_img = np.repeat(w_test_img, repeats=3, axis=-1)
    o_test_img = np.repeat(o_test_img, repeats=3, axis=-1)
    w_test_img, o_test_img = rescale(w_test_img), rescale(o_test_img)
    
    pred_test_img = eval_model.predict(w_test_img)
    
    # save a rnadom fig under path
    # NUM = random.randint(0, pred_test_list[0].shape[0]-1)
    NUM = 24
    fig_path = './results/N2N/'
    save_svg(np.expand_dims(pred_test_img[NUM],axis=0), 'N2N', 'pre', fig_path)
    save_svg(np.expand_dims(o_test_img[NUM],axis=0), 'N2N', 'o', fig_path)
    save_svg(np.expand_dims(w_test_img[NUM],axis=0), 'N2N', 'w', fig_path)
    print('test saved at:', fig_path)
    
    pred_test_norm = rescale(pred_test)
    psnr_value, ssim_value, rmse = metrics(pred_test_norm[...,:1], o_test_img[...,:1])
    print('test performance:', psnr_value.round(2), ssim_value.round(2), rmse.round(2))
    
test()