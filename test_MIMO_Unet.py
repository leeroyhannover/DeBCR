# test on different dataset
import os 
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
from natsort import natsorted

import torch
from util.MIMO_utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from util.valid import _valid
import torch.nn.functional as F

from util.utils import *
from util.MIMO_config import parser
from models.MIMOUNet import build_net
from util.loss_func import *
from util.metrics import *
from util.data import *

import time

def test(path = './weights/MIMO_Unet/'):

    params = parser.parse_args(args=[])

    params.num_epoch = 50
    params.save_freq = 10
    params.model_name = 'MIMO-UNet'
    params.print_freq = 10
    params.valid_freq = 10
    
    eval_model = build_net(params.model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # state_dict = torch.load('weights_manual/Final.pkl')
    state_dict = torch.load(path + 'Best.pkl')
    eval_model.load_state_dict(state_dict['model'])
    torch.cuda.empty_cache()

    adder = Adder()
    eval_model.to(device)
    eval_model.eval()
    
    test_dir = './data/test/'
    test_list = natsorted(os.listdir(test_dir))  
    
    test_raw = np.load(test_dir + test_list[0])  # select testset [imn, bio, storm, w_c]
    w_test_img, o_test_img = test_raw['w'], test_raw['gt']
    w_test_img, o_test_img = rescale(w_test_img), rescale(o_test_img)

    # Record the start time
    start_time = time.time()

    with torch.no_grad():
        psnr_adder = Adder()
        
        input_img, label_img = np.swapaxes(w_test_img, 1, 3), np.swapaxes(o_test_img, 1, 3)
        input_img, label_img = np.repeat(input_img, repeats=3, axis=1), np.repeat(label_img, repeats=3, axis=1)
        input_img, label_img = torch.from_numpy(input_img), torch.from_numpy(label_img)
        input_img = input_img.to(device, dtype=torch.float)
        label_img = label_img.to(device, dtype=torch.float) # [None, 3, 128, 128]
        
        pred = eval_model(input_img)[2]
        print(pred.shape)
        
        
        pred_clip = torch.clamp(pred, 0, 1)
        pred_numpy = pred_clip.squeeze(0).cpu().numpy()
        label_numpy = label_img.squeeze(0).cpu().numpy()
        input_numpy = input_img.cpu().numpy()

    # Calculate and print the elapsed time
    elapsed_time = time.time() - start_time
    time_test_single = elapsed_time/ pred.shape[0]

    print('single image :', time_test_single)
    
    # save a rnadom fig under path
    # NUM = random.randint(0, pred_test_list[0].shape[0]-1)
    NUM = 24
    fig_path = './results/MIMO_Unet/'
    save_svg(np.expand_dims(pred_test_img[NUM],axis=0), 'MIMO_Unet', 'pre', fig_path)
    save_svg(np.expand_dims(o_test_img[NUM],axis=0), 'MIMO_Unet', 'o', fig_path)
    save_svg(np.expand_dims(w_test_img[NUM],axis=0), 'MIMO_Unet', 'w', fig_path)
    print('test saved at:', fig_path)
    
    # metrics
    pred_test_norm = np.swapaxes(pred_numpy[:,1:2, :,:], 1, 3)
    print(pred_test_norm.min(), pred_test_norm.max(),o_test_img.min(), o_test_img.max())
    psnr_value, ssim_value, rmse = metrics(pred_test_norm, o_test_img)
    print(psnr_value.round(2), ssim_value.round(2), rmse.round(2))
    
test()