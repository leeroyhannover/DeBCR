# test on different dataset
import os 
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
from natsort import natsorted

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from util.utils import *
from models.MPRNet import MPRNet
import util.MPRNet_losses as losses

from util.loss_func import *
from util.metrics import *
from util.data import *

def test(path = './weights/MPRNet/'):

    model_eval = MPRNet()
    model_eval.cuda()

    # Load the model weights
    model_path = path + '55_net.pth'
    checkpoint = torch.load(model_path, map_location='cuda:0')  # Specify the correct GPU device
    model_eval.load_state_dict(checkpoint)  # Load the model weights from the checkpoint

    # Set the model to evaluation mode
    model_eval.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    
    test_dir = './data/test/'
    test_list = natsorted(os.listdir(test_dir))  
    
    test_raw = np.load(test_dir + test_list[0])  # select testset [imn, bio, storm, w_c]
    w_test_img, o_test_img = test_raw['w'], test_raw['gt']
    w_test_img = np.repeat(w_test_img, repeats=3, axis=-1)
    o_test_img = np.repeat(o_test_img, repeats=3, axis=-1)
    w_test_img, o_test_img = rescale(w_test_img), rescale(o_test_img)
    
    test_imgn = np.swapaxes(w_test_img, 1, 3)
    test_imgn = torch.FloatTensor(test_imgn).cuda() 

    with torch.no_grad():  # To disable gradient computation during inference
        pred_test = model_eval(test_imgn)
        
    pred_test = pred_test.cpu().numpy()
    pred_test = np.swapaxes(pred_test, 1, 3)
    
    # save a rnadom fig under path
    # NUM = random.randint(0, pred_test_list[0].shape[0]-1)
    NUM = 24
    fig_path = './results/MPRNet/'
    save_svg(np.expand_dims(pred_test_img[NUM],axis=0), 'MPRNet', 'pre', fig_path)
    save_svg(np.expand_dims(o_test_img[NUM],axis=0), 'MPRNet', 'o', fig_path)
    save_svg(np.expand_dims(w_test_img[NUM],axis=0), 'MPRNet', 'w', fig_path)
    print('test saved at:', fig_path)
    
    pred_test_norm = rescale(pred_test)
    #print(pred_test_norm.min(), pred_test_norm.max(),o_test_img.min(), o_test_img.max())
    psnr_value, ssim_value, rmse = metrics(pred_test_norm[...,:1], o_test_img[...,:1])
    print(psnr_value.round(2), ssim_value.round(2), rmse.round(2))
    
test()