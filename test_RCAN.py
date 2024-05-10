# test on different dataset
import os 
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import random
from natsort import natsorted
import argparse

from util.utils import *
from models.RCAN import RCAN
from util.loss_func import *
from util.metrics import *
from util.data import *

def test(path = './weights/RCAN/'):

    parser = argparse.ArgumentParser(description='rCAN')

    parser.add_argument('--arch', type=str, default='RCAN')
    parser.add_argument('--images_dir', default='./data', type=str)
    parser.add_argument('--outf', default='./weights',type=str)
    parser.add_argument('--scale', type=int, default=1)  
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=5)
    parser.add_argument('--num_rcab', type=int, default=10)  
    parser.add_argument('--reduction', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=128)  # patch size
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_fast_loader', action='store_true')
    parser.add_argument("--epoch_iter", type=int, default=20, help='iterations in one epoch')
    parser.add_argument("--epoch_val", type=int, default=10, help='val per epochs')

    opt = parser.parse_args(args=[])
    
    
    model = RCAN(opt).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # print(model)

    model_path = path + '70_best_net.pth'
    checkpoint = torch.load(model_path, map_location=device)  # Load the saved model
    model.load_state_dict(checkpoint)  # Load the model weights
    model.eval()
    
    
    test_dir = './data/test/'
    test_list = natsorted(os.listdir(test_dir))  
    
    test_raw = np.load(test_dir + test_list[0])  # select testset [imn, bio, storm, w_c]
    w_test_img, o_test_img = test_raw['w'], test_raw['gt']
    
    w_test_img = np.repeat(w_test_img, repeats=3, axis=-1)
    o_test_img = np.repeat(o_test_img, repeats=3, axis=-1)
    w_test_img, o_test_img = rescale(w_test_img), rescale(o_test_img)
    
    test_imgn = np.swapaxes(w_test_img, 1, 3)
    test_imgn = torch.FloatTensor(test_imgn).cuda() 
    
    with torch.no_grad():  
        pred_test = generator(test_imgn)
        
    pred_test = pred_test.cpu().numpy()
    pred_test = np.swapaxes(pred_test, 1, 3)
    
    # save a rnadom fig under path
    # NUM = random.randint(0, pred_test_list[0].shape[0]-1)
    NUM = 24
    fig_path = './results/RCAN/'
    save_svg(np.expand_dims(pred_test_img[NUM],axis=0), 'RCAN', 'pre', fig_path)
    save_svg(np.expand_dims(o_test_img[NUM],axis=0), 'RCAN', 'o', fig_path)
    save_svg(np.expand_dims(w_test_img[NUM],axis=0), 'RCAN', 'w', fig_path)
    print('test saved at:', fig_path)
    
    pred_test_norm = rescale(pred_test)
    print(pred_test_norm.min(), pred_test_norm.max(),o_test_img.min(), o_test_img.max())
    psnr_value, ssim_value, rmse = metrics(pred_test_norm[...,:1], o_test_img[...,:1])
    print(psnr_value.round(2), ssim_value.round(2), rmse.round(2))
    
test()