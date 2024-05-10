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
from models.DnCNN import *
from util.loss_func import *
from util.metrics import *
from util.data import *

def test(path = './weights/DnCNN/'):

    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)  # Make sure weights_init_kaiming is defined

    # Move the model to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    
    model_path = path + '75_best_net.pth'
    checkpoint = torch.load(model_path, map_location='cuda:0')  # Specify the correct GPU device
    model.load_state_dict(checkpoint)  # Load the model weights from the checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    test_dir = './data/test/'
    test_list = natsorted(os.listdir(test_dir))  
    
    test_raw = np.load(test_dir + test_list[0])  # select testset [imn, bio, storm, w_c]
    w_test_img, o_test_img = test_raw['w'], test_raw['gt']
    w_test_img, o_test_img = rescale(w_test_img), rescale(o_test_img)
    
    test_imgn = np.swapaxes(w_test_img, 1, 3)
    test_imgn = torch.FloatTensor(test_imgn).cuda() 
    
    with torch.no_grad():  # To disable gradient computation during inference
        pred_test = model(test_imgn)
        
    pred_test_img = model.predict(w_test_img)
    
    # save a rnadom fig under path
    NUM = random.randint(0, pred_test_list[0].shape[0]-1)
    fig_path = './results/DnCNN/'
    save_svg(np.expand_dims(pred_test_img[NUM],axis=0), 'DnCNN', 'pre', fig_path)
    save_svg(np.expand_dims(o_test_img[NUM],axis=0), 'DnCNN', 'o', fig_path)
    save_svg(np.expand_dims(w_test_img[NUM],axis=0), 'DnCNN', 'w', fig_path)
    print('test saved at:', fig_path)
    
    # metrics
    pred_test_norm = rescale(temp)
    psnr_value, ssim_value, rmse = metrics(pred_test_norm[...,:1], o_test_img[...,:1])
    print(psnr_value.round(2), ssim_value.round(2), rmse.round(2))
    
test()