# test on different dataset
import os 
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import random
from natsort import natsorted

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable

from util.utils import *
from models.ESRGAN.models import *
from models.ESRGAN.datasets import *
from util.loss_func import *
from util.metrics import *
from util.data import *
import argparse


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test(path = './weights/ESRGAN/'):

    parser = argparse.ArgumentParser(description='ESRGAN')

    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")  # batch_size 4
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=128, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=128, help="high res. image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels") 
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="epochs interval between model checkpoints")
    parser.add_argument("--residual_blocks", type=int, default=15, help="number of residual blocks in the generator")
    # original 23
    # parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
    parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
    parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
    parser.add_argument('--outf', default='./weights',type=str)
    parser.add_argument("--epoch_iter", type=int, default=20, help='iterations in one epoch')

    opt = parser.parse_args(args=[])

    # Initialize generator and discriminator
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)

    gen_path = path + 'generator_best_50.pth'
    checkpoint = torch.load(gen_path, map_location=device)  # Load the saved model
    generator.load_state_dict(checkpoint)
    # Set the model to evaluation mode
    generator.eval()
    
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
    fig_path = './results/ESRGAN/'
    save_svg(np.expand_dims(pred_test[NUM],axis=0), 'ESRGAN', 'pre', fig_path)
    save_svg(np.expand_dims(o_test_img[NUM],axis=0), 'ESRGAN', 'o', fig_path)
    save_svg(np.expand_dims(w_test_img[NUM],axis=0), 'ESRGAN', 'w', fig_path)
    print('test saved at:', fig_path)
    
    # metrics
    pred_test_norm = rescale(pred_test)
    #print(pred_test_norm.min(), pred_test_norm.max(),o_test_img.min(), o_test_img.max())
    psnr_value, ssim_value, rmse = metrics(pred_test_norm[...,:1], o_test_img[...,:1])
    print(psnr_value.round(2), ssim_value.round(2), rmse.round(2))
    
test()