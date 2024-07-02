import os 
import numpy as np
import keras
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import glob
import random
from natsort import natsorted

from util.utils import *
from models.DeBCR import *
from util.metrics import *
from util.data import *
from util.whole_img_tester import *


def test(args):
    print('task:', args.task_type)

    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_id)
    print('GPU ID: ', tf.config.list_physical_devices("GPU"))
    
    #####
    #config the model, load the weights
    weight_path = args.weight_path + str(args.task_type) + '/'
    eval_model = model_DeBCR()
    eval_model.compile(optimizer='adam', loss=loss_function_mimo, metrics=[metrics_func_mimo]) 

    # reload the check point
    checkpoint = tf.train.Checkpoint(model=eval_model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, weight_path, max_to_keep=5)

    # Specify the checkpoint you want to restore for testing
    checkpoint_to_restore = os.path.join(weight_path, 'ckpt-best')
    status = checkpoint.restore(checkpoint_to_restore)
    #status.assert_consumed()
    
    ### Test dataset
    # Input directory for testing data
    test_dir = args.testset_path + str(args.task_type) + '/'
    test_list = natsorted(os.listdir(test_dir))
    
    if args.task_type == '2D_denoising':
        test_raw = np.load(os.path.join(test_dir, test_list[0]))
        w_test_img, o_test_img = test_raw['low'], test_raw['gt']
        w_test_img, o_test_img = np.expand_dims(w_test_img, axis=3), np.expand_dims(o_test_img, axis=3)
        w_test_img, o_test_img = rescale(w_test_img), rescale(o_test_img)

        test_w_list, test_o_list = multi_input(w_test_img, o_test_img)
        pred_test_list = eval_model.predict(test_w_list) 
        
    elif args.task_type == '3D_denoising':
        if args.whole_predict:
            
            print('predict the whole image')
            test_raw = np.load(test_dir + test_list[1])
            w_test_img, o_test_img = test_raw['low'], test_raw['gt'] # (45, 1024, 512)
            patch_predictor = PatchPred_3D_denoiser(eval_model)
            # pred_result = patch_predictor.pred_one_patch(w_test_img, o_test_img) 
            stacked_pred_result = patch_predictor.pred_stack_patch(w_test_img, o_test_img) # whole xz
            
            # save the xz
            NUM = 511
            re_temp_pred, temp_w, temp_o = stacked_pred_result[:, NUM, :], w_test_img[:, NUM, :], o_test_img[:, NUM, :]
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(re_temp_pred, cmap='inferno')
            axes[0].set_title('Predicted')
            axes[0].axis('off')
            axes[1].imshow(temp_w, cmap='inferno')
            axes[1].set_title('Input')
            axes[1].axis('off')
            axes[2].imshow(temp_o, cmap='inferno')
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')

            plt.savefig(args.results_path + str(args.task_type) + '/comparison_yz.svg', format='svg')
            plt.close(fig)

        else:
            test_raw = np.load(test_dir + test_list[0])
            w_test_img, o_test_img = test_raw['low'], test_raw['gt']
            w_test_img, o_test_img = np.expand_dims(w_test_img, axis=3), np.expand_dims(o_test_img, axis=3)
            w_test_img, o_test_img = rescale(w_test_img), rescale(o_test_img)
            test_w_list, test_o_list = multi_input(w_test_img, o_test_img)
            pred_test_list = eval_model.predict(test_w_list)
    
    elif args.task_type == 'bright_SR':
        
        test_raw = np.load(test_dir + test_list[0])
        w_test_img, o_test_img = test_raw['low'], test_raw['gt']
        w_test_img, o_test_img = np.expand_dims(w_test_img, axis=3), np.expand_dims(o_test_img, axis=3)
        w_test_img, o_test_img = rescale(w_test_img), rescale(o_test_img)
        test_w_list, test_o_list = multi_input(w_test_img, o_test_img)
        pred_test_list = eval_model.predict(test_w_list)
    
    elif args.task_type == 'confocal_SR':
        
        test_raw = np.load(test_dir + test_list[0])
        w_test_img, o_test_img = test_raw['low'], test_raw['gt']
        w_test_img, o_test_img = np.expand_dims(w_test_img, axis=3), np.expand_dims(o_test_img, axis=3)
        w_test_img, o_test_img = rescale(w_test_img), rescale(o_test_img)
        test_w_list, test_o_list = multi_input(w_test_img, o_test_img)
        pred_test_list = eval_model.predict(test_w_list)
    
    elif args.task_type == 'low_ET':
        if args.whole_predict:
            pass  # for the prediction in whole image
        
        else:
            test_raw = np.load(os.path.join(test_dir, test_list[0]))
            X_test_patches, Y_test_patches = test_raw['X'], test_raw['Y']  # even, odd
            X_test_patches, Y_test_patches = np.expand_dims(X_test_patches, axis=3), np.expand_dims(Y_test_patches, axis=3)
            X_test_patches, Y_test_patches = rescale(X_test_patches), rescale(Y_test_patches)

            X_test_list, Y_test_list = multi_input(X_test_patches, Y_test_patches)
            
            pred_X_test_list = eval_model.predict(X_test_list)
            pred_Y_test_list = eval_model.predict(Y_test_list)
    
    elif args.task_type == 'high_ET':
        if args.whole_predict:
            pass  # for the prediction in whole image
            
        else:
            #test_raw = np.load(os.path.join(test_dir, test_list[1]))
            test_raw = np.load(os.path.join(test_dir, test_list[0]))
            print(test_raw['X'].shape, test_raw['Y'].shape)
            
            X_test_patches, Y_test_patches = test_raw['X'], test_raw['Y']  # even, odd
            X_test_patches, Y_test_patches = np.expand_dims(X_test_patches, axis=3), np.expand_dims(Y_test_patches, axis=3)
            X_test_patches, Y_test_patches = rescale(X_test_patches), rescale(Y_test_patches)
            
            X_test_patches, Y_test_patches = clip_intensity(X_test_patches), clip_intensity(Y_test_patches) 
            X_test_list, Y_test_list = multi_input(X_test_patches, Y_test_patches)

            pred_X_test_list = eval_model.predict(X_test_list)
            pred_Y_test_list = eval_model.predict(Y_test_list)

            data_path = args.results_path + str(args.task_type) + '/data'
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            
            pred_X_test_arr = np.asarray(pred_X_test_list[0]) 
            pred_Y_test_arr = np.asarray(pred_Y_test_list[0])
            pred_X_test_arr, pred_Y_test_arr = np.squeeze(pred_X_test_arr), np.squeeze(pred_Y_test_arr)
            print(pred_X_test_arr.shape, pred_Y_test_arr.shape)
            
            np.savez(data_path + '/results.npz', X=pred_X_test_arr, Y=pred_Y_test_arr)
    else:
        print('illegal task')
        
        
    #####
    # evaluation by the metrics
    if args.microscopy == 'EM':
        print('EM evaluation')
        pass
        # add FRC evaluation
        
    elif args.microscopy == 'LM' and not args.whole_predict:
        pred_test_norm, test_o_norm = rescale(pred_test_list[0]), rescale(test_o_list[0])
        psnr_value, ssim_value, rmse = metrics(pred_test_norm, test_o_norm)
        print('Test performance(PSNR, SSIM, RMSE)', psnr_value.round(2), ssim_value.round(2), rmse.round(2))
    else:
        print('illegal microscopy for evaluation')

    # save the results as fig
    if args.save_fig and not args.whole_predict:
        # save a random fig under path
        fig_path = args.results_path + str(args.task_type) + '/'
        
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        
        if args.microscopy == 'LM':
            eval_results = [psnr_value, ssim_value, rmse]
            save_grid_LM(pred_test_list[0], test_w_list[0], test_o_list[0], fig_path, str(args.task_type), eval_results)
            print('LM test results saved at:', fig_path)
            
        elif args.microscopy == 'EM':
            save_grid_EM(pred_X_test_list[0], X_test_list[0], fig_path, str(args.task_type), NUM=10)
            print('EM Test results saved at:', fig_path)
            
        else:
            print('wrong microscopy restoration for saving')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeBCR_tester")
    parser.add_argument("--microscopy", type=str, default='LM', help="LM or ET")
    parser.add_argument("--task_type", type=str, default='2D_denoising', help="2D_denoising,3D_denoising,bright_SR, confocal_SR, low_ET, high_ET")
    parser.add_argument("--weight_path", type=str, default='./weights/', help="path to load weight")
    parser.add_argument("--testset_path", type=str, default='./data/test/', help="path to load test datset")
    parser.add_argument("--save_fig", type=bool, default=False, help="save the figs or not")
    parser.add_argument("--results_path", type=str, default='./results/', help="path to save test results fig")
    parser.add_argument("--whole_predict", type=bool, default=False, help="predict the whole image for certain tasks")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to be used")

    args = parser.parse_args()
    
    test(args)

# python test_DeBCR.py --microscopy LM --task 2D_denoising --save_fig True
# python test_DeBCR.py --microscopy EM --task high_ET --save_fig True
# python test_DeBCR.py --microscopy LM --task bright_SR --save_fig True
