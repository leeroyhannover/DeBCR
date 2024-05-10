import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
    
# visualization for two images
def subShow3(IMG1, IMG2, IMG3):
    # plt.figure(figsize=(2, 3), dpi=250)
    
    plt.subplot(1,3,1)
    plt.imshow(IMG1, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.imshow(IMG2, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1,3,3)
    plt.imshow(IMG3, cmap='gray')
    plt.axis('off')
    plt.show()

# multiple inputs for the model
def multi_input(w_img, o_img):
    
    w_0, o_0 = w_img, o_img
    w_2, o_2 = w_0[:, ::2, ::2, :], o_0[:, ::2, ::2, :]
    w_4, o_4 = w_0[:, ::4, ::4, :], o_0[:, ::4, ::4, :]
    
    return [w_0, w_2, w_4], [o_0, o_2, o_4]

def save_svg(image_stack, domain, kind, path):
    # Define the path where you want to save the SVG files
    output_path = path
    os.makedirs(output_path, exist_ok=True)

    # Loop through the image stack and save each element as an SVG file
    for i in range(image_stack.shape[0]):
        image = image_stack[i, :, :, 0]

        plt.figure(figsize=(1.28, 1.28), dpi=300)  # Set figsize and dpi to match the 128x128 size
        plt.imshow(image, cmap='inferno')
        plt.axis('off')
        svg_filename = os.path.join(output_path, f"{str(kind)}_{i:03d}_{str(domain)}.svg")
        plt.savefig(svg_filename, format='svg', bbox_inches='tight', pad_inches=0)
        plt.close()

def save_3_levels(image_list, output_path, domain, kind):
    
    for i in range(len(image_list)):
        image = image_list[i]
        plt.figure(figsize=(1.28, 1.28), dpi=300)  # Set figsize and dpi to match the 128x128 size
        plt.imshow(image, cmap='inferno')
        plt.axis('off')
        svg_filename = os.path.join(output_path, f"{str(kind)}_{i:03d}_{str(domain)}.svg")
        plt.savefig(svg_filename, format='svg', bbox_inches='tight', pad_inches=0)
    print('finish save')
    
def save_1_levels(image_list, output_path, domain, kind, color='inferno'):
    
    image = image_list[0]
    plt.figure(figsize=(1.28, 1.28), dpi=300)  # Set figsize and dpi to match the 128x128 size
    plt.imshow(image, cmap=color)
    plt.axis('off')
    svg_filename = os.path.join(output_path, f"{str(kind)}_{str(domain)}.svg")
    plt.savefig(svg_filename, format='svg', bbox_inches='tight', pad_inches=0)
    
    print('finish save')
   
def rescale(image_stack, MIN=0, MAX=1):
    # Rescale the whole stack
    if image_stack[0].max() != 1:
        image_scale = []
        for stack in range(image_stack.shape[0]):
            temp = image_stack[stack, ...]
            temp_scale = np.interp(temp, (temp.min(), temp.max()), (MIN, MAX))
            image_scale.append(temp_scale.astype('float64'))
    else:
        image_scale = image_stack
    return np.asarray(image_scale)

class DataGeneratorMix:
    def __init__(self, data_dir, data_list, batch_size, noise):
        self.data_dir = data_dir
        self.data_list = data_list
        self.batch_size = batch_size
        self.noise = noise

    def _rescale(self, image_stack, MIN=0, MAX=1):
        if image_stack[0].max() != 1:
            image_scale = [np.interp(temp, (temp.min(), temp.max()), (MIN, MAX)).astype('float64')
                           for temp in image_stack]
        else:
            image_scale = image_stack
        return np.asarray(image_scale)

    def image_loader(self):
        for index, dataset_name in enumerate(self.data_list):
            print('Loading dataset:', dataset_name)
            temp_dataset = np.load(os.path.join(self.data_dir, dataset_name))
            w_imgs, o_imgs = temp_dataset['w'], temp_dataset['o']
            
            while True:
                sample_indices = np.random.choice(w_imgs.shape[0], self.batch_size, replace=False)
                w_img_temp, o_temp = w_imgs[sample_indices], o_imgs[sample_indices]

                # Add noise
                if self.noise:
                    gaussian_sigma, lambda_poisson = np.random.uniform(0.0, 0.05), np.random.uniform(0.0, 0.1)
                    gaussian_noise = np.random.normal(0, gaussian_sigma, w_img_temp.shape)
                    poisson_noise = np.random.poisson(lambda_poisson, w_img_temp.shape)
                    w_img_temp = w_img_temp + 0.5 * gaussian_noise  # + 0.5 * poisson_noise

                # Rescale into [0, 1]
                w_img_temp = self._rescale(w_img_temp, MIN=0, MAX=1)
                o_temp = self._rescale(o_temp, MIN=0, MAX=1)

                yield w_img_temp, o_temp
