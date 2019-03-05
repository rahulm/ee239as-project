from keras import backend as KB
import cv2
import numpy as np
import os

dataset_root = '/home/odin/Downloads/Celeb'
images_root_path = os.path.join(dataset_root, 'img_align_celeba')


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = KB.shape(z_mean)[0]
    dim = KB.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = KB.random_normal(shape=(batch, dim))
    return z_mean + KB.exp(0.5 * z_log_var) * epsilon

def read_image(impath):
    # takes in image path
    # returns image as [0,255] RGB read in original shape
    image = cv2.imread(impath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # can also do HSV
    return image    

def transform(image):
    # TODO: can do transform here
    return image

def resize_image(image, to_width, to_height):
    image = cv2.resize(image, (to_width, to_height), interpolation=cv2.INTER_AREA)
    return image

def normalize(image, kind='max_min'):
    if kind == 'max_min':
        return image / 127.5 - 1
    elif kind == 'mean_std':
        return (image - image.mean()) / image.std()
    # TODO Add for preloaded resnetnorm
    
def unnormalize(image, kind='max_min'):
    if kind == 'max_min':
        return (image +1) * 127.5
    elif kind == 'mean_std':
        return (image * image.std()) + image.mean()
    # TODO Add for preloaded resnetnorm
        

def preprocess_image(impath, images_root_path = None, to_width=None, to_height=None, resize=False):
    img = read_image(os.path.join(images_root_path, impath))
    img = transform(img)
    if resize:
        img = resize_image(img, to_width=to_width, to_height=to_height)

    return img

def preprocess_image_val(impath, images_root_path = None, to_width=None, to_height=None, resize=False):
    img = read_image(os.path.join(images_root_path, impath))
    if resize:
        img = resize_image(img)
    return img
    
def flip_image(image):
    return np.fliplr(image)
    
