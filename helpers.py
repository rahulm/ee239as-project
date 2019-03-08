import skimage.io
import skimage.color
import sklearn.model_selection
import scipy.io
import os
import numpy as np


def make_folder(foldername):
    try:
        os.makedirs(foldername)
    except:
        pass

def rescale(image_to_rescale):
    min_v = np.min(image_to_rescale)
    max_v = np.max(image_to_rescale)
    scaled = ((image_to_rescale - min_v) / (max_v - min_v))
    return scaled

def read_images(foldername, filenames, extension):
    return np.asarray([skimage.color.rgb2hsv(skimage.io.imread(os.path.join(foldername, fname) + extension))
                       for fname in sorted(filenames)])


def read_landmarks(foldername, filenames, extension):
    return np.asarray([scipy.io.loadmat(os.path.join(foldername, fname) + extension)['lms'] for fname in sorted(filenames)])


def split_train_test_data(img_folder, lm_folder, train_set_size):
    names = [os.path.splitext(imfile)[0] for imfile in os.listdir(img_folder)]
    images = read_images(img_folder, names, ".jpg")
    landmarks = read_landmarks(lm_folder, names, ".mat")
    
    return sklearn.model_selection.train_test_split(images, landmarks,
                                                    train_size=train_set_size,
                                                    test_size=None,
                                                    stratify=None,
                                                    shuffle=False)

def split_train_test_data_gender(img_folder, lm_folder, train_set_size):
    names = [os.path.splitext(file)[0] for file in os.listdir(img_folder)]
    images = read_images(img_folder, names, ".jpg")
    landmarks = read_landmarks(lm_folder, names, ".mat")
    
    return sklearn.model_selection.train_test_split(images, landmarks,
                                                    train_size=train_set_size,
                                                    test_size=None,
                                                    stratify=None,
                                                    shuffle=False)


