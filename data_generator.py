from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from config import Config
import matplotlib.image as mpimg
import cv2
import glob
from processing import read_image, transform, resize_image, normalize, unnormalize, preprocess_image, preprocess_image_val

def generate_training_data(data, config, batch_size=4):
    # Data passed in will be train data as train_partition
    if config is None:
        raise ValueError('inside gen train data config is None')
    
    # image_batch = np.zeros((batch_size,config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNEL))
    image_batch = np.zeros((batch_size,config.IMG_SIZE*config.IMG_SIZE*config.IMG_CHANNEL))
    while True:
        for i in range(batch_size):
            idx = np.random.randint(len(data))
            row = data.iloc[idx]
#             data.reset_index() 
            x = preprocess_image(row['image_id'], to_height=config.IMG_SIZE, to_width=config.IMG_SIZE)
            # x = normalize(x, kind='mean_std')
            x = x / 255.
            image_batch[i] = np.reshape(x, (config.IMG_SIZE*config.IMG_SIZE*config.IMG_CHANNEL))
        yield image_batch, image_batch
# Note: you can make this convolutional without flattening

# def generate_validation_data(data, config, batch_size=4):
#     if config is None:
#         raise ValueError('inside gen train data config is None')
    
#     image_batch = np.zeros((batch_size,config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNEL))
#     while True:
#         for i in range(batch_size):
#             idx = np.random.randint(len(data))
#             row = data.iloc[idx]
# #             data.reset_index() 
#             x = preprocess_image_val(row['image_id'], to_height=config.IMG_SIZE, to_width=config.IMG_SIZE)
#             image_batch[i] = normalize(x, kind='max_min')
            
            
#         yield image_batch, image_batch

def generate_validation_data(data, config):
    if config is None:
        raise ValueError('inside gen train data config is None')
    while True:
            idx = np.random.randint(len(data))
            row = data.iloc[idx]
#             data.reset_index() 
            x = preprocess_image_val(row['image_id'], to_height=config.IMG_SIZE, to_width=config.IMG_SIZE)
            x = normalize(x, kind='max_min')
            
            yield x, x