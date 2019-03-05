from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import argparse
from config import Config
import matplotlib.image as mpimg
import cv2
import glob
from processing import read_image, transform, resize_image, normalize, unnormalize, preprocess_image, preprocess_image_val

def generate_training_data(data, config, batch_size=4, resize=True):
    # Data passed in will be train data as train_partition
    # If resize is true you will resize to the passed is config.img_height, config.img_width
    if config is None:
        raise ValueError('inside gen train data config is None')
    
    # image_batch = np.zeros((batch_size,config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNEL))
    image_batch = np.zeros((batch_size,config.img_height, config.img_width, config.IMG_CHANNEL))
    while True:
        for i in range(batch_size):
            idx = np.random.randint(len(data))
            row = data.iloc[idx]
#             data.reset_index() 
            x = preprocess_image(row['image_id'], to_height=config.img_height, to_width=config.img_width, resize=resize)
            x = x / 255.
            image_batch[i] = x
            
        yield image_batch, image_batch
# Note: you can make this convolutional without flattening

def generate_validation_data(data, config, batch_size=4, resize=True):
    if config is None:
        raise ValueError('inside gen train data config is None')
    
    image_batch = np.zeros((batch_size,config.img_height*config.img_width* config.IMG_CHANNEL))
    while True:
        for i in range(batch_size):
            idx = np.random.randint(len(data))
            row = data.iloc[idx]
#             data.reset_index() 
            x = preprocess_image_val(row['image_id'], to_height=config.img_height, to_width=config.img_width, resize=resize)
            x = x / 255.
            image_batch[i] = x

            
        yield image_batch, image_batch

# def generate_validation_data(data, config):
#     if config is None:
#         raise ValueError('inside gen train data config is None')
#     while True:
#             idx = np.random.randint(len(data))
#             row = data.iloc[idx]
# #             data.reset_index() 
#             x = preprocess_image_val(row['image_id'], to_height=config.IMG_SIZE, to_width=config.IMG_SIZE)
#             x = normalize(x, kind='max_min')
            
#             yield x, x



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE on MNIST or FACE.')
    

    parser.add_argument('--mode', required=False,
                    metavar="train or inf", 
                    default="train",
                    help="train or inference")

    # TODO (Jon) - change required to True here
    # and add dataset path
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        default='./<addyourstuff>',
                        help='Path to dataset')
                        
    parser.add_argument('--name', required=False,
                        metavar="/path/to/dataset/",
                        default='celeb',
                        help='Name of dataset')

    parser.add_argument('--decoder', required=True,
                            metavar="naive",
                            help="decoder network style")

    parser.add_argument('--encoder', required=True,
                            metavar="naive",
                            help="backbone network style")

    parser.add_argument('--loss', required=True,
                            metavar="mse",
                            help="loss func")
   
    parser.add_argument('--res', required=False,
                    metavar='resolution',
                    type=int,
                    help='Input resolution',
                    default=28)
    
    parser.add_argument('--img_height', required=False,
                    metavar='image height',
                    type=int,
                    help='Input height',
                    default=28)
    
    parser.add_argument('--img_width', required=False,
                    metavar='img_width',
                    type=int,
                    help='Input width',
                    default=28)


    parser.add_argument('--channel', required=False,
                    metavar='resolution',
                    type=int,
                    help='Input resolution channel',
                    default=3)
    
    parser.add_argument('--load_weights', required=False,
                    metavar='load_weights path',
                    help="file to load weights from",
                    default=None)

    # TODO (Jon) - F -> T
    parser.add_argument('--save_weights', required=False,
                    metavar='save_weights path',
                    help="file to save weights from",
                    default='vae_mlp_celeb.h5')     

    parser.add_argument('--tensorboard', 
                    required=False,
                    default=False,
                    help="Whether to use tensorboard callback or not",
                    action='store_true')

    parser.add_argument('--lr', required=False,
                    default=0.0001,
                    type=float)

    parser.add_argument('--epochs', required=False,
                    default=60,
                    type=int)             

    parser.add_argument('--gpus', required=False,
                    default=1, type=int)

    parser.add_argument('--batch_size', required=False,
                    default=128, type=int,
                    help='must be divisible by number of gpus')

    parser.add_argument('--intermediate_dim', required=False,
                    default=512, type=int,
                    help='intermediate dim')

    parser.add_argument('--latent_dim', required=False,
                    default=2, type=int,
                    help='latent dim')
    
    parser.add_argument('--output_stride', required=False,
                    default=16, type=int)

    parser.add_argument('--freeze_batchnorm', required=False,
                    type=int, default=0)
    
    # Initial epoch for fine-tuning
    parser.add_argument('--initial_epoch', required=False,
                    type=int, default=0)

    args = parser.parse_args()
    print('Mode: ', args.mode)
    print("Resolution: ", args.res)
    print("Encoder: ", args.encoder)
    print("Decoder: ", args.decoder)
    print('Dataset: ', args.dataset)
    print('Load weights file: ', args.load_weights)
    print('Save weights file: ', args.save_weights)
    print('Use tb? : ', args.tensorboard)


    # Grab data

    # Obtain data
    dataset_root = '/home/odin/Downloads/Celeb'

    # read data
    images_root_path = os.path.join(dataset_root, 'img_align_celeba')

    data_partitions = pd.read_csv(os.path.join(dataset_root, 'list_eval_partition.csv'))

    landmarks = pd.read_csv(os.path.join(dataset_root, 'list_landmarks_align_celeba.csv'))

    crops = pd.read_csv(os.path.join(dataset_root, 'list_bbox_celeba.csv'))

    # Train test split
    train_df = data_partitions[data_partitions['partition']==0]
    val_df = data_partitions[data_partitions['partition']==1]
    test_df = data_partitions[data_partitions['partition']==2]

    # Instance config
    inuse_config = Config(name=args.name,
                          IMG_SIZE=args.res, 
                          BATCH_SIZE=args.batch_size,
                          IMG_CHANNEL=args.channel,
                          DATASET_SIZE = len(train_df) + len(val_df) + len(test_df),
                          img_height=args.img_height,
                          img_width=args.img_width)

    # Data generation sample
    train_datagen = generate_training_data(train_df, inuse_config, batch_size=inuse_config.BATCH_SIZE)
    val_datagen = generate_validation_data(val_df, inuse_config)

    sample_data = train_datagen.next()


    for samp in sample_data[0]:
        plt.imshow((samp * 255).astype(np.uint8))
        plt.show()