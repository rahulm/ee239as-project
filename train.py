from __future__ import print_function
import os
from time import time
import argparse
import sys
import tensorflow as tf
import h5py
from keras import models as KM
from keras import backend as KB
from keras import layers as KL
from keras.datasets import mnist
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback, TensorBoard
import matplotlib.pyplot as plt
from config import Config
import numpy as np
from keras.utils import plot_model
from plotting import plot_results
from losses import add_custom_loss
import pandas as pd
from model import get_model
from data_generator import generate_training_data, generate_validation_data

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

parser = argparse.ArgumentParser(description='Train VAE on MNIST or FACE.')
# TODO Add parsing arguments


# MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# FACES small Dataset
x_train = np.load('./data/face_images_train_aligned_F.npy')
x_test = np.load('./data/face_images_test_aligned_F.npy')


# Remove preprocessing from network so we can unprocess and preprocess w/ options
image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


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
                        default='mnist_binary',
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

    # parser.add_argument('--decoder', required=True,
    #                         metavar="naive",
    #                         help="decoder network style")

    parser.add_argument('--logs', required=False,
                            default=DEFAULT_LOGS_DIR,
                            metavar="/path/to/logs/",
                            help='Logs and checkpoints directory (default=logs/)')
    # TODO (Jon) change F to T
    parser.add_argument('--res', required=False,
                    metavar='resolution',
                    type=int,
                    help='Input resolution',
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
                    default='vae_mlp_mnist_default.h5')     

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

    if args.encoder not in ['resnet101', 'resnet50', 'mobilenet', 'mobilenetv2', 'xception', 'naive']:
        raise ValueError("The backbone you selected: ", args.backbone, 'not valid!')

    if args.decoder not in ['unet', 'deeplabv3+', 'naive_upsampling', 'naive']:
        raise ValueError("The decoder you selected: ", args.decoder, "is invalid!")
    
    LOGS_DIR = args.logs
    MODELS_DIR = os.path.join(LOGS_DIR, args.encoder)


    load_weights_path = args.load_weights
    save_weights_path = args.save_weights #'bbox_instance_resnet101_deeplabv3+.h5'

    if not os.path.isdir(LOGS_DIR):
	    os.makedirs(LOGS_DIR)

    if not os.path.isdir(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    MODELS_DIR = os.path.join(MODELS_DIR, args.decoder)

    if not os.path.isdir(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    inuse_config = Config(name=args.name,
                          IMG_SIZE=args.res, 
                          BATCH_SIZE=args.batch_size,
                          IMG_CHANNEL=args.channel,
                          DATASET_SIZE = len(train_df) + len(val_df) + len(test_df))

    # DATA LOADING HERE
    with tf.device('/cpu:0'):
        print('we are inside before get model')
        # TODO Change input shape when using convolutional networks

        # If MNIST:
        if inuse_config.name == 'mnist_binary':
            model, encoder, decoder = get_model(config=inuse_config, input_shape=(inuse_config.original_dim,))
        elif inuse_config.name == 'celeb':

            train_datagen = generate_training_data(train_df, inuse_config, batch_size=inuse_config.BATCH_SIZE)
            val_datagen = generate_validation_data(val_df, inuse_config)

            model, encoder, decoder = get_model(config=inuse_config, 
                                                input_tensor=KL.Input(shape=(inuse_config.original_dim,)))

    # Note: Can use this for fine-tuning pre-trained models later
    # if load_weights_path is not None and args.mode != 'train':
    #     print('loading weights from: ', load_weights_path, '...')
    #     model.load_weights(load_weights_path, by_name=True)
    #     print('weights loaded!')
    
    # TODO: Check if this works -> we may have to overwrite call function
    # or make custom frozen batchnorm layer (was issue in past)
    if args.freeze_batchnorm != 0:
        print('freezing batchnorm...')
        # Check through Encoder
        for l in encoder.layers:
            if isinstance(l, KL.BatchNormalization):
                l.trainable = False
        # Check through Decoder
        for l in decoder.layers:
            if isinstance(l, KL.BatchNormalization):
                l.trainable = False

    def save_model(epoch, logs):
        model.save(os.path.join(MODELS_DIR, save_weights_path))
    
    callbacks_list = [LambdaCallback(on_epoch_end=save_model)]

    # TODO: (Jon) - Hook in tensorboard
    if args.tensorboard:
        tb = TensorBoard(log_dir=MODELS_DIR, histogram_freq=0, write_graph=True, write_images=False)
        callbacks_list.append(tb)

    # Model compilation
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    
    # vae_loss = add_custom_loss(model=model, config=inuse_config, kind=args.loss)

    # model.add_loss(vae_loss)
    # # Note: you can do optimizer=Adam(lr=args.lr) here
    # model.compile(optimizer='adam')
    # model.summary()
    plot_model(model,
               to_file='vae_mlp_mine.png',
               show_shapes=True)
    print('args mode: ', args.mode)
    if args.mode == 'train':
        # model.fit(x_train,
        #           epochs = args.epochs,
        #           batch_size = inuse_config.BATCH_SIZE,
        #           validation_data=(x_test, None))
                ##   callbacks=callbacks_list) # Keep commented for now
                ## We will use callbacks list later when we have deep conv models to save on 
                ## each epoch
    
        model.fit_generator(train_datagen, 
                            steps_per_epoch=inuse_config.dataset_size//inuse_config.BATCH_SIZE,
                            epochs=args.epochs)


        model.save_weights(save_weights_path)
    else:
        if load_weights_path is not None:
            print('loading weights from: ', load_weights_path, '...')
            model.load_weights(load_weights_path, by_name=True)
            print('weights loaded!')
    
        # inference / generation pipeline goes here
        # Load weights if model not already loaded
        print('Model is already loaded')

    # Note: Encoder, Decoder = model.layers[1], model.layers[2]
    plot_results((model.layers[1], model.layers[2]),
                 data,
                 batch_size=args.batch_size,
                 model_name='vae_faces')
    
        









