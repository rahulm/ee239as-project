from __future__ import print_function
import os
from time import time
import argparse
import sys
import cv2
import pickle
import tensorflow as tf
import h5py
from keras import models as KM
from keras.models import load_model
from keras import backend as KB
from keras import layers as KL
from keras.datasets import mnist
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras import optimizers as KO
from keras.callbacks import LambdaCallback, TensorBoard
import matplotlib.pyplot as plt
import copy
from config import Config
import numpy as np
from keras.utils import plot_model
from plotting import plot_results
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
import pandas as pd
from model import get_model
from data_generator import generate_training_data, generate_validation_data


ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

if not os.path.isdir('output_images'):
    os.mkdir('output_images')

parser = argparse.ArgumentParser(description='Train VAE on MNIST or FACE.')
# TODO Add parsing arguments

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE on MNIST or FACE.')
    parser.add_argument('--mode',              required=False, default="train", help="train or inference", metavar="train or inf")
    parser.add_argument('--dataset',           required=False, default='./<addyourstuff>', help='Path to dataset', metavar="/path/to/dataset/")
    parser.add_argument('--name',              required=False, default='celeb', help='Name of dataset', metavar="/path/to/dataset/")
    parser.add_argument('--decoder',           required=False, default='naive', metavar="naive", help="decoder network style")
    parser.add_argument('--encoder',           required=False, default='naive', metavar="naive", help="backbone network style")
    parser.add_argument('--loss',              required=False, default='ce',    metavar="mse",   help="loss func")
    parser.add_argument('--logs',              required=False, default=DEFAULT_LOGS_DIR, metavar="/path/to/logs/", help='Logs and checkpoints directory')
    parser.add_argument('--tensorboard',       required=False, default=False,  help="Whether to use tensorboard callback or not", action='store_true')
    parser.add_argument('--save_weights',      required=False, default='vae_mlp_zhuface.h5', metavar='save_weights path', help="file to save weights from")
    parser.add_argument('--load_weights',      required=False, default=None, metavar='load_weights path', help="file to load weights from")
    parser.add_argument('--res',               required=False, default=28, metavar='resolution', type=int, help='Input resolution')
    parser.add_argument('--img_height',        required=False, default=28, metavar='image height', type=int, help='Input height')
    parser.add_argument('--img_width',         required=False, default=28, metavar='img_width', type=int, help='Input width')
    parser.add_argument('--channel',           required=False, default=3, metavar='channel', type=int, help='Input resolution channel')
    parser.add_argument('--lr',                required=False, default=0.0001, type=float)
    parser.add_argument('--epochs',            required=False, default=60,     type=int)             
    parser.add_argument('--gpus',              required=False, default=1,      type=int)
    parser.add_argument('--batch_size',        required=False, default=128,    type=int, help='must be divisible by number of gpus')
    parser.add_argument('--intermediate_dim',  required=False, default=512,    type=int, help='intermediate dim')
    parser.add_argument('--latent_dim',        required=False, default=2,      type=int, help='latent dim')
    parser.add_argument('--output_stride',     required=False, default=16,     type=int)
    parser.add_argument('--freeze_batchnorm',  required=False, default=0,      type=int)
    parser.add_argument('--initial_epoch',     required=False, default=0,      type=int)
    parser.add_argument('--use_subset',        required=False, default=False, help="Whether to use tensorboard callback or not", action='store_true')
    parser.add_argument('--opt',               required=False, metavar='save_weights path', help="Optimizer to select", default='adam')

    args = parser.parse_args()
    print('Mode: ', args.mode)
    print("Resolution: ", args.res)
    print("Encoder: ", args.encoder)
    print("Decoder: ", args.decoder)
    print('Dataset: ', args.dataset)
    print('Load weights file: ', args.load_weights)
    print('Save weights file: ', args.save_weights)
    print('Use tb? : ', args.tensorboard)

    # Relax encoder decoder restriction for now
    if args.encoder not in ['resnet101', 'resnet50', 'mobilenet', 'mobilenetv2', 'xception', 'naive', 'vae_light']:
        raise ValueError("The backbone you selected: ", args.backbone, 'not valid!')
    if args.decoder not in ['unet', 'deeplabv3+', 'naive_upsampling', 'naive', 'vae_light']:
        raise ValueError("The decoder you selected: ", args.decoder, "is invalid!")

    LOGS_DIR = args.logs
    MODELS_DIR = os.path.join(LOGS_DIR, args.encoder)
    load_weights_path = args.load_weights
    save_weights_path = args.save_weights
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
                          img_height=args.img_height,
                          img_width=args.img_width,
                          encoder=args.encoder,
                          dataset_root=args.dataset,
                          latent_dim=args.latent_dim,
                          intermediate_dim=args.intermediate_dim,
                          DATASET_SIZE=60000)

    # DATA LOADING HERE
    with tf.device('/cpu:0'):
        print('we are inside before get model')
        # TODO Change input shape when using convolutional networks
        (x_train, y_train), (x_test, y_test) = mnist.load_data()        
        image_size = x_train.shape[1]
        original_dim = image_size * image_size
        x_train = np.reshape(x_train, [-1, original_dim])
        x_test = np.reshape(x_test, [-1, original_dim])
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255     
        model, encoder, decoder = get_model(config=inuse_config, input_shape=(inuse_config.original_dim,))      
        models = (encoder, decoder)
        data = (x_test, y_test)

    def save_model(epoch, logs):
        model.save_weights(os.path.join(MODELS_DIR, str(epoch) + save_weights_path))
    callbacks_list = [LambdaCallback(on_epoch_end=save_model)]
    if args.tensorboard:
        tb = TensorBoard(log_dir=MODELS_DIR, histogram_freq=0, write_graph=True, write_images=False)
        callbacks_list.append(tb)

    # Model compilation
    args = parser.parse_args()
    models = (encoder, decoder)

    if args.loss == 'mse':
        reconstruction_loss = mse(model.inputs[0], model.outputs[0])
    elif args.loss == 'ce':
        reconstruction_loss = binary_crossentropy(model.inputs[0], model.outputs[0])

    reconstruction_loss *= inuse_config.original_dim
    z_log_var = encoder.get_layer('z_log_var').output
    z_mean = encoder.get_layer('z_mean').output
    kl_loss = 1 + z_log_var - KB.square(z_mean) - KB.exp(z_log_var)
    kl_loss = KB.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    loss = KB.mean(reconstruction_loss + kl_loss)
    model.add_loss(loss)

    if args.opt == 'adam':
        optimizer = KO.Adam(lr=args.lr)
    elif args.opt == 'rmsprop':
        optimizer = KO.RMSprop(lr=args.lr)

    model.compile(optimizer=optimizer)
    # model.summary()
    plot_model(model, to_file='dec_' + inuse_config.DECODER + 'enc_' + inuse_config.ENCODER +'vae_mlp_mine.png', show_shapes=True)

    print('args mode: ', args.mode)
    if args.mode == 'train':
        history = model.fit(x_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(x_test, None),
            callbacks=callbacks_list)

        model.save_weights('vae_mlp_mnist_aelight.h5')

        losses = {'loss':     history.history['loss'],
                  'val_loss': history.history['val_loss'],
                  'epoch':    history.epoch}
        with open(args.loss + 'history.pkl', 'wb') as pkl_file:
            pickle.dump(losses, pkl_file)

    else: ## INFERENCE
        # Mode should be inf here
        if load_weights_path is not None:
            print('loading weights from: ', load_weights_path, '...')
            model.load_weights(os.path.join(MODELS_DIR, load_weights_path), by_name=True)
            print('weights loaded!')

        
    # Plot results
    plot_results((model.layers[1], model.layers[2]),
                    data,
                    batch_size=args.batch_size,
                    model_name=args.loss + 'vae_faces')

        # Reconstruction loss run
        
    
        









