from __future__ import print_function
import os
from time import time
import argparse
import sys
import pickle
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
import copy
import glob
from processing import *
from config import Config
import numpy as np
from keras.utils import plot_model
from plotting import plot_results
# Note: we should also try these loss functions
from losses import add_custom_loss
from keras.losses import mse, binary_crossentropy, sparse_categorical_crossentropy,categorical_crossentropy
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
                    default='vae_mlp_celeb2.h5')     

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

    parser.add_argument('--use_subset', 
                    required=False,
                    default=False,
                    help="Whether to use tensorboard callback or not",
                    action='store_true')

    parser.add_argument('--tpu', 
                    required=False,
                    default=False,
                    help="Whether to use TPU or not.",
                    action='store_true')


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


    # read data
    images_root_path = os.path.join(args.dataset, 'img_align_celeba')
    data_partitions = pd.read_csv(os.path.join(args.dataset, 'list_eval_partition.csv'))
    # landmarks = pd.read_csv(os.path.join(args.dataset, 'list_landmarks_align_celeba.csv'))
    # crops = pd.read_csv(os.path.join(args.dataset, 'list_bbox_celeba.csv'))

    # Train test split
    train_df = data_partitions[data_partitions['partition']==0]
    val_df = data_partitions[data_partitions['partition']==1]
    test_df = data_partitions[data_partitions['partition']==2]

    # Try loading CIFAR-10
    # Try loading diff dataset



    if args.use_subset:
        print('Using subset of original dataset')
        train_df = train_df[0:19000]
        val_df = val_df[0:2000]
        test_df = test_df[0:1000]

    inuse_config = Config(name=args.name,
                          IMG_SIZE=args.res, 
                          BATCH_SIZE=args.batch_size,
                          IMG_CHANNEL=args.channel,
                          DATASET_SIZE = len(train_df) + len(val_df) + len(test_df),
                          img_height=args.img_height,
                          img_width=args.img_width,
                          encoder=args.encoder,
                          dataset_root=args.dataset)

    

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

            model, encoder, decoder = get_model(config=inuse_config, input_shape=(256,256,3)) 
                                                
            plot_model(model, to_file='linear_model.png')
            plot_model(encoder, to_file='linear_encoder.png')
            plot_model(decoder, to_file='linear_decoder.png')
            

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
        model.save_weights(os.path.join(MODELS_DIR, str(epoch) + save_weights_path))
    
    callbacks_list = [LambdaCallback(on_epoch_end=save_model)]

    # TODO: (Jon) - Hook in tensorboard
    if args.tensorboard:
        tb = TensorBoard(log_dir=MODELS_DIR, histogram_freq=0, write_graph=True, write_images=False)
        callbacks_list.append(tb)

    # Model compilation
    args = parser.parse_args()
    models = (encoder, decoder)

    if args.loss == 'mse':
        reconstruction_loss = mse
    elif args.loss == 'ce':
        # Change to sparse_categorical crossentropy
        reconstruction_loss = sparse_categorical_crossentropy
    # Adds KL Loss
    z_log_var = encoder.get_layer('z_log_var').output
    z_mean = encoder.get_layer('z_mean').output
    kl_loss = 1 + z_log_var - KB.square(z_mean) - KB.exp(z_log_var)
    kl_loss = KB.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    kl_loss = KB.mean(kl_loss)
    # Note: might need to take mean of it here

    model.add_loss(kl_loss)
    # # Note: you can do optimizer=Adam(lr=args.lr) here
    model.compile(optimizer='adam', loss=reconstruction_loss)
    # model.summary()
    plot_model(model,
               to_file='vae_mlp_mine.png',
               show_shapes=True)

    if args.tpu:
        # TODO: Rahul === uncomment here
        # currently hard-coding tpu_name to the VM used
        # tpu_name = 'ee239asproject'
        # tpu = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_name)
        # tpu_strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu)
        # model = tf.contrib.tpu.keras_to_tpu_model(
        #     model,
        #     strategy=tpu_strategy)

        training_images_paths = glob.glob(images_root_path + '/*.jpg')
        training_images_paths = sorted(training_images_paths)

        train_data_paths = training_images_paths[0:len(train_df)]
        val_data_paths = training_images_paths[len(train_df): len(train_df) + len(val_df)]
        test_data_paths = training_images_paths[len(train_df) + len(val_df): len(train_df) + len(val_df) + len(test_df)]

        # CUT according to data loaded in TPU instance
        tpu_last_data_idx = 4000 # TODO: Rahul add amount of images here
        train_data_paths = train_data_paths[0:1000]
        val_data_paths = val_data_paths[0:200]

        X_train = np.array([preprocess_image(impath, 
                                                images_root_path=images_root_path, 
                                                to_width=inuse_config.img_width, 
                                                to_height=inuse_config.img_width, 
                                                resize=True)/255. for impath in train_data_paths])
        X_val = np.array([preprocess_image_val(impath, 
                                                images_root_path=images_root_path, 
                                                to_width=inuse_config.img_width, 
                                                to_height=inuse_config.img_width, 
                                                resize=True)/255. for impath in val_data_paths])


    print('args mode: ', args.mode)
    # TPU TRAIN MODE
    if args.mode == 'train' and args.tpu:
        history = model.fit(X_train, X_train,
                  epochs = args.epochs,
                  batch_size = inuse_config.BATCH_SIZE,
                  validation_data=(X_val, None))
                #   callbacks=callbacks_list) # Keep commented for now
                # We will use callbacks list later when we have deep conv models to save on 
                # each epoch
        losses = {'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'epoch': history.epoch}
        with open('history.pkl', 'wb') as pkl_file:
            pickle.dump(losses, pkl_file)

        model.save_weights(save_weights_path)


    elif args.mode=='train':
        history = model.fit_generator(train_datagen, 
                            steps_per_epoch= len(train_df)//inuse_config.BATCH_SIZE,
                            epochs=args.epochs,
                            validation_data=val_datagen,
                            validation_steps= len(val_df)//4,
                            verbose=1,
                            callbacks=callbacks_list,
                            )

        losses = {'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'epoch': history.epoch}
        with open('history.pkl', 'wb') as pkl_file:
            pickle.dump(losses, pkl_file)

        model.save_weights(save_weights_path)
    else: ## INFERENCE
        if load_weights_path is not None:
            print('loading weights from: ', load_weights_path, '...')
            model.load_weights(os.path.join(MODELS_DIR, load_weights_path), by_name=True)
            print('weights loaded!')

            sample_im2test = train_datagen.next()[0][0]

            sample_image_resized = copy.deepcopy(sample_im2test).reshape((inuse_config.IMG_SIZE, inuse_config.IMG_SIZE, inuse_config.IMG_CHANNEL))
            sample_image_resized_and_scaled = (sample_image_resized * 255).astype(np.uint8)

            blob = model.predict([np.expand_dims(sample_im2test,axis=0)])[0]

            output_img = blob.reshape((inuse_config.IMG_SIZE, inuse_config.IMG_SIZE, inuse_config.IMG_CHANNEL))
            output_img = (output_img * 255).astype(np.uint8)

            plt.subplot(211)
            plt.title('sample_image')
            plt.imshow(sample_image_resized_and_scaled)

            plt.subplot(212)
            plt.title('reconstructed image')
            plt.imshow(output_img)
            plt.savefig('./output_images/sample_reconstruction.png')
        # inference / generation pipeline goes here
        # Load weights if model not already loaded
        print('Model is already loaded')

    # Note: Encoder, Decoder = model.layers[1], model.layers[2]
    # plot_results((model.layers[1], model.layers[2]),
    #              data,
    #              batch_size=args.batch_size,
    #              model_name='vae_faces')
    
        









