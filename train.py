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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from warper import plot

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
    parser.add_argument('--load_weights',      required=False, default=None,   metavar='load_weights path', help="file to load weights from")
    parser.add_argument('--res',               required=False, default=28,     type=int, metavar='resolution', help='Input resolution')
    parser.add_argument('--img_height',        required=False, default=28,     metavar='image height', type=int, help='Input height')
    parser.add_argument('--img_width',         required=False, default=28,     metavar='img_width', type=int, help='Input width')
    parser.add_argument('--channel',           required=False, default=3,      metavar='channel', type=int, help='Input resolution channel')
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
    parser.add_argument('--opt',               required=False, default='adam', metavar='save_weights path', help="Optimizer to select")

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
    plot_model(encoder, to_file='encoder_vae_mlp_mine.png', show_shapes=True)
    plot_model(decoder, to_file='decoder_vae_mlp_mine.png', show_shapes=True)

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

        filename = os.path.join(MODELS_DIR, save_weights_path[:-3] + '_losses.png')
        plt.plot(np.arange(args.epochs)+1, losses['loss'], label='train_loss')
        plt.plot(np.arange(args.epochs)+1, losses['val_loss'], label='val_loss')
        plt.xlabel('epochs')
        plt.ylabel('BCE loss')
        plt.title('Training and val loss')
        plt.legend()
        plt.gcf()
        plt.savefig(filename)
        plt.show()

    else: ## INFERENCE
        # Mode should be inf here
        if load_weights_path is not None:
            print('loading weights from: ', load_weights_path, '...')
            model.load_weights(os.path.join(MODELS_DIR, load_weights_path), by_name=True)
            print('weights loaded!')

        if args.latent_dim == 2:
            mse_score = model.evaluate(x_test)
            print('mse_score for latent dim 2: ', mse_score)
            filename = os.path.join(MODELS_DIR, load_weights_path[:-4] + 'vae_mean_dim2.png')
            # Perform latent dim analysis using latent dim 2
            z_mean, _, _ = encoder.predict(x_test, batch_size=args.batch_size)
            # Z_mean is shape (200, 2) for the case latent vector dim 2
            plt.figure(figsize=(12, 10))
            plt.title('Latent vector space in R2')
            plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
            plt.colorbar()
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.savefig(filename)
            plt.show()

            # Here we attempt to show sampled reconstructions from the latent space
            filename = os.path.join(MODELS_DIR, load_weights_path[:-4] + 'digits_over_latent_dim2.png')
            n = 30
            digit_size = 28
            figure = np.zeros((digit_size * n, digit_size * n))
            # linearly spaced coordinates corresponding to the 2D plot
            # of digit classes in the latent space
            grid_x = np.linspace(-4, 4, n)
            grid_y = np.linspace(-4, 4, n)[::-1]

            for i, yi in enumerate(grid_y):
                for j, xi in enumerate(grid_x):
                    z_sample = np.array([[xi, yi]])
                    x_decoded = decoder.predict(z_sample) * 1 # variance
                    digit = x_decoded[0].reshape(digit_size, digit_size)
                    figure[i * digit_size: (i + 1) * digit_size,
                        j * digit_size: (j + 1) * digit_size] = digit

            plt.figure(figsize=(10, 10))
            start_range = digit_size // 2
            end_range = n * digit_size + start_range + 1
            pixel_range = np.arange(start_range, end_range, digit_size)
            sample_range_x = np.round(grid_x, 1)
            sample_range_y = np.round(grid_y, 1)
            plt.xticks(pixel_range, sample_range_x)
            plt.yticks(pixel_range, sample_range_y)
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.imshow(figure, cmap='Greys_r')
            plt.savefig(filename)

            num_samples = 20
            x_recon = model.predict(x_test)
            x_recon = np.reshape(x_recon, (x_recon.shape[0], 28, 28, 1))[0:num_samples]
            plot(x_recon, Nh=4, Nc=5, channel=1, IMG_HEIGHT=28, IMG_WIDTH=28)
            plt.gcf()
            plt.savefig('output_images/reconstructed_images_vae_ld_2.jpg')
            
            x_test_samples = x_test.reshape(x_test.shape[0], 28,28,1)[0:num_samples]
            plot(x_test_samples, Nh=4, Nc=5, channel=1, IMG_HEIGHT=28, IMG_WIDTH=28)
            plt.gcf()
            plt.savefig('output_images/x_test_samples_ld_2.jpg')
            plt.show()
        else:
            # RESULT -> Values so entangles that we don't notice great results
            
            # Perform latent dim analysis using latent dim 2


            # Now perform PCA on z_mean to map it to a lower space
            # z_mean of shape (200, 100) for the case latent vector is 500 dim
            # Perform latent dimension analysis using PCA
            z_mean, _, _ = encoder.predict(x_test, batch_size=args.batch_size)

            preds = model.predict(x_test, batch_size=args.batch_size)
            mse_score = model.evaluate(x_test)
            print('mse_score: ', mse_score)


            # Cant sample latent space with 32 dim Z model
            filename = os.path.join(MODELS_DIR, load_weights_path[:-4] + 'digits_over_latent_dim32.png')
            n = 30
            digit_size = 28
            figure = np.zeros((28, digit_size * n))
            # linearly spaced coordinates corresponding to the 2D plot
            # of digit classes in the latent space
            grid_x = np.linspace(-10, 10, n)


            for j, xi in enumerate(grid_x):
                z_sample = np.random.normal(loc=0, scale=1, size=args.latent_dim)
                x_decoded = decoder.predict(np.expand_dims(z_sample, axis=0)) * 1 # variance
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[0:28, j * digit_size: (j + 1) * digit_size] = digit
            start_range = digit_size // 2
            end_range = n * digit_size + start_range + 1
            pixel_range = np.arange(start_range, end_range, digit_size)
            sample_range_x = np.round(grid_x, 1)
            plt.xticks(pixel_range, sample_range_x)
            plt.xlabel("z mean")
            plt.imshow(figure, cmap='Greys_r')
            plt.show()
            plt.gcf()
            plt.savefig(filename)
            
            # Reconstruction test
            num_samples = 20
            x_recon = model.predict(x_test)
            x_recon = np.reshape(x_recon, (x_recon.shape[0], 28, 28, 1))[0:num_samples]
            plot(x_recon, Nh=4, Nc=5, channel=1, IMG_HEIGHT=28, IMG_WIDTH=28)
            plt.gcf()
            plt.savefig('output_images/reconstructed_images_vae_ld_32.jpg')
            
            x_test_samples = x_test.reshape(x_test.shape[0], 28,28,1)[0:num_samples]
            plot(x_test_samples, Nh=4, Nc=5, channel=1, IMG_HEIGHT=28, IMG_WIDTH=28)
            plt.gcf()
            plt.savefig('output_images/x_test_samples_ld_32.jpg')

            # Almost a perfect reconstruction!
            # TODO: Compute MSE

             # Now visualize
            pca = PCA(n_components=2)
            z_mean_lowdim_pca = pca.fit_transform(z_mean)

            # Now visualize
            filename = os.path.join(MODELS_DIR, load_weights_path[:-4] + 'vae_mean_dim32PCA.png')
            plt.figure(figsize=(12, 10))
            plt.title('PCA Projected Latent vector space in R2 for Z=32 model')
            plt.scatter(z_mean_lowdim_pca[:, 0], z_mean_lowdim_pca[:, 1], c=y_test)
            plt.colorbar()
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.savefig(filename)
            plt.show()

            filename = os.path.join(MODELS_DIR, load_weights_path[:-4] + 'vae_mean_dim32TSNE.png')
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            z_mean_lowdim_tsne = tsne.fit_transform(z_mean)
            plt.title('T-SNE Projected Latent vector space in R2 for Z=32 model')
            plt.scatter(z_mean_lowdim_tsne[:, 0], z_mean_lowdim_tsne[:, 1], c=y_test)
            plt.colorbar()
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.savefig(filename)
            plt.show()



    # Plot results
    # plot_results((model.layers[1], model.layers[2]),
    #                 data,
    #                 batch_size=args.batch_size,
    #                 model_name=args.loss + 'vae_faces')

    # Reconstruction loss run

    # Sampling
    # Make 2 gaussians for Z 

    # digit_size = 28
    # encoder, decoder = models

    # z_sample = np.random.normal(loc=0.0, scale=1.0, size=args.latent_dim)
    # x_decoded = decoder.predict(np.expand_dims(z_sample,axis=0))

    # digit = x_decoded[0].reshape(digit_size, digit_size)

    # # Build Z latent vector
    # epsilon = 1
    # z_sample = np.array([0,0]) * epsilon
    # x_decoded = decoder.predict([z_sample])








