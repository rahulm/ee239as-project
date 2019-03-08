from keras import layers as KL
from keras import models as KM
from keras import backend as KB
from keras.utils import plot_model
import tensorflow as tf
from processing import sampling
from keras import objectives
from keras.losses import binary_crossentropy, mse
from keras.applications import ResNet50

def get_encoder(input_tensor, config=None):
    if config is None:
        raise ValueError('inside model.py inside encoder config is None')

    latent_dim = config.latent_dim

    x = KL.Dense(config.intermediate_dim, activation='relu')(input_tensor)
    z_mean = KL.Dense(config.latent_dim, name='z_mean')(x)
    z_log_var = KL.Dense(config.latent_dim, name='z_log_var')(x)

     # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = KL.Lambda(sampling, output_shape=(config.latent_dim,), name='z')([z_mean, z_log_var])

     # instantiate encoder model
    # encoder_outputs => [z_mean, z_log_var, z]
    encoder = KM.Model(input_tensor, [z_mean, z_log_var, z], name='encoder')
    return encoder

def get_decoder(latent_inputs = None, config=None):
    # build decoder model
    # latent_inputs = KL.Input(shape=(config.latent_dim,), name='z_sampling')
    x = KL.Dense(config.intermediate_dim, activation='relu')(latent_inputs) 
    outputs = KL.Dense(config.original_dim, activation='sigmoid')(x)
    decoder = KM.Model(latent_inputs, outputs, name='decoder')
    return decoder

def get_vae(input_tensor, config=None):
    # Get encoder
    encoder = get_encoder(input_tensor = input_tensor, config=config)
    encoder.summary()

        # Get decoder
    latent_inputs = KL.Input(shape=(config.latent_dim,), name='z_sampling')
    decoder = get_decoder(latent_inputs=latent_inputs, config=config)
    decoder.summary()
    plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
    outputs = decoder(encoder(input_tensor)[2])
    vae = KM.Model(input_tensor, outputs, name='vae_mlp')
    vae.summary()
    return vae, encoder, decoder

def get_model(config=None, input_shape=(None, None, 1), input_tensor=None):
    if input_tensor is None:
        input_tensor = KL.Input(input_shape, name = 'encoder_input')
    elif not KB.is_keras_tensor(input_tensor):
        input_tensor = KL.Input(tensor=input_tensor, name="encoder_input")
    return get_vae(input_tensor, config)
        