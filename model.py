from keras import layers as KL
from keras import models as KM
from keras import backend as KB
from keras.utils import plot_model
import tensorflow as tf
from processing import sampling
from keras import objectives
from keras.losses import binary_crossentropy, mse

eps = 1e-5

def conv_BN_relu(x, filters, kernel_size=3, stride=1, dilation=1):
    x = KL.Conv2D(filters, kernel_size=kernel_size, strides=stride, dilation_rate=dilation, padding='same', use_bias=False)(x)
    x = KL.BatchNormalization(epsilon=eps)(x)
    x = KL.Activation('relu')(x)
    return x


def get_encoder(input_tensor, config=None):
    # if config.name == 'celeb':
    #     latent_dim = config.latent_dim
    #     x = conv_BN_relu(input_tensor, filters=64, kernel_size=3, stride=1, dilation=1)
    #     x = KL.Flatten()(x)
    #     x = KL.Dense(config.intermediate_dim, activation='relu')(x)

    

    latent_dim = config.latent_dim
    x = KL.Dense(config.intermediate_dim, activation='relu')(input_tensor)

    # For Mnist
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
    # Add to outputs
    decoder = KM.Model(latent_inputs, outputs, name='decoder')
    return decoder
    


def get_vae(input_tensor, config=None):
    
    # Get encoder
    encoder = get_encoder(input_tensor = input_tensor, config=config)
    encoder.summary()

    [z_mean, z_log_var, z] = encoder.outputs


    # Get decoder
    latent_inputs = KL.Input(shape=(config.latent_dim,), name='z_sampling')
    decoder = get_decoder(latent_inputs=latent_inputs, config=config)
    x_decoded_mean = decoder(encoder(input_tensor)[2])

    vae = KM.Model(input_tensor, x_decoded_mean, name='vae_mlp')

    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.binary_crossentropy(input_tensor, x_decoded_mean)
        kl_loss = -0.5 * KB.mean(1 + z_log_var - KB.square(z_mean) - KB.exp(z_log_var))
        loss = xent_loss + kl_loss
        return loss
    
    vae.compile(optimizer='adam', loss=vae_loss)

    return vae, encoder, decoder

        
    
# TODO (Jon) - Add in preprocessing and unprocessing
def get_model(config=None, input_shape=(None, None, 1), input_tensor=None):
    if input_tensor is None:
        input_tensor = KL.Input(shape=input_shape, name = 'encoder_input')

    elif not KB.is_keras_tensor(input_tensor):
        input_tensor = KL.Input(tensor=input_tensor, name="encoder_input")

    return get_vae(input_tensor, config)