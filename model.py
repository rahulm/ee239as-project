from keras import layers as KL
from keras import models as KM
from keras import backend as KB
from keras.utils import plot_model
import tensorflow as tf
from processing import sampling
from keras import objectives
from keras.losses import binary_crossentropy, mse
from keras.applications import ResNet50

eps = 1e-5

def conv_BN_relu(x, filters, kernel_size=3, stride=1, dilation=1, padding='SAME'):
    x = KL.Conv2D(filters, 
                  kernel_size=kernel_size, 
                  strides=stride, 
                  dilation_rate=dilation, 
                  padding=padding, 
                  use_bias=False)(x)
    x = KL.BatchNormalization(epsilon=eps)(x)
    x = KL.Activation('relu')(x)
    return x

def conv_block(x, filters, kernel_size, stride, block_id=None, stage=None, padding='SAME'):
    L = KL.Conv2D(filters, 
                 (kernel_size, kernel_size), 
                 padding=padding,
                 strides=stride,
                #  activation='relu',
                 use_bias=False,
                 name='conv_stg_' + str(stage) +'_' + str(block_id  ))(x)
    L = KL.BatchNormalization()(L)
    L = KL.LeakyReLU()(L)

    return L



def get_encoder(input_tensor, config=None):
    # if config.name == 'celeb':
    #     latent_dim = config.latent_dim
    #     x = conv_BN_relu(input_tensor, filters=64, kernel_size=3, stride=1, dilation=1)
    #     x = KL.Flatten()(x)
    #     x = KL.Dense(config.intermediate_dim, activation='relu')(x)

    enc_input = input_tensor

    if config.ENCODER == 'resnet50':
        base = ResNet50(input_tensor=input_tensor, include_top=False, weights='imagenet')
        base_features = base.output
        enc_input = base_features
        enc_input = KL.GlobalAveragePooling2D(name='avg_pool')(enc_input)

    if config.ENCODER == 'naive':
        # Naive backbone
        x = conv_BN_relu(input_tensor, filters=32)
        x = conv_BN_relu(x, filters=32, kernel_size=3, stride=2)
        x = conv_BN_relu(x, filters=32, kernel_size=3, stride=2)
        x = conv_BN_relu(x, filters=32, kernel_size=3, stride=2)
        x = conv_BN_relu(x, filters=32, kernel_size=3, stride=2)
        # Note: You can either use GlobalAvgPooling or you can Flatten()
        # x = KL.GlobalAveragePooling2D(name='avg_pool')(x)
        # Can add max pool here if want
        x = KL.Flatten()(x)
        enc_input = x



    x = KL.Dense(config.intermediate_dim, activation='relu')(enc_input)

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

    x = KL.Reshape((1,1,config.intermediate_dim), input_shape=(config.intermediate_dim,))(x)

    # Shape (16,16, 512)
    x = KL.UpSampling2D((16,16))(x)

    x = conv_block(x, 
                   filters=64,
                   kernel_size=3,
                   stride=2,
                   block_id=2,
                   stage='dec')

    x = KL.UpSampling2D((8,8))(x)
    x = conv_block(x, 
                   filters=32,
                   kernel_size=3,
                   stride=2,
                   block_id=3,
                   stage='dec')

    x = KL.UpSampling2D((4,4))(x)
    x = conv_block(x, 
                   filters=3,
                   kernel_size=3,
                   stride=2,
                   block_id=4,
                   stage='dec')

    x = KL.UpSampling2D((4,4))(x)
    x = KL.Conv2D(3, 
                kernel_size=3, 
                strides=1, 
                padding='SAME', 
                use_bias=False)(x)

    x = KL.BatchNormalization(epsilon=eps)(x)
    outputs = KL.Activation('sigmoid')(x)

    # Add to outputs
    decoder = KM.Model(latent_inputs, outputs, name='decoder')
    return decoder

def get_vae(input_tensor, config=None):
    
    # Get encoder
    encoder = get_encoder(input_tensor = input_tensor, config=config)
    encoder.summary()

    # Get decoder ## Maybe these inputs are different from the ones before
    # Make spot for input then send in input
    latent_inputs = KL.Input(shape=(config.latent_dim,), name='z_sampling')
    decoder = get_decoder(latent_inputs=latent_inputs, config=config)
    x_decoded_mean = decoder(encoder(input_tensor)[2])
    decoder.summary()

    vae = KM.Model(input_tensor, x_decoded_mean, name='vae_mlp')

    return vae, encoder, decoder
    
def get_model(config=None, input_shape=(None, None, 1), input_tensor=None):
    if input_tensor is None:
        input_tensor = KL.Input(shape=input_shape, name = 'encoder_input')

    elif not KB.is_keras_tensor(input_tensor):
        input_tensor = KL.Input(tensor=input_tensor, name="encoder_input")

    return get_vae(input_tensor, config)