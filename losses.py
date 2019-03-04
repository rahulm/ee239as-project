import keras.backend as KB
import tensorflow as tf

from keras.losses import mse, binary_crossentropy
from keras import backend as KB
from keras import layers as KL

def add_custom_loss(model=None, config=None, kind='mse'):
    # defaults to mse
    # Adds custom loss as tensor
    encoder = model.layers[1]
    if kind not in ['mse', 'ce']:
        raise ValueError('Loss type unidentified')
    
    if config is None or config.original_dim is None:
        raise ValueError('Config missing data inside custom loss')
    

    if kind == 'mse':
        # Note: Change back if we do fully convolutional
        reconstruction_loss = mse(KL.Flatten()(model.inputs[0]), model.outputs[0])
    elif kind == 'ce':
        reconstruction_loss = binary_crossentropy(model.inputs[0], model.outputs[0])
    
    reconstruction_loss *= config.original_dim
    z_log_var = encoder.get_layer('z_log_var').output
    z_mean = encoder.get_layer('z_mean').output
    kl_loss = 1 + z_log_var - KB.square(z_mean) - KB.exp(z_log_var)
    kl_loss = KB.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    loss = KB.mean(reconstruction_loss + kl_loss)
    return loss


