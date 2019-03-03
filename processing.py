from keras import backend as KB

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = KB.shape(z_mean)[0]
    dim = KB.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = KB.random_normal(shape=(batch, dim))
    return z_mean + KB.exp(0.5 * z_log_var) * epsilon