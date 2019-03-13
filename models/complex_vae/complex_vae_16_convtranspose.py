from . import complex_vae_x_convtranspose

class Model(complex_vae_x_convtranspose.Model):
    
    def __init__(self, latent_dim_size, use_cuda=False):
        super(Model, self).__init__(num_filters=16, latent_dim_size=latent_dim_size, use_cuda=use_cuda)
