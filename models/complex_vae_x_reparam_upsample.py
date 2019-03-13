import torch
import torch.nn as nn
from torch.autograd import Variable

# class appearance_VAE(nn.Module):
class Model(nn.Module):
    MODEL_TYPE = 'vae'
    MODEL_NAME = "complex_vae_reparam_upsample"
    MODEL_DATASET = 'faces'
    
    def __init__(self, num_filters, latent_dim_size, use_cuda=False):
        super(Model, self).__init__()
        self.use_cuda = use_cuda
        self.latent_dim_size = latent_dim_size
        self.num_filters = num_filters

        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.num_filters, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.num_filters),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            
            nn.Conv2d(self.num_filters, self.num_filters * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.num_filters * 2),
            nn.LeakyReLU(),
            nn.Dropout2d(),

            nn.Conv2d(self.num_filters * 2, self.num_filters * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.num_filters * 4),
            nn.LeakyReLU(),
            nn.Dropout2d(),

            nn.Conv2d(self.num_filters * 4, self.num_filters*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.num_filters * 8),
            nn.LeakyReLU(),
            nn.Dropout2d(),
        )

        self.fc1 = nn.Sequential(                    
            nn.Linear(self.num_filters*8*8*8, self.latent_dim_size),
        )

        self.fc2 = nn.Sequential(                    
            nn.Linear(self.num_filters*8*8*8, self.latent_dim_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim_size, self.num_filters*8*8*8),
            nn.LeakyReLU(),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReplicationPad2d(1),
            nn.Conv2d(),
            nn.BatchNorm2d(),

            nn.Sigmoid(),
        )

    def reparametrize(self, mu, var):
        std = var.mul(0.5).exp_()
        # if torch.cuda.is_available():
        if self.use_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def get_latent_vec(self, x):
        before_sample = self.encoder(x)
        mu = self.fc1(before_sample.view(-1, self.num_filters*8*8*8))
        var = self.fc2(before_sample.view(-1, self.num_filters*8*8*8*8))

        latent_vec = self.reparametrize(mu, var)
        return mu, var, latent_vec
    
    def get_recon_from_latent_vec(self, latent_vec):
        x_recon = self.decoder(latent_vec.view(-1, self.latent_dim_size, 1, 1))
        return x_recon
        
    def forward(self, x):
        mu, var, latent_vec = self.get_latent_vec(x)
        x_recon = self.get_recon_from_latent_vec(latent_vec)
        return x_recon, mu, var
        