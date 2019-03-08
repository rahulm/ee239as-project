import torch
import torch.nn as nn
from torch.autograd import Variable

class landmark_VAE(nn.Module):
    def __init__(self, latent_dim_size):
        super(landmark_VAE, self).__init__()

        self.latent_dim_size = latent_dim_size

        self.encoder = nn.Sequential(
            nn.Linear(68*2, 100),
            nn.LeakyReLU(),
        )

        self.fc1 = nn.Sequential(                    
            nn.Linear(100, self.latent_dim_size),
            nn.LeakyReLU(),
        )

        self.fc2 = nn.Sequential(                    
            nn.Linear(100, self.latent_dim_size),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim_size,100),
            nn.LeakyReLU(),
            nn.Linear(100, 68*2),
            nn.Sigmoid(),
        )
    
    def reparametrize(self, mu, var):
        std = var.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):

        before_sample = self.encoder(x)
        mu = self.fc1(before_sample)
        var = self.fc2(before_sample)

        latent_vec = self.reparametrize(mu, var)

        x_recon = self.decoder(latent_vec)
        return x_recon, mu, var