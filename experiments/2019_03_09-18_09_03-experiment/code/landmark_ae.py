import torch
import torch.nn as nn

class landmark_autoencoder(nn.Module):
    def __init__(self, latent_dim_size):
        super(landmark_autoencoder, self).__init__()

        self.latent_dim_size = latent_dim_size

        self.encoder = nn.Sequential(
            nn.Linear(68*2, 100),
            nn.LeakyReLU(),
            nn.Linear(100, self.latent_dim_size),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear( self.latent_dim_size,100),
            nn.LeakyReLU(),
            nn.Linear(100, 68*2),
            nn.Sigmoid(),
        )
    
    def forward_to_encoder(self, x):
        return self.encoder(x)
    
    def forward_to_decoder(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon