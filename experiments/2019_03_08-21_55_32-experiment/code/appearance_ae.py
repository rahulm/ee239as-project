import torch
import torch.nn as nn

class appearance_autoencoder(nn.Module):
    def __init__(self, latent_dim_size):
        super(appearance_autoencoder, self).__init__()

        self.latent_dim_size = latent_dim_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        self.fc1 = nn.Sequential(                    
            nn.Linear(128*8*8, self.latent_dim_size),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim_size, 128, kernel_size=8, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4,  stride=2, padding=1),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.encoder(x)
        z = self.fc1(x.view(-1, 128*8*8))
        x_recon = self.decoder(z.view(-1, self.latent_dim_size, 1, 1))
        return x_recon
        