import torch
import torch.nn as nn
from torch.autograd import Variable

# class appearance_VAE(nn.Module):
class Model(nn.Module):
    
    def __init__(self, num_filters, latent_dim_size, use_cuda=False):
        super(Model, self).__init__()
        self.use_cuda = use_cuda
        self.latent_dim_size = latent_dim_size
        self.num_filters = num_filters
        
        self.MODEL_TYPE = 'vae'
        self.MODEL_NAME = "complex_vae_{}_convtranspose".format(self.num_filters)
        self.MODEL_DATASET = 'faces'
        
        
        
        # encoder
        self.ec1 = nn.Conv2d(3, self.num_filters, kernel_size=3, stride=2, padding=1)
        self.eb1 = nn.BatchNorm2d(self.num_filters, eps=1e-05, momentum=0.1)
        self.el1 = nn.LeakyReLU(negative_slope=0.01)
        self.ed1 = nn.Dropout2d(p=0.5)
        
        self.ec2 = nn.Conv2d(self.num_filters, self.num_filters * 2, kernel_size=3, stride=2, padding=1)
        self.eb2 = nn.BatchNorm2d(self.num_filters * 2, eps=1e-05, momentum=0.1)
        self.el2 = nn.LeakyReLU(negative_slope=0.01)
        self.ed2 = nn.Dropout2d(p=0.5)

        self.ec3 = nn.Conv2d(self.num_filters * 2, self.num_filters * 4, kernel_size=3, stride=2, padding=1)
        self.eb3 = nn.BatchNorm2d(self.num_filters * 4, eps=1e-05, momentum=0.1)
        self.el3 = nn.LeakyReLU(negative_slope=0.01)
        self.ed3 = nn.Dropout2d(p=0.5)

        self.ec4 = nn.Conv2d(self.num_filters * 4, self.num_filters*8, kernel_size=3, stride=2, padding=1)
        self.eb4 = nn.BatchNorm2d(self.num_filters * 8, eps=1e-05, momentum=0.1)
        self.el4 = nn.LeakyReLU(negative_slope=0.01)
        self.ed4 = nn.Dropout2d(p=0.5)

        self.encoder = nn.Sequential(
            self.ec1,
            self.eb1,
            self.el1,
            self.ed1,
            
            self.ec2,
            self.eb2,
            self.el2,
            self.ed2,
            
            self.ec3,
            self.eb3,
            self.el3,
            self.ed3,
            
            self.ec4,
            self.eb4,
            self.el4,
            self.ed4
        )
        
        # TODO: make sure this math works
        self.fc_mean = nn.Linear(self.num_filters*8*8*8, self.latent_dim_size)
        self.fc_var = nn.Linear(self.num_filters*8*8*8, self.latent_dim_size)
        
        
        # decoder pre
        self.df_pre = nn.Linear(self.latent_dim_size, (self.num_filters*8)*(8*8))
        self.dl_pre = nn.LeakyReLU()
        
        # decoder sequential
        self.dc1 = nn.ConvTranspose2d(self.num_filters*8, self.num_filters*4, kernel_size=4, stride=2, padding=1)
        self.db1 = nn.BatchNorm2d(self.num_filters*4, eps=1e-05, momentum=0.1)
        self.dl1 = nn.LeakyReLU(negative_slope=0.01)
        self.dd1 = nn.Dropout2d(p=0.5)
        
        self.dc2 = nn.ConvTranspose2d(self.num_filters*4, self.num_filters*2, kernel_size=4, stride=2, padding=1)
        self.db2 = nn.BatchNorm2d(self.num_filters*2, eps=1e-05, momentum=0.1)
        self.dl2 = nn.LeakyReLU(negative_slope=0.01)
        self.dd2 = nn.Dropout2d(p=0.5)
        
        self.dc3 = nn.ConvTranspose2d(self.num_filters*2, self.num_filters, kernel_size=4, stride=2, padding=1)
        self.db3 = nn.BatchNorm2d(self.num_filters, eps=1e-05, momentum=0.1)
        self.dl3 = nn.LeakyReLU(negative_slope=0.01)
        self.dd3 = nn.Dropout2d(p=0.5)
        
        self.dc4 = nn.ConvTranspose2d(self.num_filters, 3, kernel_size=4, stride=2, padding=1)
        self.ds4 = nn.Sigmoid()
        
        
        self.decoder_pre = nn.Sequential(
            self.df_pre,
            self.dl_pre
        )
        
        self.decoder = nn.Sequential(
            self.dc1,
            self.db1,
            self.dl1,
            self.dd1,
                    
            self.dc2,
            self.db2,
            self.dl2,
            self.dd2,
                    
            self.dc3,
            self.db3,
            self.dl3,
            self.dd3,
                    
            self.dc4,
            self.ds4
        )
        
        
        
        # # decoder
        # self.dc1 = nn.ConvTranspose2d(self.latent_dim_size, self.num_filters*8, kernel_size=8, stride=1)
        # self.db1 = nn.BatchNorm2d(self.num_filters*8, eps=1e-05, momentum=0.1)
        # self.dl1 = nn.LeakyReLU(negative_slope=0.01)
        # self.dd1 = nn.Dropout2d(p=0.5)
        
        # self.dc2 = nn.ConvTranspose2d(self.num_filters*8, self.num_filters*4, kernel_size=4, stride=2, padding=1)
        # self.db2 = nn.BatchNorm2d(self.num_filters * 4, eps=1e-05, momentum=0.1)
        # self.dl2 = nn.LeakyReLU(negative_slope=0.01)
        # self.dd2 = nn.Dropout2d(p=0.5)
        
        # self.dc3 = nn.ConvTranspose2d(self.num_filters*4, self.num_filters*2, kernel_size=4, stride=2, padding=1)
        # self.db3 = nn.BatchNorm2d(self.num_filters * 2, eps=1e-05, momentum=0.1)
        # self.dl3 = nn.LeakyReLU(negative_slope=0.01)
        # self.dd3 = nn.Dropout2d(p=0.5)
        
        # self.dc4 = nn.ConvTranspose2d(self.num_filters*2, self.num_filters, kernel_size=4, stride=2, padding=1)
        # self.db4 = nn.BatchNorm2d(self.num_filters, eps=1e-05, momentum=0.1)
        # self.dl4 = nn.LeakyReLU(negative_slope=0.01)
        # self.dd4 = nn.Dropout2d(p=0.5)

        # nn.ConvTranspose2d(self.num_filters, 3, kernel_size=4,  stride=2, padding=1)
        # nn.Sigmoid()
        
        
        
        
        # self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(self.latent_dim_size, self.num_filters*8, kernel_size=8, stride=1),
            # nn.BatchNorm2d(self.num_filters*8, eps=1e-05, momentum=0.1),
            # nn.LeakyReLU(negative_slope=0.01),
            # nn.Dropout2d(p=0.5),

            # nn.ConvTranspose2d(self.num_filters*8, self.num_filters*4, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(self.num_filters * 4, eps=1e-05, momentum=0.1),
            # nn.LeakyReLU(negative_slope=0.01),
            # nn.Dropout2d(p=0.5),

            # nn.ConvTranspose2d(self.num_filters*4, self.num_filters*2, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(self.num_filters * 2, eps=1e-05, momentum=0.1),
            # nn.LeakyReLU(negative_slope=0.01),
            # nn.Dropout2d(p=0.5),

            # nn.ConvTranspose2d(self.num_filters*2, self.num_filters, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(self.num_filters, eps=1e-05, momentum=0.1),
            # nn.LeakyReLU(negative_slope=0.01),
            # nn.Dropout2d(p=0.5),

            # nn.ConvTranspose2d(self.num_filters, 3, kernel_size=4,  stride=2, padding=1),
            # nn.Sigmoid(),
        # )

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
        
        mu = self.fc_mean(before_sample.view(-1, self.num_filters*8*8*8))
        var = self.fc_var(before_sample.view(-1, self.num_filters*8*8*8))
        
        latent_vec = self.reparametrize(mu, var)
        return mu, var, latent_vec
    
    def get_recon_from_latent_vec(self, latent_vec):
        x_recon_pre = self.decoder_pre(latent_vec)
        x_recon = self.decoder(x_recon_pre.view(-1, self.num_filters*8, 8, 8))
        # x_recon = self.decoder(latent_vec.view(-1, self.latent_dim_size, 1, 1))
        return x_recon
        
    def forward(self, x):
        mu, var, latent_vec = self.get_latent_vec(x)
        x_recon = self.get_recon_from_latent_vec(latent_vec)
        return x_recon, mu, var
        