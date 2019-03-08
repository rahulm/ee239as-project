import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import skimage
from skimage import io, transform
import scipy.io as sio
from scipy.signal import argrelextrema

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from mywarper import warp, plot

from model_trainer import ae_trainer, vae_trainer

from appearance_ae import appearance_autoencoder
from landmark_ae import landmark_autoencoder
from landmark_vae import landmark_VAE
from appearance_vae import appearance_VAE

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=70)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--image_dir', type=str, default='./images/')
parser.add_argument('--landmark_dir', type=str, default='./landmarks/')
parser.add_argument('--male_img_dir', type=str, default='./male_images/')
parser.add_argument('--female_img_dir', type=str, default='./female_images/')
parser.add_argument('--male_landmark', type=str, default='./male_landmarks/')
parser.add_argument('--female_landmark', type=str, default='./female_landmarks/')
parser.add_argument('--path', type=str, default='./results/model/')
parser.add_argument('--log', type=str, default='./results/log/')
parser.add_argument('--appear_lr', type=float, default=7e-4)
parser.add_argument('--landmark_lr', type=float, default=1e-4)

# Read Dataset
class data_reader(object):
    def __init__(self, root_dir, file_str_len, origin_name, file_format):
        self.root_dir = root_dir
        self.file_str_len = file_str_len
        self.origin_name = origin_name
        self.file_format = file_format

    def read(self, split, read_type):
        files_len = len([name for name in os.listdir(self.root_dir) 
                        if os.path.isfile(os.path.join(self.root_dir, name))])
        counter = 0
        idx = counter
        dataset = []
        train_dataset = []
        test_dataset = []
        while counter < files_len:
            name = self.origin_name + str(idx)
            if len(name) > self.file_str_len:
                name = name[len(name)-self.file_str_len:]
            try:
                if read_type == 'image':
                    data = io.imread(self.root_dir + name + self.file_format)
                elif read_type == 'landmark':
                    mat_data = sio.loadmat(self.root_dir + name + self.file_format)

                    data = mat_data['lms']
                dataset.append(data)
                counter += 1
            except FileNotFoundError:
                pass
            idx += 1
        train_dataset = dataset[:split]
        test_dataset = dataset[split:]
        return train_dataset, test_dataset

# Construct Dataset
class ImgToTensor(object):
    def __call__(self, sample):
        sample = sample.transpose((2, 0, 1))
        return torch.tensor(sample, dtype=torch.float32)/255

class LandmarkToTensor(object):
    def __call__(self, sample):
        sample = sample.reshape(-1)
        return torch.tensor(sample, dtype=torch.float32)/128

class dataset_constructor(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample_data = self.dataset[idx]
        if self.transform:
            sample_data = self.transform(sample_data)
        return sample_data

def vae_loss(x, x_recon, mu, var, recon_loss_func):
    recon_loss = recon_loss_func(x_recon, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(var.exp()).mul_(-1).add_(1).add_(var)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return recon_loss + KLD

def train_ae_appearance_model(learning_rate, num_epochs, batch_size, cuda_avail, face_images_train_warped):
    face_trainset = dataset_constructor(face_images_train_warped, transform=transforms.Compose([ImgToTensor()]))

    face_trainloader = torch.utils.data.DataLoader(face_trainset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=2)

    app_model = appearance_autoencoder(latent_dim_size=50)
    optimizer = optim.Adam(app_model.parameters(), lr=learning_rate)
    trainer = ae_trainer(optimizer=optimizer,
                            use_cuda=cuda_avail,
                            model=app_model, 
                            loss_func=nn.MSELoss(), 
                            model_name="Appearance AE")
    
    trainer.train_model(num_epochs, face_trainloader)

def train_ae_landmark_model(learning_rate, num_epochs, batch_size, cuda_avail, landmark_train):

    landmark_trainset = dataset_constructor(landmark_train, transform=transforms.Compose([LandmarkToTensor()]))

    landmark_trainloader = torch.utils.data.DataLoader(landmark_trainset, 
                                                        batch_size=batch_size, 
                                                        shuffle=False, 
                                                        num_workers=2)

    lm_model = landmark_autoencoder(latent_dim_size=10)
    optimizer = optim.Adam(lm_model.parameters(), lr=learning_rate)
    trainer = ae_trainer(optimizer=optimizer,
                            use_cuda=cuda_avail,
                            model=lm_model, 
                            loss_func=nn.MSELoss(), 
                            model_name="Landmark AE")

    trainer.train_model(num_epochs, landmark_trainloader)

def train_vae_appearance_model(learning_rate, num_epochs, batch_size, cuda_avail, face_images_train_warped):
    face_trainset = dataset_constructor(face_images_train_warped, transform=transforms.Compose([ImgToTensor()]))

    face_trainloader = torch.utils.data.DataLoader(face_trainset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=2)

    app_model = appearance_autoencoder(latent_dim_size=50)
    optimizer = optim.Adam(app_model.parameters(), lr=learning_rate)
    trainer = vae_trainer(optimizer=optimizer,
                            use_cuda=cuda_avail,
                            model=app_model, 
                            loss_func=vae_loss,
                            recon_loss_func=nn.BCELoss(),
                            model_name="Appearance VAE")
    
    trainer.train_model(num_epochs, face_trainloader)

def train_vae_landmark_model(learning_rate, num_epochs, batch_size, cuda_avail, landmark_train):
    landmark_trainset = dataset_constructor(landmark_train, transform=transforms.Compose([LandmarkToTensor()]))

    landmark_trainloader = torch.utils.data.DataLoader(landmark_trainset, 
                                                        batch_size=batch_size, 
                                                        shuffle=False, 
                                                        num_workers=2)

    lm_model = landmark_autoencoder(latent_dim_size=10)
    optimizer = optim.Adam(lm_model.parameters(), lr=learning_rate)
    trainer = vae_trainer(optimizer=optimizer,
                            use_cuda=cuda_avail,
                            model=lm_model, 
                            loss_func=vae_loss, 
                            recon_loss_func=nn.BCELoss(),
                            model_name="Landmark VAE")

    trainer.train_model(num_epochs, landmark_trainloader)


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    
    # if args.cuda:
    #     torch.cuda.set_device(args.device)

    if not os.path.exists('./saved_weights'):
        os.makedirs('./saved_weights')

    if not os.path.exists('./train_plots'):
        os.makedirs('./train_plots')

    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)

    face_images_reader = data_reader(args.image_dir, 6, '000000', '.jpg')
    face_images_train, face_images_test = face_images_reader.read(split=800, read_type='image')

    face_images_train = np.asarray(face_images_train)
    face_images_test = np.asarray(face_images_test)

    face_landmark_reader = data_reader(args.landmark_dir, 6, '000000', '.mat')
    face_landmark_train, face_landmark_test = face_landmark_reader.read(split=800, read_type='landmark')

    face_landmark_train = np.asarray(face_landmark_train)
    face_landmark_test = np.asarray(face_landmark_test)


    face_images_train_warped = None
    face_images_test_warped = None

    if not os.path.exists('./warped-train-images.npy') or not os.path.exists('./warped-test-images.npy'):

        mean_train_landmark = np.mean(face_landmark_train, axis=0)
        mean_test_landmark = np.mean(face_landmark_test, axis=0)
        print("Warping training images")
        face_images_train_warped = np.copy(face_images_train)
        for i in range(len(face_images_train)):
            face_images_train_warped[i] = warp(np.copy(face_images_train_warped[i]), face_landmark_train[i], mean_train_landmark)
        np.save("warped-train-images.npy", face_images_train_warped)

        print("Warping testing images")
        face_images_test_warped = np.copy(face_images_test)
        for i in range(len(face_images_test)):
            face_images_test_warped[i] = warp(np.copy(face_images_test_warped[i]), face_landmark_test[i], mean_train_landmark)
        np.save("warped-test-images.npy", face_images_test_warped)
    else:
        face_images_train_warped = np.load("./warped-train-images.npy")
        face_images_test_warped = np.load("./warped-test-images.npy")


    #   Train Autoencoders
    # train_ae_appearance_model(args.appear_lr, args.epochs, args.batch_size, args.cuda, face_images_train_warped)
    # train_ae_landmark_model(args.landmark_lr, args.epochs, args.batch_size, args.cuda, face_landmark_train)

    #   Train Variational Autoencoders
    train_vae_appearance_model(args.appear_lr, args.epochs, args.batch_size, args.cuda, face_images_train_warped)
    train_vae_landmark_model(args.landmark_lr, args.epochs, args.batch_size, args.cuda, face_landmark_train)

