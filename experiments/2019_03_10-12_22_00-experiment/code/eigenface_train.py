import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import skimage
from skimage import io, transform
import scipy.io as sio
from scipy.signal import argrelextrema
import glob
import shutil

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


# some experiment logging setup
ALL_EXPERIMENTS_DIR = "experiments"
EXP_LOSS_PLOTS_DIR = "loss_plots"
EXP_METRICS_DIR = "metrics"
EXP_MODELS_DIR = "models"
EXP_CODE_DIR = "code"

class ExperimentConfig:
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.exp_dir = os.path.join(ALL_EXPERIMENTS_DIR, self.exp_name)
        self.exp_loss_plots_dir = os.path.join(self.exp_dir, EXP_LOSS_PLOTS_DIR)
        self.exp_metrics_dir = os.path.join(self.exp_dir, EXP_METRICS_DIR)
        self.exp_models_dir = os.path.join(self.exp_dir, EXP_MODELS_DIR)
        self.exp_code_dir = os.path.join(self.exp_dir, EXP_CODE_DIR)
        
        self.dirs = [self.exp_dir, self.exp_loss_plots_dir, self.exp_metrics_dir, self.exp_models_dir, self.exp_code_dir]
        for dir in self.dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)
        
        self.str_repr = "ExperimentConfig for experiment {}\n".format(self.exp_name)
        for dir in self.dirs:
            self.str_repr += ("\t" + dir + "\n")
        self.str_repr += "\n"
        
        # copy all code into backup folder
        for pyfile in glob.glob('*.py'):
            shutil.copy(pyfile, self.exp_code_dir)
    
    def __str__(self):
        return self.str_repr
        
def setup_custom_logging(exp_name=""):
    import datetime
    import sys
    
    curr_date_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    
    # make current experiment directory
    curr_exp_dt_name = "{}-experiment{}".format(curr_date_time, ("-"+exp_name) if (exp_name != "") else "")
    exp_config = ExperimentConfig(curr_exp_dt_name)
    
    curr_exp_log = os.path.join(exp_config.exp_dir, "log.txt")
    outfile = open(curr_exp_log, 'w')
    
    class CustomLogging:
        def __init__(self, orig_stream):
            self.orig_stream = orig_stream
            self.fileout = outfile
        def write(self, data):
            self.orig_stream.write(data)
            self.orig_stream.flush()
            self.fileout.write(data)
            self.fileout.flush()
        def flush(self):
            self.orig_stream.flush()
            self.fileout.flush()
    
    sys.stdout = CustomLogging(sys.stdout)
    
    return exp_config

def get_args(print_args=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--seed', type=int, default=12345,
        help="random seed")
    parser.add_argument('--use_cuda', action='store_true', default=False,
        help="attempts to enable cuda training, if cuda available")
    parser.add_argument('--device', type=int, default=0,
        help="Device to use for cuda, only applicable if cuda is available and --use_cuda is set.")
    
    parser.add_argument('--image_dir', type=str, default='images',
        help="location of eigenface images")
    parser.add_argument('--landmark_dir', type=str, default='landmarks',
        help="location of eigenface landmarks")
    # parser.add_argument('--cache_dir', type=str, default='cache')
    
    parser.add_argument('--appear_lr', type=float, default=7e-4,
        help="learning rate of appearance model.")
    parser.add_argument('--landmark_lr', type=float, default=1e-4,
        help="learning rate of landmark model.")
    parser.add_argument('--appear_latent_dim', type=int, default=50,
        help="number of elements in the latent vector for the appearance model")
    parser.add_argument('--landmark_latent_dim', type=int, default=10,
        help="number of elements in the latent vector for the landmark model")
    
    parser.add_argument('--epochs', type=int, default=70,
        help="number of epochs to train both models")
    parser.add_argument('--batch_size', type=int, default=32,
        help="batch size to use in training of both models")
    
    required_group = parser.add_argument_group('required arguments:')
    required_group.add_argument('--model', type=str, required=True, choices=('ae', 'vae'),
        help="type of model to train, choose from 'ae' or 'vae'")
    required_group.add_argument('--faces', type=str, required=True, choices=('aligned', 'unaligned'),
        help="type of faces data to train on, choose from 'aligned' or 'unaligned'")
    
    args = parser.parse_args()
    
    if args.use_cuda and not torch.cuda.is_available():
        args.use_cuda = False
        print("args.use_cuda set to False because cuda is not available")
    
    if print_args:
        print("Arguments:")
        print(args)
        print('\n')
    
    return args


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
                    # data = io.imread(self.root_dir + name + self.file_format)
                    data = io.imread(os.path.join(self.root_dir, name + self.file_format))
                elif read_type == 'landmark':
                    # mat_data = sio.loadmat(self.root_dir + name + self.file_format)
                    mat_data = sio.loadmat(os.path.join(self.root_dir, name + self.file_format))

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



def train_ae_appearance_model(exp_config, latent_dim, learning_rate, num_epochs, batch_size, cuda_avail, loss_function, face_images_train_warped):
    face_train_split = face_images_train_warped[:-100]
    face_val_split = face_images_train_warped[-100:]
    # face_trainset = dataset_constructor(face_images_train_warped, transform=transforms.Compose([ImgToTensor()]))
    face_trainset = dataset_constructor(face_train_split, transform=transforms.Compose([ImgToTensor()]))
    face_valset = dataset_constructor(face_val_split, transform=transforms.Compose([ImgToTensor()]))

    face_trainloader = torch.utils.data.DataLoader(face_trainset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=2)
    
    face_valloader = torch.utils.data.DataLoader(face_valset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=2)

    # app_model = appearance_autoencoder(latent_dim_size=50)
    app_model = appearance_autoencoder(latent_dim_size=latent_dim)
    optimizer = optim.Adam(app_model.parameters(), lr=learning_rate)
    trainer = ae_trainer(optimizer=optimizer,
                            use_cuda=cuda_avail,
                            model=app_model, 
                            loss_func=loss_function, 
                            model_name="Appearance-AE", exp_config=exp_config)
    
    trainer.train_model(num_epochs, face_trainloader, face_valloader)

def train_ae_landmark_model(exp_config, latent_dim, learning_rate, num_epochs, batch_size, cuda_avail, loss_function, landmark_train):
    landmark_train_split = landmark_train[:-100]
    landmark_val_split = landmark_train[-100:]
    # landmark_trainset = dataset_constructor(landmark_train, transform=transforms.Compose([LandmarkToTensor()]))
    landmark_trainset = dataset_constructor(landmark_train_split, transform=transforms.Compose([LandmarkToTensor()]))
    landmark_valset = dataset_constructor(landmark_val_split, transform=transforms.Compose([LandmarkToTensor()]))

    landmark_trainloader = torch.utils.data.DataLoader(landmark_trainset, 
                                                        batch_size=batch_size, 
                                                        shuffle=False, 
                                                        num_workers=2)
    
    landmark_valloader = torch.utils.data.DataLoader(landmark_valset, 
                                                        batch_size=batch_size, 
                                                        shuffle=False, 
                                                        num_workers=2)

    # lm_model = landmark_autoencoder(latent_dim_size=10)
    lm_model = landmark_autoencoder(latent_dim_size=latent_dim)
    optimizer = optim.Adam(lm_model.parameters(), lr=learning_rate)
    trainer = ae_trainer(optimizer=optimizer,
                            use_cuda=cuda_avail,
                            model=lm_model, 
                            loss_func=loss_function, 
                            model_name="Landmark-AE", exp_config=exp_config)

    trainer.train_model(num_epochs, landmark_trainloader, landmark_valloader)

def train_vae_appearance_model(exp_config, latent_dim, learning_rate, num_epochs, batch_size, cuda_avail, loss_function, face_images_train_warped):
    face_train_split = face_images_train_warped[:-100]
    face_val_split = face_images_train_warped[-100:]
    # face_trainset = dataset_constructor(face_images_train_warped, transform=transforms.Compose([ImgToTensor()]))
    face_trainset = dataset_constructor(face_train_split, transform=transforms.Compose([ImgToTensor()]))
    face_valset = dataset_constructor(face_val_split, transform=transforms.Compose([ImgToTensor()]))

    face_trainloader = torch.utils.data.DataLoader(face_trainset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=2)
    
    face_valloader = torch.utils.data.DataLoader(face_valset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=2)

    # app_model = appearance_VAE(latent_dim_size=50, use_cuda=cuda_avail)
    app_model = appearance_VAE(latent_dim_size=latent_dim, use_cuda=cuda_avail)
    optimizer = optim.Adam(app_model.parameters(), lr=learning_rate)
    trainer = vae_trainer(optimizer=optimizer,
                            use_cuda=cuda_avail,
                            model=app_model, 
                            recon_loss_func=loss_function,
                            model_name="Appearance-VAE", exp_config=exp_config)
    
    trainer.train_model(num_epochs, face_trainloader, face_valloader)

def train_vae_landmark_model(exp_config, latent_dim, learning_rate, num_epochs, batch_size, cuda_avail, loss_function, landmark_train):
    landmark_train_split = landmark_train[:-100]
    landmark_val_split = landmark_train[-100:]
    # landmark_trainset = dataset_constructor(landmark_train, transform=transforms.Compose([LandmarkToTensor()]))
    landmark_trainset = dataset_constructor(landmark_train_split, transform=transforms.Compose([LandmarkToTensor()]))
    landmark_valset = dataset_constructor(landmark_val_split, transform=transforms.Compose([LandmarkToTensor()]))

    landmark_trainloader = torch.utils.data.DataLoader(landmark_trainset, 
                                                        batch_size=batch_size, 
                                                        shuffle=False, 
                                                        num_workers=2)
    
    landmark_valloader = torch.utils.data.DataLoader(landmark_valset, 
                                                        batch_size=batch_size, 
                                                        shuffle=False, 
                                                        num_workers=2)

    # lm_model = landmark_VAE(latent_dim_size=10, use_cuda=cuda_avail)
    lm_model = landmark_VAE(latent_dim_size=latent_dim, use_cuda=cuda_avail)
    optimizer = optim.Adam(lm_model.parameters(), lr=learning_rate)
    trainer = vae_trainer(optimizer=optimizer,
                            use_cuda=cuda_avail,
                            model=lm_model, 
                            recon_loss_func=loss_function,
                            model_name="Landmark-VAE", exp_config=exp_config)

    trainer.train_model(num_epochs, landmark_trainloader, landmark_valloader)


if __name__ == '__main__':
    args = get_args(print_args=True)
    
    exp_config = setup_custom_logging()
    print("args\n{}\n".format(args))
    print("ExperimentConfig\n{}\n".format(exp_config))    
    
    if args.use_cuda:
        torch.cuda.set_device(args.device)
        print('Setting torch.cuda.set_device({})'.format(args.device))
        torch.cuda.manual_seed(args.seed)
        print('Setting torch.cuda.manual_seed({})\n'.format(args.seed))
    
    # set the model type and loss functions
    app_loss_func, landmark_loss_func = None, None
    app_train_func, landmark_train_func = None, None
    if args.model == 'ae':
        app_loss_func, landmark_loss_func = nn.MSELoss(), nn.MSELoss()
        app_train_func = train_ae_appearance_model
        landmark_train_func = train_ae_landmark_model
    elif args.model == 'vae':
        app_loss_func, landmark_loss_func = nn.BCELoss(reduction='sum'), nn.BCELoss(reduction='sum')
        app_train_func = train_vae_appearance_model
        landmark_train_func = train_vae_landmark_model
    
    # choose appropriate data to read from for faces
    if args.faces == 'aligned':
        faces_data_loc = 'all-warped-images.npy'
    elif args.faces == 'unaligned':
        faces_data_loc = 'all-raw-images.npy'
    
    # read the appropriate data, make train/test split
    all_face_images_warped = np.load(faces_data_loc)
    face_images_train_warped = all_face_images_warped[:-100]
    face_images_test_warped = all_face_images_warped[-100:]
    print("Read cached images from {}".format(faces_data_loc))
    
    # always train the appearance model for both aligned and unaligned face options
    app_train_func(exp_config, args.appear_latent_dim, args.appear_lr, args.epochs, args.batch_size, args.use_cuda, app_loss_func, face_images_train_warped)
    
    # only train the landmark model if we are training the aligned version
    if args.faces == 'aligned':
        face_landmark_reader = data_reader(args.landmark_dir, 6, '000000', '.mat')
        face_landmark_train, face_landmark_test = face_landmark_reader.read(split=800, read_type='landmark')
        print("read landmarks")

        face_landmark_train = np.asarray(face_landmark_train)
        face_landmark_test = np.asarray(face_landmark_test)
        
        landmark_train_func(exp_config, args.landmark_latent_dim, args.appear_lr, args.epochs, args.batch_size, args.use_cuda, landmark_loss_func, face_landmark_train)
    
    # face_images_reader = data_reader(args.image_dir, 6, '000000', '.jpg')
    # face_images_train, face_images_test = face_images_reader.read(split=800, read_type='image')
    # print("read images")

    # face_images_train = np.asarray(face_images_train)
    # face_images_test = np.asarray(face_images_test)
    
    
    #   Train Autoencoders
    # train_ae_appearance_model(exp_config, args.appear_lr, args.epochs, args.batch_size, args.use_cuda, nn.MSELoss(), face_images_train_warped)
    # train_ae_landmark_model(exp_config, args.landmark_lr, args.epochs, args.batch_size, args.use_cuda, nn.MSELoss(), face_landmark_train)

    #   Train Variational Autoencoders
    # loss_func = nn.BCELoss()
    # loss_func.size_average = False
    # train_vae_appearance_model(exp_config, args.appear_lr, args.epochs, args.batch_size, args.use_cuda, loss_func, face_images_train_warped)
    # train_vae_landmark_model(exp_config, args.landmark_lr, args.epochs, args.batch_size, args.use_cuda, loss_func, face_landmark_train)

