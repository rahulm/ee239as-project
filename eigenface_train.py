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

import importlib
import csv


# some experiment logging setup
ALL_EXPERIMENTS_DIR = "experiments"
EXP_LOSS_PLOTS_DIR = "loss_plots"
EXP_METRICS_DIR = "metrics"
# EXP_MODELS_DIR = "models"
EXP_MODELS_DIR = "weights"
EXP_CODE_DIR = "code"
EXP_MODEL_ARCHITECTURES_DIR = "models"
EXP_RECONSTRUCTION_DIR = "reconstructions"
EXP_GENERATION_DIR = "generations"
EXP_NEARESTNEIGHBOR_DIR = "nearestneighbor"

class ExperimentConfig:
    def __init__(self, exp_name, exp_datetime=None):
        self.exp_name = exp_name
        self.exp_dir = os.path.join(ALL_EXPERIMENTS_DIR, self.exp_name)
        
        self.exp_loss_plots_dir = os.path.join(self.exp_dir, EXP_LOSS_PLOTS_DIR)
        self.exp_metrics_dir = os.path.join(self.exp_dir, EXP_METRICS_DIR)
        self.exp_models_dir = os.path.join(self.exp_dir, EXP_MODELS_DIR)
        
        self.exp_code_dir = os.path.join(self.exp_dir, EXP_CODE_DIR)
        self.exp_model_architectures_dir = os.path.join(self.exp_code_dir, EXP_MODEL_ARCHITECTURES_DIR)
        self.exp_reconstruction_dir = os.path.join(self.exp_dir, EXP_RECONSTRUCTION_DIR)
        self.exp_generation_dir = os.path.join(self.exp_dir, EXP_GENERATION_DIR)
        self.exp_nn_dir = os.path.join(self.exp_dir, EXP_NEARESTNEIGHBOR_DIR)
        
        self.exp_datetime = exp_datetime
        
        self.dirs = [self.exp_dir, self.exp_loss_plots_dir, self.exp_metrics_dir, self.exp_models_dir, self.exp_code_dir, \
            self.exp_model_architectures_dir, self.exp_reconstruction_dir, self.exp_generation_dir, self.exp_nn_dir]
            
        for dir in self.dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)
        
        self.str_repr = "ExperimentConfig for experiment {}\n".format(self.exp_name)
        # 2016 April 05 13:09:15 format
        self.str_repr += (self.exp_datetime.strftime("%Y %B %d %H:%M:%S") + "\n")
        for dir in self.dirs:
            self.str_repr += ("\t" + dir + "\n")
        self.str_repr += "\n"
        
        # copy all code into backup folder
        for pyfile in glob.glob('*.py'):
            shutil.copy(pyfile, self.exp_code_dir)
        for pyfile in glob.glob(os.path.join(EXP_MODEL_ARCHITECTURES_DIR, '*.py')):
            shutil.copy(pyfile, self.exp_model_architectures_dir)
    
    def __str__(self):
        return self.str_repr
        
def setup_custom_logging(exp_name=""):
    import datetime
    import sys
    
    curr_datetime = datetime.datetime.now()
    curr_datetime_str = curr_datetime.strftime("%Y_%m_%d-%H_%M_%S")
    
    # make current experiment directory
    # curr_exp_dt_name = "{}-experiment{}".format(curr_datetime_str, ("-"+exp_name) if (exp_name != "") else "")
    curr_exp_dt_name = "{}{}".format(curr_datetime_str, ('-' + exp_name) if (exp_name != "") else "")
    exp_config = ExperimentConfig(curr_exp_dt_name, exp_datetime=curr_datetime)
    
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

def str_to_bool(s):
    sv = s.lower()
    if sv in ('yes', 'true', 't', 'y', '1'):
        return True
    elif sv in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        print("argument {} not recognized as a boolean.".format(s))
        exit(1)

def get_args(print_args=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--seed', type=int, default=12345,
        help="random seed")
    # parser.add_argument('--use_cuda', action='store_true', default=False,
        # help="attempts to enable cuda training, if cuda available")
    parser.add_argument('--use_cuda', type=str_to_bool, default=False, nargs='?', const=True,
        help="whether or not to train with cuda, if cuda available")
    parser.add_argument('--device', type=int, default=0,
        help="Device to use for cuda, only applicable if cuda is available and --use_cuda is set.")
    
    # parser.add_argument('--image_dir', type=str, default='images',
        # help="location of eigenface images")
    parser.add_argument('--landmark_dir', type=str, default='landmarks',
        help="location of eigenface landmarks")
    parser.add_argument('--exp_name', type=str, default="",
        help="optional experiment name")
        
    
    parser.add_argument('--recon_gen_interval', type=int, default=None,
        help="epoch interval to reconstruct sample test images and generate images if applicable")
    parser.add_argument('--num_recon', type=int, default=5,
        help="number of test images to reconstruct after each interval")
    parser.add_argument('--num_gen', type=int, default=5,
        help="number of test images to generate after each interval, ONLY IF VAE")
    
    parser.add_argument('--checkpoint_interval', type=int, default=1,
        help="specifies the interval (in number of epochs) to checkpoint the weighs at.")
    
    # parser.add_argument('--latent_vec_reg', type=str_to_bool, default=False, nargs='?', const=True,
        # help="whether or not to perform l1 regularization on latent vector")
    parser.add_argument('--latent_vec_reg', type=float, default=0,
        help="coefficient to apply to latent vector l1 regularization, set to 0 for no regularization.")
    parser.add_argument('--results_csv', type=str, default="./results.csv",
        help="path to save results of experiment in csv format. must include name of csv at end of path.")
        
    
    required_group = parser.add_argument_group('required arguments:')
    # required_group.add_argument('--model', type=str, required=True, choices=('ae', 'vae'),
        # help="type of model to train, choose from 'ae' or 'vae'")
    required_group.add_argument('--loss_func', type=str, choices=('BCELoss', 'MSELoss'),
        help="type of loss function to use with the model")
    required_group.add_argument('--optimizer', type=str, required=True, choices=('Adam', 'RMSprop'),
        help="optimizer")
    required_group.add_argument('--latent_dim', type=int, required=True, # ex: 10 and 50
        help="number of elements in the latent vector for the model")
    required_group.add_argument('--lr', type=float, required=True,
        help="learning rate")
    required_group.add_argument('--epochs', type=int, required=True, # 70
        help="number of epochs to train model")
    required_group.add_argument('--batch_size', type=int, required=True, # 32
        help="batch size to use in training of model")
    
        
    # Ex: use ae231.face_model or something
    required_group.add_argument('--model', type=str, required=True,
        help="name (folder.file) of file in which 'Model' class exists")
    
    
    
    # required_group.add_argument('--dataset', type=str, required=True, choices=('faces', 'landmarks'),
        # help="type of dataset to train on")
    
    parser.add_argument('--faces', type=str, choices=('aligned', 'unaligned'), default=None,
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
    
def train_model(exp_config,
    model,
    optimizer,
    num_epochs,
    batch_size,
    loss_function,
    latent_vec_reg,
    train_dataset,
    test_dataset,
    dataset_transform,
    checkpoint_interval,
    recon_gen_interval,
    num_recon,
    num_gen,
    use_cuda,
    shuffle=False):
    
    train_split = train_dataset[:-100]
    val_split = train_dataset[-100:]
    

    trainset = dataset_constructor(train_split, transform=transforms.Compose([dataset_transform()]))
    valset = dataset_constructor(val_split, transform=transforms.Compose([dataset_transform()]))
    testset = dataset_constructor(test_dataset, transform=transforms.Compose([dataset_transform()]))
    
    trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=2)
    
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=2)
    
    testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=2)
    
    if model.MODEL_TYPE == 'vae':
        trainer_class = vae_trainer
    elif model.MODEL_TYPE == 'ae':
        trainer_class = ae_trainer
    else:
        print("MODEL_TYPE {} not recognized. only use 'ae' or 'vae'.".format(model.MODEL_TYPE))
        exit(1)
    
    trainer = trainer_class(optimizer=optimizer,
                            use_cuda=use_cuda,
                            model=model,
                            loss_func=loss_function,
                            latent_vec_reg=latent_vec_reg,
                            model_name=exp_config.exp_name,
                            exp_config=exp_config)
    
    if num_recon > len(test_dataset):
        print("Requested for {} reconstructions, but there are only {} samples in the test set".format(num_recon, len(test_dataset)))
        exit(1)

    sample_test_tensors = []
    # orig_test = test_dataset[0:num_recon]
    offset = 59
    orig_test = test_dataset[offset:offset+num_recon]
    for i in range(num_recon):
        sample_test_tensors.append(dataset_transform()(test_dataset[i]))
    sample_test_tensors = torch.stack(sample_test_tensors)
    sample_test_tensors = sample_test_tensors.to('cuda:0' if use_cuda else 'cpu')
    
    all_samples = np.concatenate((train_dataset, test_dataset), axis=0)


    final_train_loss, final_val_loss, final_test_loss = trainer.train_model(
                                                                epochs=num_epochs,
                                                                trainloader=trainloader,
                                                                valloader=valloader,
                                                                testloader=testloader,
                                                                checkpoint_interval=checkpoint_interval,
                                                                test_samples=orig_test,
                                                                test_tensors=sample_test_tensors,
                                                                all_samples=all_samples,
                                                                recon_gen_interval=recon_gen_interval,
                                                                num_gen=num_gen)

    return final_train_loss, final_val_loss, final_test_loss
if __name__ == '__main__':
    args = get_args(print_args=True)
    
    exp_config = setup_custom_logging(args.exp_name)
    print("args\n{}\n".format(args))
    print("ExperimentConfig\n{}\n".format(exp_config))    
    
    if args.use_cuda:
        torch.cuda.set_device(args.device)
        print('Setting torch.cuda.set_device({})'.format(args.device))
        torch.cuda.manual_seed(args.seed)
        print('Setting torch.cuda.manual_seed({})\n'.format(args.seed))
    
    # setup model
    model_pkg = importlib.import_module(args.model)
    model = model_pkg.Model(latent_dim_size=args.latent_dim, use_cuda=args.use_cuda)
    
    # loss_reduction = 'sum' if (model.MODEL_TYPE == 'vae') else 'mean'
    # loss_reduction = 'sum' if (model.MODEL_TYPE == 'vae') else 'none'
    loss_function_class = getattr(nn, args.loss_func)
    if model.MODEL_TYPE == 'vae':
        loss_function = loss_function_class(reduction='sum')
    elif model.MODEL_TYPE == 'ae':
        loss_function = loss_function_class()
    
    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr)
    
    train_dataset, test_dataset = None, None
    if model.MODEL_DATASET == 'faces':
        # choose appropriate data to read from for faces
        if args.faces == 'aligned':
            faces_data_loc = 'all-warped-images.npy'
        elif args.faces == 'unaligned':
            faces_data_loc = 'all-raw-images.npy'
        else:
            print("args.faces {} not recognized. use 'aligned' or 'unaligned' for appearance model".format(args.faces))
            exit(1)
        
        # read the appropriate data, make train/test split
        all_face_images = np.load(faces_data_loc)
        train_dataset = all_face_images[:-100]
        test_dataset = all_face_images[-100:]
        print("Read cached images from {}".format(faces_data_loc))
        
        data_transform = ImgToTensor
    elif model.MODEL_DATASET == 'landmarks':
        face_landmark_reader = data_reader(args.landmark_dir, 6, '000000', '.mat')
        face_landmark_train, face_landmark_test = face_landmark_reader.read(split=900, read_type='landmark')
        print("read landmarks from {}".format(args.landmark_dir))
        
        train_dataset = np.asarray(face_landmark_train)
        test_dataset = np.asarray(face_landmark_test)
        
        data_transform = LandmarkToTensor
    else:
        print("MODEL_DATASET {} not recognized. only use 'faces' or 'landmarks'".format(model.MODEL_DATASET))
        exit(1)
    
    recon_gen_interval = args.recon_gen_interval
    # train model
    final_train_loss, final_val_loss, final_test_loss = train_model(
                                                            exp_config=exp_config,
                                                            model=model,
                                                            optimizer=optimizer,
                                                            num_epochs=args.epochs,
                                                            batch_size=args.batch_size,
                                                            loss_function=loss_function,
                                                            latent_vec_reg=args.latent_vec_reg,
                                                            train_dataset=train_dataset,
                                                            test_dataset=test_dataset,
                                                            dataset_transform=data_transform,
                                                            checkpoint_interval=args.checkpoint_interval,
                                                            recon_gen_interval=recon_gen_interval,
                                                            num_recon=args.num_recon,
                                                            num_gen=args.num_gen,
                                                            use_cuda=args.use_cuda,
                                                            shuffle=True)
                                                            # shuffle=False)
    
    path_to_results_csv = args.results_csv
    results_csv, results_csv_writer = None, None
    if not os.path.exists(path_to_results_csv):
        results_csv = open(path_to_results_csv, 'a+', newline='')
        results_csv_writer = csv.writer(results_csv)
        results_csv_writer.writerow(["exp_name", "seed", "model", "latent_dim", "lr", "loss_func",
            "optimizer", "batch_size", "epochs", "faces", "final_train_loss", "final_val_loss", "final_test_loss"])
    else:
        results_csv = open(path_to_results_csv, 'a+', newline='')
        results_csv_writer = csv.writer(results_csv)

    results_csv_writer.writerow([args.exp_name, str(args.seed), args.model, str(args.latent_dim), str(args.lr),
        args.loss_func, args.optimizer, str(args.batch_size), str(args.epochs), args.faces, final_train_loss,
            final_val_loss, final_test_loss])
    
    results_csv.flush()
    results_csv.close()

    exit()

