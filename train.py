# from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from glob import glob
# from util import *
import numpy as np
import os
from PIL import Image
from VAE import VAE


def get_args(print_args=False):
    parser = argparse.ArgumentParser(description='PyTorch VAE')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')

    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
                        
    parser.add_argument('--dataset', required=True, type=str, metavar='path/to/dataset',
                        help='path to dataset')

    parser.add_argument('--num-train-imgs', required=True, type=int, metavar='N',
                        help='number of training images')
    
    parser.add_argument('--num-val-imgs', required=True, type=int, metavar='N',
                        help='number of validation images')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.no_cuda:
        args.cuda = False
    elif torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False
        print("args.cuda set to False because cuda is not available")

    if print_args:
        print(args)
        print('\n')

    if args.num_train_imgs > 162770:
        print("There are only 162770 training images")
        exit(1)
    
    if args.num_val_imgs > 19866:
        print("There are only 19866 validation images")
        exit(1)
    return args


def load_batch(batch_idx, istrain):
    if istrain:
        #   Pull from training set
        l = [str(batch_idx*batch_size + i).zfill(6) for i in range(1, batch_size+1)]
        # l = [str(batch_idx*batch_size + i).zfill(6) for i in range(batch_size)]
    else:
        #   Pull from validation set
        l = [str(batch_idx*batch_size + i + 162770).zfill(6) for i in range(1, batch_size+1)]
        # l = [str(batch_idx*batch_size + i + 162771).zfill(6) for i in range(batch_size)]

    data = []
    for idx in l:
        img = img_transforms(Image.open(template%idx))
        # img = Image.open(os.path.abspath(template%idx))
        data.append(np.array(img))
    data = [totensor(i) for i in data]
    
    return torch.stack(data, dim=0)

def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD

def load_last_model():
    models = glob('./models/*.pth')
    start_epoch = None
    last_cp = None
    if len(models) != 0:
        model_ids = [(int(f.split('_')[1]), f) for f in models]
        start_epoch, last_cp = max(model_ids, key=lambda item:item[0])
        print('Last checkpoint: ', last_cp)
        model.load_state_dict(torch.load(last_cp))
    else:
        print("No previous checkpoints available - fresh model beginning to train")
        start_epoch = 0
    return start_epoch, last_cp

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx in train_loader:
        data = load_batch(batch_idx, istrain=True)
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        # train_loss += loss.data[0]
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), (len(train_loader)*batch_size),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
                # loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / (len(train_loader)*batch_size)))
    return train_loss / (len(train_loader)*batch_size)

def validate(epoch):
    model.eval()
    val_loss = 0
    for batch_idx in val_loader:
        data = load_batch(batch_idx, istrain=False)
        data = Variable(data, volatile=True)
        if args.cuda:
            data = data.cuda()
        recon_batch, mu, logvar = model(data)
        # val_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        val_loss += loss_function(recon_batch, data, mu, logvar).item()

        torchvision.utils.save_image(data.data, './recons/Epoch_{}_batch_{}_data.jpg'.format(epoch, batch_idx), nrow=8, padding=2)
        torchvision.utils.save_image(recon_batch.data, './recons/Epoch_{}_batch_{}_recon.jpg'.format(epoch, batch_idx), nrow=8, padding=2)

    val_loss /= (len(val_loader)*batch_size)
    print('====> Val set loss: {:.4f}'.format(val_loss))
    return val_loss

def resume_training():
    start_epoch, _ = load_last_model()
    start_epoch = 0 if start_epoch == 0 else start_epoch + 1
    for epoch in range(start_epoch, start_epoch + args.epochs + 1):
        train_loss = train(epoch)
        val_loss = validate(epoch)
        torch.save(model.state_dict(), './models/Epoch_{}_Train_loss_{:.4f}_Val_loss_{:.4f}.pth'.format(epoch, train_loss, val_loss))



def setup_custom_logging():
    import datetime
    import sys
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    curr_date_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    
    logfilename = os.path.join('logs', "log-{}.txt".format(curr_date_time))
    outfile = open(logfilename, 'w')
    
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
    return logfilename
    


if __name__ == '__main__':
    
    logfilename = setup_custom_logging()
    print("Logging to {}\n".format(logfilename))
    
    args = get_args(print_args=True)

    #   CHANGE THE PATH SUITED TO YOUR MACHINE
    #   'template' variable is used in the function load_batch
    # template = '/Users/calvinpham/Downloads/Celeb/img_align_celeba/%s.jpg'
    template = os.path.join(args.dataset, '%s.jpg')

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    #   Network hyperparameters
    model = VAE(nc=3, ngf=128, ndf=128, latent_variable_size=500, use_cuda=args.cuda)
    
    if args.cuda:
        model.cuda()
    batch_size = args.batch_size
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    reconstruction_function = nn.BCELoss()
    reconstruction_function.size_average = False

    #   Train set: images 0 to 162770
    #   Validationn set: images 162711 to 182637
    #   Test set: images 182638 to 202599
    num_train_imgs = args.num_train_imgs
    num_val_imgs = args.num_val_imgs
    train_loader = range(num_train_imgs // batch_size) # <- will train with batch_size * 500 images
    val_loader = range(num_val_imgs // batch_size)

    
    # set up custom transform
    totensor = transforms.ToTensor()
    img_transforms = transforms.Scale((128, 128))
    
    #  Check if recons and models folder already exists or not
    if not os.path.exists("recons"):
        os.makedirs("recons")
    if not os.path.exists("models"):
        os.makedirs("models")

    resume_training()
