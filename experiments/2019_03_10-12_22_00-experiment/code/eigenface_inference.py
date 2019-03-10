from eigenface_train import ImgToTensor
import numpy as np
from appearance_vae import appearance_VAE
from appearance_ae import appearance_autoencoder
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

# read arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, required=True,
    help="path to model weights to perform inference with")
parser.add_argument('--model', type=str, required=True, choices=('ae', 'vae'),
    help="model type, either 'ae' or 'vae'")
parser.add_argument('--num_imgs', type=int, default=5,
    help="number of random images to sample and reconstruct.")
parser.add_argument('--faces', type=str, required=True, choices=('aligned', 'unaligned'),
    help="type of faces data to reconstruct from, choose from 'aligned' or 'unaligned'")
args = parser.parse_args()

# read images
if args.faces == 'aligned':
    all_face_images = np.load('all-warped-images.npy')
elif args.faces == 'unaligned':
    all_face_images = np.load('all-raw-images.npy')
face_images_train = all_face_images[:-100]
face_images_test = all_face_images[-100:]

# get model and weights
if args.model == 'ae':
    app_model = appearance_autoencoder(latent_dim_size=50)
elif args.model == 'vae':
    app_model = appearance_VAE(latent_dim_size=50)
else:
    print("Model {} not recognized. Please use 'ae' or 'vae' only.".format(args.model))
app_model.load_state_dict(torch.load(args.weights, map_location=lambda storage, loc: storage)())


# get original images
num_imgs = args.num_imgs
img_inds = np.random.choice(np.arange(len(all_face_images)), size=num_imgs, replace=False)
sample_imgs = np.copy(all_face_images[img_inds])

# num_imgs = 20
# sample_imgs = np.copy(all_face_images[list(range(800,821))])

sample_img_tensors = []
for i in range(num_imgs):
    sample_img_tensors.append(ImgToTensor()(sample_imgs[i]))
sample_img_tensors = torch.stack(sample_img_tensors)

# recon
app_model.eval()
out = app_model(sample_img_tensors)
if args.model == 'vae':
    out = out[0]
sample_img_recons = out.data.cpu().numpy().transpose((0, 2, 3, 1))

# plot
fig = plt.figure()
rows = num_imgs
cols = 2
for i in range(num_imgs):
    fig.add_subplot(rows, cols, (i*2)+1)
    plt.imshow(sample_imgs[i])
    fig.add_subplot(rows, cols, (i*2)+2)
    plt.imshow(sample_img_recons[i])
plt.show()
