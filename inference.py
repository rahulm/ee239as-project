from train import ImgToTensor
import numpy as np
from appearance_vae import appearance_VAE
from appearance_ae import appearance_autoencoder
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, required=True,
    help="path to model weights to perform inference with")
args = parser.parse_args()


all_face_images_warped = np.load('all-warped-images.npy')
face_images_train_warped = all_face_images_warped[:-100]
face_images_test_warped = all_face_images_warped[-100:]

#   Inference
app_vae = appearance_VAE(latent_dim_size=50)
# app_vae.load_state_dict(torch.load("./experiments/2019_03_08-01_37_45-experiment/models/Appearance-VAE-weights-epoch_5.pth", map_location=lambda storage, loc: storage)())
app_vae.load_state_dict(torch.load(args.weights, map_location=lambda storage, loc: storage)())


# img_ind = 12
img_ind = np.random.randint(0, high=len(all_face_images_warped)+1)
sample_img = np.copy(all_face_images_warped[img_ind])
# sample_img = np.copy(face_images_train_warped[12])

app_vae.eval()
sample_img_recon_batch = app_vae(ImgToTensor()(sample_img).unsqueeze(0))
sample_img_recon = sample_img_recon_batch[0].squeeze().data.cpu().numpy().transpose((1, 2, 0))

fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(sample_img)
fig.add_subplot(1, 2, 2)
plt.imshow(sample_img_recon)
plt.show()


#   Inference with old AE
# app_model = appearance_autoencoder(latent_dim_size=50)
# app_model.load_state_dict(torch.load("appearance_model_weights31October2018-13_53.pth", map_location=lambda storage, loc: storage)())

# app_model.eval()
# selected_img = app_model(ImgToTensor()(np.copy(face_images_test_warped[12])).unsqueeze(0))
# sample_app_recon = selected_img.squeeze().data.cpu().numpy().transpose((1, 2, 0))

# plt.imshow(sample_app_recon)
# plt.show()
