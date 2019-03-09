from train import ImgToTensor
import numpy as np
from appearance_vae import appearance_VAE
import matplotlib.pyplot as plt
import torch


all_face_images_warped = np.load('all-warped-images.npy')
face_images_train_warped = all_face_images_warped[:-100]
face_images_test_warped = all_face_images_warped[-100:]

#   Inference
app_vae = appearance_VAE(latent_dim_size=50)
app_vae.load_state_dict(torch.load("./experiments/2019_03_08-01_37_45-experiment/models/Appearance-VAE-weights-epoch_4.pth", map_location=lambda storage, loc: storage)())

app_vae.eval()
selected_img = app_vae(ImgToTensor()(np.copy(face_images_test_warped[12])).unsqueeze(0))
sample_app_recon = selected_img[0].squeeze().data.cpu().numpy().transpose((1, 2, 0))

plt.imshow(sample_app_recon)
plt.show()


#   Inference with old AE
# app_model = appearance_autoencoder(latent_dim_size=50)
# app_model.load_state_dict(torch.load("appearance_model_weights31October2018-13_53.pth", map_location=lambda storage, loc: storage)())

# app_model.eval()
# selected_img = app_model(ImgToTensor()(np.copy(face_images_test_warped[12])).unsqueeze(0))
# sample_app_recon = selected_img.squeeze().data.cpu().numpy().transpose((1, 2, 0))

# plt.imshow(sample_app_recon)
# plt.show()
