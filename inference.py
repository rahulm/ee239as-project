import torch
import torch.utils.data
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from VAE import VAE

class ImgToTensor(object):
    def __call__(self, sample):
        sample = sample.transpose((2, 0, 1))
        return torch.tensor(sample, dtype=torch.float32)/255

cuda_avail = torch.cuda.is_available()
all_face_images_warped = np.load('all-warped-images.npy')
face_images_train_warped = all_face_images_warped[:-100]
face_images_test_warped = all_face_images_warped[-100:]

totensor = transforms.ToTensor()
# sample_img = Image.open("/Users/calvinpham/Documents/EE 239AS/Term_Project/ee239as-project/images/000007.jpg")
app_vae = VAE(nc=3, ngf=128, ndf=128, latent_variable_size=500, use_cuda=cuda_avail)

# app_vae.load_state_dict(torch.load("./experiments/2019_03_08-01_37_45-experiment/models/Appearance-VAE-weights-epoch_5.pth", map_location=lambda storage, loc: storage)())
app_vae.load_state_dict(torch.load("./cpu_Epoch_28_Train_loss_25425.9503_Test_loss_25172.4120.pth", map_location=lambda storage, loc: storage))

app_vae.eval()
recon, _, _ = app_vae(ImgToTensor()(np.copy(face_images_train_warped[14])).unsqueeze(0))
# recon, _, _ = app_vae(totensor(sample_img).unsqueeze(0))
sample_app_recon = recon[0].squeeze().data.cpu().numpy().transpose((1, 2, 0))

plt.imshow(sample_app_recon)
plt.savefig("test.jpg")

