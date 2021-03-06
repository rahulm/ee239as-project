# from eigenface_train import ImgToTensor
import numpy as np
import torch
import csv
import os
from mywarper import plot, plot_with_titles


def perform_eigenface_inference(model, test_images, test_tensor, path_to_save_inference,
                                all_images=None, path_to_save_nn=None,
                                knn_csv_path=None, epoch=None):
    model.eval()
    out = model(test_tensor)
    if model.MODEL_TYPE == 'vae':
        out = out[0]
    
    sample_img_recons = out.data.cpu().numpy().transpose((0, 2, 3, 1))

    orig_and_recons = np.concatenate((test_images, sample_img_recons), axis=0)
    plot(orig_and_recons, 2, len(test_images), 3, 128, 128, path_to_save_inference)
    
    if path_to_save_nn is not None and all_images is not None:
        num_imgs = len(all_images)
        flat_imgs = np.reshape(all_images, (num_imgs, -1))
        
        num_recons = len(sample_img_recons)
        flat_recons = np.reshape(sample_img_recons, (num_recons, -1))
        
        nearest_neighbor_inds = []
        nearest_neighbor_dists = []
        nearest_neighbors = []
        # for recon in sample_img_recons:
        for recon in flat_recons:
            dists = np.linalg.norm(flat_imgs - (recon * 255), axis=1)
            min_i = np.argmin(dists)
            nearest_neighbor_inds.append(min_i)
            nearest_neighbor_dists.append(dists[min_i])
            nearest_neighbors.append(all_images[min_i])
        
        nearest_neighbors = np.asarray(nearest_neighbors)
        nn_and_recons = np.concatenate((test_images, sample_img_recons, nearest_neighbors), axis=0)
        plot(nn_and_recons, 3, len(test_tensor), 3, 128, 128, path_to_save_nn)
        
        
        # save to csv
        if knn_csv_path is not None:
            
            # create headers
            headers = ['epoch', 'ave-euclidean_dist']
            header_per_image = ['img_{}-knn_ind', 'img_{}-euclidean_dist']
            for i in range(num_recons):
                headers.extend([h.format(i) for h in header_per_image])
            
            # if it already exists, assume headers are the same. otherwise create
            needs_headers = (not os.path.exists(knn_csv_path))
            with open(knn_csv_path, 'a+', newline='') as knn_csv_file:
                writer = csv.writer(knn_csv_file)
                
                if needs_headers:
                    writer.writerow(headers)
                
                ave_dist = np.mean(nearest_neighbor_dists)
            
                row = [str(epoch), "{:f}".format(ave_dist)]
                for img_ind, dist in zip(nearest_neighbor_inds, nearest_neighbor_dists):                    
                    row.append(str(img_ind))
                    row.append("{:f}".format(dist))
                
                writer.writerow(row)
        
        

def perform_eigenface_sampling(model, use_cuda, num_generate, all_images, img_out_path, knn_csv_path, epoch):
    model.eval()
    z = torch.randn(num_generate, model.latent_dim_size)
    z = z.to('cuda:0' if use_cuda else 'cpu')
    generated_faces = model.get_recon_from_latent_vec(z)
    generated_faces = np.asarray(generated_faces.data.cpu().numpy().transpose((0, 2, 3, 1)))

    num_imgs = len(all_images)
    flat_imgs = np.reshape(all_images, (num_imgs, -1))
    
    generated_flat = np.reshape(generated_faces, (num_generate, -1))
    
    nearest_neighbor_inds = []
    nearest_neighbor_dists = []
    nearest_neighbors = []
    for gen_i, gen_face in enumerate(generated_flat):
        dists = np.linalg.norm(flat_imgs - (gen_face * 255), axis=1)
        min_i = np.argmin(dists)
        nearest_neighbor_inds.append(min_i)
        nearest_neighbors.append(all_images[min_i])
        # nearest_neighbors.append(all_images[np.argmin(dists)])
        
        # nearest_neighbor_dists[gen_i + num_generate] = dists[min_i]
        nearest_neighbor_dists.append(dists[min_i])
    
    nearest_neighbors = np.asarray(nearest_neighbors)
    gen_and_nn = np.concatenate((generated_faces, nearest_neighbors), axis=0)
    plot(gen_and_nn, 2, num_generate, 3, 128, 128, img_out_path)
    # plot_with_titles(gen_and_nn, 2, num_generate, 3, 128, 128, nearest_neighbor_dists, img_out_path)
    
    # save to csv
    if knn_csv_path is not None:
        
        # create headers
        headers = ['epoch', 'ave-euclidean_dist']
        header_per_image = ['sample_{}-knn_ind', 'sample_{}-euclidean_dist']
        for i in range(num_generate):
            headers.extend([h.format(i) for h in header_per_image])
        
        # if it already exists, assume headers are the same. otherwise create
        needs_headers = (not os.path.exists(knn_csv_path))
        with open(knn_csv_path, 'a+', newline='') as knn_csv_file:
            writer = csv.writer(knn_csv_file)
            
            if needs_headers:
                writer.writerow(headers)
            
            ave_dist = np.mean(nearest_neighbor_dists)
            
            row = [str(epoch), "{:f}".format(ave_dist)]
            for img_ind, dist in zip(nearest_neighbor_inds, nearest_neighbor_dists):                    
                row.append(str(img_ind))
                row.append("{:f}".format(dist))
            
            writer.writerow(row)


# def find_eigenface_NN(model, test_tensor, all_images, path_to_save):
    # model.eval()
    # out = model(test_tensor)
    # if model.MODEL_TYPE == 'vae':
        # out = out[0]
    
    # sample_img_recons = out.data.cpu().numpy().transpose((0, 2, 3, 1))

    # num_imgs = len(all_images)
    # flat_imgs = np.reshape(all_images, (num_imgs, -1))
    
    # num_recons = len(sample_img_recons)
    # flat_recons = np.reshape(sample_img_recons, (num_recons, -1))
    
    # nearest_neighbors = []
    # # for recon in sample_img_recons:
    # for recon in flat_recons:
        # # dists = np.linalg.norm(all_images - (recon * 255), axis=1)
        # dists = np.linalg.norm(flat_imgs - (recon * 255), axis=1)
        # print('len', len(dists))
        # print('shape', dists.shape)
        # nearest_neighbors.append(all_images[np.argmin(dists)])
    
    # nearest_neighbors = np.asarray(nearest_neighbors)
    # nn_and_recons = np.concatenate((nearest_neighbors, sample_img_recons), axis=0)
    # plot(nn_and_recons, 2, len(test_tensor), 3, 128, 128, path_to_save)


# # read arguments
# import argparse
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser.add_argument('--num_imgs', type=int, default=5,
#     help="number of random images to sample and reconstruct.")
# parser.add_argument('--appear_latent_dim', type=int, default=50,
#     help="number of elements in the latent vector for the appearance model")
# # parser.add_argument('--landmark_latent_dim', type=int, default=10,
#     # help="number of elements in the latent vector for the landmark model")

# required_group = parser.add_argument_group('required arguments')
# required_group.add_argument('--weights', type=str, required=True,
#     help="path to model weights to perform inference with")
# required_group.add_argument('--model', type=str, required=True, choices=('ae', 'vae'),
#     help="model type, either 'ae' or 'vae'")
# required_group.add_argument('--faces', type=str, required=True, choices=('aligned', 'unaligned'),
#     help="type of faces data to reconstruct from, choose from 'aligned' or 'unaligned'")

# args = parser.parse_args()


# read images
# if args.faces == 'aligned':
#     all_face_images = np.load('all-warped-images.npy')
# elif args.faces == 'unaligned':
#     all_face_images = np.load('all-raw-images.npy')
# face_images_train = all_face_images[:-100]
# face_images_test = all_face_images[-100:]

# # get model and weights
# if args.model == 'ae':
#     # app_model = appearance_autoencoder(latent_dim_size=50)
#     app_model = appearance_autoencoder(latent_dim_size=args.appear_latent_dim)
# elif args.model == 'vae':
#     # app_model = appearance_VAE(latent_dim_size=50)
#     app_model = appearance_VAE(latent_dim_size=args.appear_latent_dim)
# else:
#     print("Model {} not recognized. Please use 'ae' or 'vae' only.".format(args.model))
# app_model.load_state_dict(torch.load(args.weights, map_location=lambda storage, loc: storage)())


# # get original images
# num_imgs = args.num_imgs
# img_inds = np.random.choice(np.arange(len(all_face_images)), size=num_imgs, replace=False)
# sample_imgs = np.copy(all_face_images[img_inds])

# # num_imgs = 20
# # sample_imgs = np.copy(all_face_images[list(range(800,821))])

# sample_img_tensors = []
# for i in range(num_imgs):
#     sample_img_tensors.append(ImgToTensor()(sample_imgs[i]))
# sample_img_tensors = torch.stack(sample_img_tensors)

# # recon
# app_model.eval()
# out = app_model(sample_img_tensors)
# if args.model == 'vae':
#     out = out[0]
# sample_img_recons = out.data.cpu().numpy().transpose((0, 2, 3, 1))

# # plot
# fig = plt.figure()
# rows = num_imgs
# cols = 2
# for i in range(num_imgs):
#     fig.add_subplot(rows, cols, (i*2)+1)
#     plt.imshow(sample_imgs[i])
#     fig.add_subplot(rows, cols, (i*2)+2)
#     plt.imshow((sample_img_recons[i]*255).astype(np.uint8)) # FIXED Mult by 255 -> int conversion
# plt.show()


# Latent vector analysis / sampling
# Input -> model
# Output -> sampled images in the experiments/sampling directory with model name prepended
# z = torch.randn(args.num_imgs, app_model.latent_dim_size)
# z = Variable(z, volatile=True)
# recon = app_model.get_recon_from_latent_vec(z)

# weights_dir, weights_file = os.path.split(args.weights)
# experiment_dir, _ = os.path.split(weights_dir)
# _, experiment_name = os.path.split(experiment_dir)

# model_spec = weights_file.replace('.pth', '')
# output_images_path = os.path.join(experiment_dir, 'sampling')

# if not os.path.isdir(output_images_path):
#     print('creating output image reconstruction directory...')
#     os.mkdir(output_images_path)
# print("using {}\n".format(output_images_path))

# # Reconstructions built in utilities
# # torchvision.utils.save_image(recon.data, output_images_path+'rand_faces_'+str(args.model) + str(model_spec)+ '.jpg', nrow=args.num_imgs, padding=2)
# recon_output_file = os.path.join(output_images_path, 'rand_faces-{}-{}.jpg'.format(args.model, model_spec))
# torchvision.utils.save_image(recon.data, recon_output_file, nrow=args.num_imgs, padding=2)

# # ===============================
# # === Do not delete =============
# # Reconstruction without utilities
# recon_data = recon.data.numpy()
# recon_data = recon_data.transpose((0, 2, 3, 1))
# recon_data = (recon_data * 255).astype(np.uint8)
# # plt.imshow(recon_data[0])
# # plt.show()
# # ===============================

# # Latent space analysis
# # uses sample_image_recons with KNN 
# # uses reconstructed images with KNN
# def compute_L2_distances_vectorized(X_recon, X):
#     """
#     Compute the distance between each test point in X and each training point
#     in self.X_train WITHOUT using any for loops.
#     Inputs:
#     - X: A numpy array of shape (num_test, D) containing test data.
#     Returns:
#     - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
#       is the Euclidean distance between the ith test point and the jth training
#       point.
#     """
#     # num_test = X.shape[0]
#     # num_train = X_recon.shape[0]
#     # dists = np.zeros((num_test, num_train))
    
#     # # dists = np.linalg.norm(X_recon - X, axis=1)
#     # X_testsq = np.sum(np.square(X), axis=1, keepdims=True)
#     # X_trainsq = np.sum(np.square(X_recon), axis=1)
#     # XTX = X.dot(X_recon.T)

#     # dists = np.sqrt((X_trainsq - 2*XTX) + X_testsq)
    
#     dists = np.linalg.norm(X - X_recon, axis=1)
#     return dists

# # Can compare the reconstructed image distances to its corresponding sample
# all_face_images_knn = all_face_images.reshape(all_face_images.shape[0], -1)
# sample_img_recons_knn = sample_img_recons.reshape(args.num_imgs, -1)


# l2_dists = []
# min_dists = []
# train_neighbors = []
# for i, recon in enumerate(sample_img_recons_knn):
#     recon_unprocessed = np.expand_dims((recon * 255).astype(np.uint8), axis=0)
#     # l2_dist = compute_L2_distances_vectorized(all_face_images_knn, recon_unprocessed)[0]
#     # l2_dist[np.isnan(l2_dist)] = 0
#     # train_neighbor = np.argmin(l2_dist[[l2_dist>0]])
#     # min_dist = np.min(l2_dist[l2_dist>0])
#     # min_dists.append(min_dist)
#     # train_neighbors.append(train_neighbor)
    
#     l2_dist = compute_L2_distances_vectorized(recon_unprocessed, all_face_images_knn)
#     train_neighbor = np.argmin(l2_dist)
#     min_dist = l2_dist[train_neighbor]
    
#     fig = plt.figure()
    
#     # plot original
#     # ax = fig.add_subplot(nrows=3, ncols=1, index=1)
#     ax = fig.add_subplot(311)
#     ax.set_title('Original image')
#     plt.imshow(sample_imgs[i])
    
#     # plot reconstruction
#     # ax = fig.add_subplot(nrows=3, ncols=1, index=2)
#     ax = fig.add_subplot(312)
#     ax.set_title('Recon image')
#     plt.imshow((sample_img_recons[i] * 255).astype(np.uint8))
    
#     # plot neearest neighbor
#     # ax = fig.add_subplot(nrows=3, ncols=1, index=3)
#     ax = fig.add_subplot(313)
#     ax.set_title('Train set neighbor' + str(min_dist)+ ' (l2)')
#     plt.imshow(all_face_images[train_neighbor])
    
    
#     # plt.gcf()
#     plot_output_file = os.path.join(output_images_path, 'img_reconknn-{}-{}-{}.jpg'.format(args.model, model_spec, i))
#     plt.savefig(plot_output_file)


# TODO: Same thing for sampled images once they look better
# Can compare sampled images to dataset (generated samples)
# recon_data_flat = recon_data.reshape(args.num_imgs, -1)
# l2_dists = compute_L2_distances_vectorized(all_face_images_knn, recon_data_flat)