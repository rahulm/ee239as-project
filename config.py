import os
class Config(object):

    # Name of dataset
    

    name = ''           # This will get hit inside get_model inside model.py

    OUTPUT_STRIDE = 16
    ENCODER = ''
    DECODER = ''

    def __init__(self, name, latent_dim, intermediate_dim, IMG_SIZE, IMG_CHANNEL, BATCH_SIZE, DATASET_SIZE, img_width=None, img_height=None, encoder='naive', dataset_root=None):
        """
        @OUTPUT_STRIDE: output stride of base network 
        STRIDE: output stride of base network where openpose keypoint network connects
        BATCH_SIZE: Batch size
        IMG_SIZE: Resolution of input image to network
        """
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        
        # For training:
        self.name = name

        if self.name == 'celeb':
            self.dataset_size = DATASET_SIZE
        elif self.name == 'mnist_binary':
            self.dataset_size = 60000 # 10k for test also

        self.IMG_SIZE = IMG_SIZE
        
        self.IMG_CHANNEL = IMG_CHANNEL
        self.BATCH_SIZE = BATCH_SIZE # should be 128
        self.VAL_BATCH_SIZE = 4
        self.NUM_GPUS = 1

        if img_width is not None:
            self.img_width = img_width

        if img_height is not None:
            self.img_height = img_height
            
        self.original_dim = self.img_height * self.img_width * self.IMG_CHANNEL

        self.ENCODER = encoder

        self.dataset_root = dataset_root
        self.images_root_path = os.path.join(self.dataset_root, 'img_align_celeba')
