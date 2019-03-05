
class Config(object):

    # Name of dataset
    

    name = ''           # This will get hit inside get_model inside model.py
    latent_dim = 2
    intermediate_dim = 512

    OUTPUT_STRIDE = 16
    ENCODER = ''
    DECODER = ''

    



    def __init__(self, name, IMG_SIZE, IMG_CHANNEL, BATCH_SIZE, DATASET_SIZE, img_width=None, img_height=None, encoder='naive'):
        """
        @OUTPUT_STRIDE: output stride of base network 
        STRIDE: output stride of base network where openpose keypoint network connects
        BATCH_SIZE: Batch size
        IMG_SIZE: Resolution of input image to network
        """
        
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

