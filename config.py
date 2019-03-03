
class Config(object):

    # Name of dataset
    

    DATASET = 'mnist_binary'
    latent_dim = 2
    intermediate_dim = 512

    ENCODER = ''
    DECODER = ''
    



    def __init__(self, IMG_SIZE, BATCH_SIZE):
        """
        @OUTPUT_STRIDE: output stride of base network 
        STRIDE: output stride of base network where openpose keypoint network connects
        BATCH_SIZE: Batch size
        IMG_SIZE: Resolution of input image to network
        """
        
        # For training:
        self.IMG_SIZE = IMG_SIZE
        self.BATCH_SIZE = BATCH_SIZE # should be 128
        self.NUM_GPUS = 1
        self.original_dim = self.IMG_SIZE * self.IMG_SIZE

