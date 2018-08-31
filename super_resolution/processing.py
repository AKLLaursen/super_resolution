import numpy as np
import keras.backend as K

def preproc(x):
    '''
        Demean input using Imagenet means from the original VGG paper and
        reverse channels.

        Args:
            x (tensor): 4-dimensional input tensor (an image of 
                        (batches x height x width x channels))

        Returns: The 4-dimensional input tensor with Imagenet mean 
                 subtracted and channels reversed
    '''

    rn_mean = np.array([123.68, 116.779, 103.939], dtype = np.float32)
    
    return (x - rn_mean)[:, :, :, ::-1]


def get_output(model, layer):
    '''
        Function returning a given VGG conv block layer of specified name.

        Args:
            model (tensor):     A model as tensor to extract model layers
                                from.

            layer (tupple):     A tuple consisting of (block_num, layer_num),
                                where block_num is an integer representing the
                                given block number of VGG to use and layer_num
                                is an integer, representing the conv number to
                                use in the given block. 

        Returns: A tensor representing the given layer
    '''

    return model.get_layer('block_' + str(layer[0]) + '_conv_' \
        + str(layer[1])).output


def mean_sqr(diff):
    '''
        Calculate the mean squared error based on a given input difference.

        Args:
            diff (tensor): Two 4-tensors subtracted from each other

        Returns: The MSE
    '''
    # Get tensor dimensions
    dims = list(range(1, K.ndim(diff)))
    
    # Calculate MSE
    return K.expand_dims(K.sqrt(K.mean(diff ** 2, dims)), 0)

