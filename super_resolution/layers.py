import tensorflow as tf

from keras.layers import Input, Activation, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add

def conv_block(inputs, num_filters, kernel_size, strides = (2, 2),
               padding = 'same', activation = True):
    '''
        Simple convolutional block with batchnormalisation and relu
        activation if specified.

        Args:
            inputs (tensor):        A tensor with input layer (Keras 
                                    functional API)

            num_filters (integer):  An integer with the number of filters to
                                    use in the convolutional layers.

            kernel_size (tuple):    An integer or tuple/list of 2 integers,
                                    specifying the width and height of the
                                    2D convolution window. Can be a single 
                                    integer to specify the same value for 
                                    all spatial dimensions.

            strides (tuple):        An integer or tuple/list of 2 integers, 
                                    specifying the strides of the 
                                    convolution along the width and height. 
                                    Can be a single integer to specify the 
                                    same value for all spatial dimensions. 
                                    Specifying any stride value != 1 is 
                                    incompatible with specifying any 
                                    dilation_rate value != 1.

            padding (string):       One of 'valid' or 'same' (case-
                                    insensitive).

            activation (boolean):   Wether or not relu activation should be
                                    used.

        Returns: 4-tensor
    '''

    x = Conv2D(filters = num_filters,
               kernel_size = kernel_size,
               strides = strides,
               padding = padding)(inputs)
    x = BatchNormalization()(x)

    return Activation('relu')(x) if activation else x


def res_block(inputs, num_filters = 64):
    '''
        Define residual resnet style block.
        Args:
            inputs (tensor):        A tensor with input layer (Keras 
                                    functional API)
            num_filters (integer):  An integer with the number of filters to
                                    use in the convolutional layers.
        Returns: Residual block tensor
    '''

    # Convolutional blocks
    x = conv_block(inputs,
                   num_filters = num_filters,
                   kernel_size = (3, 3),
                   strides = (1, 1))
    x = conv_block(x,
                   num_filters = num_filters,
                   kernel_size = (3, 3),
                   strides = (1, 1),
                   activation = False)

    # Return as residual layer by adding block input and block output
    return add([x, inputs])


def up_block(inputs, num_filters, kernel_size):
    '''
        Simple upsample and convolutional block with batchnormalisation and
        relu activation.
        Args:
            inputs (tensor):        A tensor with input layer (Keras 
                                    functional API)
            num_filters (integer):  An integer with the number of filters to
                                    use in the convolutional layers.
            kernel_size (tuple):    An integer or tuple/list of 2 integers,
                                    specifying the width and height of the
                                    2D convolution window. Can be a single 
                                    integer to specify the same value for 
                                    all spatial dimensions.
    '''

    x = UpSampling2D()(inputs)

    x = Conv2D(num_filters,
               kernel_size = kernel_size,
               padding = 'same')(x)

    x = BatchNormalization()(x)

    return Activation('relu')(x)

