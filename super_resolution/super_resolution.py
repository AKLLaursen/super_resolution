import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from keras_tqdm import TQDMNotebookCallback

from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.convolutional import Conv2D
import keras.backend as K

from .layers import conv_block, up_block, res_block
from .processing import (preproc, get_output, mean_sqr)
from .plotting import display_image_in_actual_size

from kerastools.vgg import Vgg

class SuperResolution(object):

    def __init__(self,
                 shape = (288, 288, 3),
                 w = [0.025, 0.8, 0.15, 0.025], 
                 vgg_layers = [(2, 1), (3, 1), (4, 2), (5, 2)]):
        
        self.shape = shape
        self.w = w
        self.vgg_layers = vgg_layers

        self.create()

    def tot_loss(self, x):
        '''
            Calculate the total loss between content loss x and the style loss 
            based on given weights to the perceptual loss (individual conv layer 
            loss).

            Args:
                x (tensor):             A 4-tensor of (batch size, img height, 
                                        img width, channels)

            Returns: The calculated total loss.
        '''

        # Initialise
        loss = 0
        n = len(self.vgg_layers)

        # Loop over the number of style target (number of conv layers used in
        # perceptual loss).
        for i in range(n):

            # Content loss - The super resolution vgg layers and the non super
            # resolution layers should be equal.
            loss += mean_sqr(x[i] - x[i + n]) * self.w[i]

        return loss


    def create(self):
        '''
            Create style transfer model in the style of Johnston et. al. with 
            given inputs

            Args:

            Returns: The given style transfer model
        '''


        # Define super resolution model
        # Input. Dimensions are set as None in order to use dynamic image size
        inputs = Input((None, None, 3))

        # Convolutional block layers
        conv_layer = conv_block(inputs, 64, (9, 9), (1, 1))

        # Residual resnet style blocks
        res_layer = conv_layer
        for i in range(4):
            res_layer = res_block(res_layer)
        
        # Upsample blocks
        up_layer = up_block(res_layer, 64, (3, 3))
        #up_layer = up_block(up_layer, 64, (3, 3))

        # Final super resolution (3 x 288 x 288) conv layer with tanh activation
        final_conv = Conv2D(3,
                            kernel_size = (9, 9),
                            activation = 'tanh',
                            padding = 'same')(up_layer)

        # Transform final tanh activation layer to output signal in range of
        # [0, 255]
        outputs = Lambda(lambda x: (x + 1) * 127.5)(final_conv)

        # Define VGG model (no top) to be used as content model
        vgg_inp = Input(self.shape)
        vgg = Vgg(include_top = False,
                  input_tensor = Lambda(preproc)(vgg_inp),
                  pooling = 'max').model

        # Make the the VGG layers untrainable
        for l in vgg.layers:
            l.trainable = False

        # Define content model, with output from the 4 last content layers
        vgg_content = Model(vgg_inp,
                            [get_output(vgg, layer) for layer in self. vgg_layers])

        # In Keras' functional API, any layer (and a model is a layer), can be 
        # treated as if it was a function. So we can take any model and pass it 
        # a tensor, and Keras will join the model together.

        # Just the VGG model with perceptual loss.
        vgg1 = vgg_content(vgg_inp)

        # The VGG model with perceptual los on top and the super resolution
        # model on the buttom.
        vgg2 = vgg_content(outputs)

        # Define total loss
        loss = Lambda(self.tot_loss)(vgg1 + vgg2 + [outputs])

        # Define final style transfer model
        self.m_style = Model([inputs, vgg_inp], loss)
        self.inputs = inputs
        self.outputs = outputs


    def compile_single_gpu(self, optimizer = 'adam', loss = 'mae'):
        '''
            Compile model.

            Args:
                None

            Returns:
                Nothing
        '''
        self.m_style.compile(optimizer = optimizer,
                             loss = loss)
        
        
    def compile_multi_gpu(self, optimizer = 'adam', loss = 'mae'):
        '''
            Compile model for multi (>2) GPU usage.

            Args:
                None

            Returns:
                Nothing
        '''
        
        pass


    def fit(self, train_data_lr, train_data_hr, batch_size, epochs, learning_rate = 1e-3):
        '''
            Function to fit the style transfer model.

            Args:
                train_data (ndarray):   Array of symmetrical train images in 
                                        (288 x 288) size.

                batch_size (int):       Batch size for training

                epochs (int):           Number of epochs to run

            Returns: None

        '''

        # Dictionary to use TQDMN to visualise training
        params = {'verbose': 0,
                 'callbacks': [TQDMNotebookCallback(leave_inner = True)]}

        # Our target is 0 loss of the MSE (final model output)
        target = np.zeros((train_data_lr.shape[0], 1))

        # Set learning rate
        K.set_value(self.m_style.optimizer.lr, learning_rate)

        # Fit model
        self.m_style.fit(x = [train_data_lr, train_data_hr],
                         y = target,
                         batch_size = batch_size,
                         epochs = epochs,
                         **params)


    def get_top(self):
        '''
            Get top layer model for doing super resolution transfer
            
            Args:
                None

            Returns: None
        '''

        self.top_model = Model(self.inputs, self.outputs)


    def predict(self, image_path, plot_image = True):

        # Read image
        input_image = Image.open(image_path)

        # Convert to numpy array
        input_image = np.expand_dims(np.array(input_image), 0)

        # Predict (do style transfer)
        self.image_with_style = self.top_model.predict(input_image)

        if plot_image: display_image_in_actual_size(self.image_with_style)
            