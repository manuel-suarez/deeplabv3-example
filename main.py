# Import all modules for model building
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import Add, concatenate

# Import all modules for losses and training
import skimage.measure
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from math import exp, isnan, pow, ceil
from tensorflow.keras.metrics import Precision, Recall, IoU
from tensorflow.keras.callbacks import ModelCheckpoint

# Import all modules for data providing
import glob
import itertools
import os
import random
from sklearn.model_selection import train_test_split

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    y = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    y = BatchNormalization(name=bn_name_base + '2a')(y)
    y = Activation('relu')(y)

    y = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(y)
    y = BatchNormalization(name=bn_name_base + '2b')(y)
    y = Activation('relu')(y)

    y = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(y)
    y = BatchNormalization(name=bn_name_base + '2c')(y)

    y = Add()([y, input_tensor])
    y = Activation('relu')(y)
    return y


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    y = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    y = BatchNormalization(name=bn_name_base + '2a')(y)
    y = Activation('relu')(y)

    y = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(y)
    y = BatchNormalization(name=bn_name_base + '2b')(y)
    y = Activation('relu')(y)

    y = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(y)
    y = BatchNormalization(name=bn_name_base + '2c')(y)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    y = Add()([y, shortcut])
    y = Activation('relu')(y)
    return y


def ResNet50(input_shape, refinenet=False, deeplab=False):
    """Instantiates the ResNet50 architecture.
    # Arguments
    input_shape: tuple
        Input image shape with chanels
    refinenet: bool, optional
        Is this ResNet50 used in RefineNet
    deeplab: bool, optional
        Is this ResNet50 used in DeepLabV3
    # Returns
        A tensorflow.keras.Model instance of ResNet50
    """
    img_input = Input(input_shape)

    out = []

    x = ZeroPadding2D((2, 2))(img_input)
    x = Conv2D(64, (5, 5), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    if refinenet:
        out.append(x)

    if deeplab:
        out.append(x)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    if refinenet:
        out.append(x)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    if refinenet:
        out.append(x)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    out.append(x)

    return Model(img_input, out)