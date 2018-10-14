
# coding: utf-8

# # Keras tutorial - the Happy House
#
# Welcome to the first assignment of week 2. In this assignment, you will:
# 1. Learn to use Keras, a high-level neural networks API (programming framework), written in Python and capable of running on top of several lower-level frameworks including TensorFlow and CNTK.
# 2. See how you can in a couple of hours build a deep learning algorithm.
#
# Why are we using Keras? Keras was developed to enable deep learning engineers to build and experiment with different models very quickly. Just as TensorFlow is a higher-level framework than Python, Keras is an even higher-level framework and provides additional abstractions. Being able to go from idea to result with the least possible delay is key to finding good models. However, Keras is more restrictive than the lower-level frameworks, so there are some very complex models that you can implement in TensorFlow but not (without more difficulty) in Keras. That being said, Keras will work fine for many common models.
#
# In this exercise, you'll work on the "Happy House" problem, which we'll explain below. Let's load the required packages and solve the problem of the Happy House!

# In[ ]:

import numpy as np
from keras import layers
from keras import losses
from keras import optimizers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras import regularizers
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model


def leNet(input_shape):
    X_input = Input(input_shape)
    X = Conv2D(6, (5, 5), strides=(1, 1), name='conv1')(X_input)
    X = Activation('relu')(X)
    X = AveragePooling2D((2, 2), strides=(2, 2), name='av_pool1')(X)
    X = Conv2D(16, (5, 5), strides=(1, 1), name='conv2')(X)
    X = Activation('relu')(X)
    X = AveragePooling2D((2, 2), strides=(2, 2), name='av_pool2')(X)
    X = Flatten()(X)
    X = Dense(120, activation='relu', name='fc1')(X)
    X = Dense(84, activation='relu', name='fc2')(X)
    X = Dense(1, activation='sigmoid', name='final')(X)
    model = Model(inputs=X_input, outputs=X, name='HappyModel')
    return model


def alexNet(input_shape, dropout=0.3, lamd = 0.002):
    X_input = Input(input_shape)
    X = Conv2D(96, (7, 7), strides=(3, 3), name='conv1', activation="relu", padding="same")(X_input)
    X = MaxPooling2D((3, 3), strides=(1, 1), name='max_pool1')(X)
    X = Conv2D(256, (5, 5), strides=(3, 3), name='conv2', activation="relu", padding="same")(X)
    X = MaxPooling2D((3, 3), strides=(1, 1), name='max_pool2')(X)
    X = Conv2D(384, (3, 3), strides=(1, 1), name='conv3', activation="relu", padding="same")(X)
    X = Conv2D(384, (3, 3), strides=(1, 1), name='conv4', activation="relu", padding="same")(X)
    X = Conv2D(384, (3, 3), strides=(1, 1), name='conv5', activation="relu", padding="same")(X)
    X = MaxPooling2D((3, 3), strides=(2, 2), name='max_pool3')(X)
    X = Flatten()(X)
    X = Dense(9216, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(lamd))(X)
    X = Dropout(dropout)(X)
    X = Dense(4096, activation='relu', name='fc2', kernel_regularizer=regularizers.l2(lamd))(X)
    X = Dropout(dropout)(X)
    X = Dense(4096, activation='relu', name='fc3', kernel_regularizer=regularizers.l2(lamd))(X)
    X = Dropout(dropout)(X)
    X = Dense(1, activation='sigmoid', name='final')(X)

    model = Model(inputs=X_input, outputs=X, name='HappyModel')
    return model


happyModel = alexNet(X_train[0].shape)
happyModel.compile(optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=0.0, decay=0.0, amsgrad=False),
                   loss=losses.binary_crossentropy, metrics = ["accuracy"])
happyModel.fit(x = X_train, y = Y_train, epochs = 70, batch_size = 32)
predsT = happyModel.evaluate(x = X_train, y = Y_train)
preds = happyModel.evaluate(x = X_test, y = Y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Train Accuracy = " + str(predsT[1]))
print ("Test Accuracy = " + str(preds[1]))
happyModel.summary()

