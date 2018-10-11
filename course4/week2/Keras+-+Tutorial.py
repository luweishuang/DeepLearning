
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
    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well. 
    
    
    ### END CODE HERE ###

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
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


# You have now built a function to describe your model. To train and test this model, there are four steps in Keras:
# 1. Create the model by calling the function above
# 2. Compile the model by calling `model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])`
# 3. Train the model on train data by calling `model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)`
# 4. Test the model on test data by calling `model.evaluate(x = ..., y = ...)`
# 
# If you want to know more about `model.compile()`, `model.fit()`, `model.evaluate()` and their arguments, refer to the official [Keras documentation](https://keras.io/models/model/).
# 
# **Exercise**: Implement step 1, i.e. create the model.

# In[ ]:

### START CODE HERE ### (1 line)
happyModel = HappyModel(X_train[0].shape)
### END CODE HERE ###


# **Exercise**: Implement step 2, i.e. compile the model to configure the learning process. Choose the 3 arguments of `compile()` wisely. Hint: the Happy Challenge is a binary classification problem.

# In[ ]:

### START CODE HERE ### (1 line)
happyModel.compile(optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                   loss=losses.binary_crossentropy, metrics = ["accuracy"])
### END CODE HERE ###


# **Exercise**: Implement step 3, i.e. train the model. Choose the number of epochs and the batch size.

# In[ ]:

### START CODE HERE ### (1 line)
happyModel.fit(x = X_train, y = Y_train, epochs = 40, batch_size = 32)
### END CODE HERE ###


# Note that if you run `fit()` again, the `model` will continue to train with the parameters it has already learnt instead of reinitializing them.
# 
# **Exercise**: Implement step 4, i.e. test/evaluate the model.

# In[ ]:

### START CODE HERE ### (1 line)
preds = happyModel.evaluate(x = X_test, y = Y_test)
### END CODE HERE ###
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
exit(0)

# If your `happyModel()` function worked, you should have observed much better than random-guessing (50%) accuracy on the train and test sets.
# 
# To give you a point of comparison, our model gets around **95% test accuracy in 40 epochs** (and 99% train accuracy) with a mini batch size of 16 and "adam" optimizer. But our model gets decent accuracy after just 2-5 epochs, so if you're comparing different models you can also train a variety of models on just a few epochs and see how they compare. 
# 
# If you have not yet achieved a very good accuracy (let's say more than 80%), here're some things you can play around with to try to achieve it:
# 
# - Try using blocks of CONV->BATCHNORM->RELU such as:
# ```python
# X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X)
# X = BatchNormalization(axis = 3, name = 'bn0')(X)
# X = Activation('relu')(X)
# ```
# until your height and width dimensions are quite low and your number of channels quite large (≈32 for example). You are encoding useful information in a volume with a lot of channels. You can then flatten the volume and use a fully-connected layer.
# - You can use MAXPOOL after such blocks. It will help you lower the dimension in height and width.
# - Change your optimizer. We find Adam works well. 
# - If the model is struggling to run and you get memory issues, lower your batch_size (12 is usually a good compromise)
# - Run on more epochs, until you see the train accuracy plateauing. 
# 
# Even if you have achieved a good accuracy, please feel free to keep playing with your model to try to get even better results. 
# 
# **Note**: If you perform hyperparameter tuning on your model, the test set actually becomes a dev set, and your model might end up overfitting to the test (dev) set. But just for the purpose of this assignment, we won't worry about that here.
# 

# ## 3 - Conclusion
# 
# Congratulations, you have solved the Happy House challenge! 
# 
# Now, you just need to link this model to the front-door camera of your house. We unfortunately won't go into the details of how to do that here. 

# <font color='blue'>
# **What we would like you to remember from this assignment:**
# - Keras is a tool we recommend for rapid prototyping. It allows you to quickly try out different model architectures. Are there any applications of deep learning to your daily life that you'd like to implement using Keras? 
# - Remember how to code a model in Keras and the four steps leading to the evaluation of your model on the test set. Create->Compile->Fit/Train->Evaluate/Test.

# ## 4 - Test with your own image (Optional)
# 
# Congratulations on finishing this assignment. You can now take a picture of your face and see if you could enter the Happy House. To do that:
#     1. Click on "File" in the upper bar of this notebook, then click "Open" to go on your Coursera Hub.
#     2. Add your image to this Jupyter Notebook's directory, in the "images" folder
#     3. Write your image's name in the following code
#     4. Run the code and check if the algorithm is right (0 is unhappy, 1 is happy)!
#     
# The training/test sets were quite similar; for example, all the pictures were taken against the same background (since a front door camera is always mounted in the same position). This makes the problem easier, but a model trained on this data may or may not work on your own data. But feel free to give it a try! 

# In[ ]:

### START CODE HERE ###
img_path = 'images/my_image.jpg'
### END CODE HERE ###
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))


# ## 5 - Other useful functions in Keras (Optional)
# 
# Two other basic features of Keras that you'll find useful are:
# - `model.summary()`: prints the details of your layers in a table with the sizes of its inputs/outputs
# - `plot_model()`: plots your graph in a nice layout. You can even save it as ".png" using SVG() if you'd like to share it on social media ;). It is saved in "File" then "Open..." in the upper bar of the notebook.
# 
# Run the following code.

# In[ ]:

happyModel.summary()


# In[ ]:

plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))

