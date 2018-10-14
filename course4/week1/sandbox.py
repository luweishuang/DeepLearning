import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

def prepare_dataset():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    X_train, X_test = normalize_dataset(X_train_orig, X_test_orig)
    Y_train, Y_test = convert_to_nn_type(Y_train_orig, Y_test_orig, classes)
    return X_train, X_test, Y_train, Y_test

def normalize_dataset(X_train_orig, X_test_orig):
    X_train_norm = X_train_orig / 255.
    X_test_norm = X_test_orig / 255.
    return X_train_norm, X_test_norm

def convert_to_nn_type(Y_train_orig, Y_test_orig, classes):
    class_size = len(classes)
    Y_train = np.eye(class_size)[Y_train_orig.reshape(-1)]
    Y_test = np.eye(class_size)[Y_test_orig.reshape(-1)]
    return Y_train, Y_test

def conv_forward_layer(input, filter, s, p):
    Z = tf.nn.conv2d(input, filter, strides=[1, s, s, 1], padding="SAME")
    A = tf.nn.relu(Z)
    P = tf.nn.max_pool(A, ksize=[1, p, p, 1], strides=[1, p, p, 1], padding="SAME")
    return P

def conv_forward(X, W, Y):
    W1, W2, W3 = W
    A1 = conv_forward_layer(X, W1, s=1, p=2)
    A2 = conv_forward_layer(A1, W2, s=1, p=2)
    A3 = conv_forward_layer(A2, W3, s=1, p=4)
    F = tf.contrib.layers.flatten(A3)
    A4 = tf.contrib.layers.fully_connected(F, 10, activation_fn=tf.nn.relu)
    A5 = tf.contrib.layers.fully_connected(A4, 10, activation_fn=tf.nn.relu)
    A6 = tf.contrib.layers.fully_connected(A5, 6, activation_fn=None)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=A6, labels=Y))
    return cost, A6


def model(num_epochs, learning_rate, print_cost=True):
    np.random.seed(1)
    X_train, X_test, Y_train, Y_test = prepare_dataset()


    X = tf.placeholder(tf.float32, (None, X_train.shape[1], X_train.shape[2], X_train.shape[3]), name="X")
    Y = tf.placeholder(tf.float32, (None, Y_train.shape[1]), name="Y")
    W1 = tf.get_variable("W1", [3, 3, 3, 8], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [5, 5, 8, 16], initializer=tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable("W3", [7, 7, 16, 32], initializer=tf.contrib.layers.xavier_initializer())
    W = (W1, W2, W3)
    cost, A6 = conv_forward(X, W, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        # Do the training loop
        for epoch in range(num_epochs):
            _, temp_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            if print_cost and epoch % 2 == 0:
                print("Cost after epoch %i: %f" % (epoch, temp_cost))


        # Calculate the correct predictions
        predict_op = tf.argmax(A6, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy


model(500, 0.009)