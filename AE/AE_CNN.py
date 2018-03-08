#!/usr/bin/python
# coding=utf-8

'''
AutoEncoder
'''

import tensorflow as tf
import numpy as np
import skimage
from skimage.io import imsave
import cPickle as pickle

# Load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Constants
num_steps = 30000
batch_size = 64
learning_rate = 0.01
display_interval = 1000

num_hidden1 = 256
num_hidden2 = 128
pic_width = 28
num_input = pic_width * pic_width
width = 32

# Placeholders
# For ANN
#X = tf.placeholder(tf.float32, [None, num_input])
# For CNN
X = tf.placeholder(tf.float32, [None, width, width, 1])

def batch_resize(x):
    '''
    Resize image from (28, 28) to (32, 32) using sklearn.transform

    Args:
        x: image batch (-1, 28, 28, 1)
    Returns:
        image batch (-1, 32, 32, 1)
    '''
    resized_imgs = np.zeros((x.shape[0], 32, 32, 1))
    for i in range(x.shape[0]):
        resized_imgs[i, ..., 0] = skimage.resize(x[i, ..., 0], (32, 32))
    return resized_imgs

def cnn_encoder(x):
    '''
    CNN encoder
    32 * 32 * 1 -> 16 * 16 * 32
    16 * 16 * 32 -> 8 * 8 * 16
    8 * 8 * 16 -> 2 * 2 * 8

    Args:
        x: image batch (-1, 32, 32, 1)
    Returns:
        image batch (-1, 2, 2, 8)
    '''
    layer1 = tf.layers.conv2d(x, 32, [5, 5], strides=2, padding='SAME')
    layer2 = tf.layers.conv2d(layer1, 16, [5, 5], strides=2, padding='SAME')
    layer3 = tf.layers.conv2d(layer2, 8, [5, 5], strides=4, padding='SAME')
    return layer3

def cnn_decoder(x):
    '''
    CNN decoder
    2 * 2 * 8 -> 8 * 8 * 16
    8 * 8 * 16 -> 16 * 16 * 32
    16 * 16 * 32 -> 32 * 32 * 1

    Args:
        x: image batch (-1, 2, 2, 8)
    Returns:
        image batch (-1, 32, 32, 1)
    '''
    layer1 = tf.layers.conv2d_transpose(x, 16, [5, 5], strides=4, padding='SAME')
    layer2 = tf.layers.conv2d_transpose(layer1, 32, [5, 5], strides=2, padding='SAME')
    layer3 = tf.layers.conv2d_transpose(layer2, 1, [5, 5], strides=2, padding='SAME', activation=tf.nn.tanh)
    return layer3

    

# Model

# Fead-Forward NN
# encode_X = encoder(X)
# pred_X = decoder(encode_X)

# CNN
#X = X.reshape([None, 28, 28, 1])
encode_X = cnn_encoder(X)
pred_X = cnn_decoder(encode_X)

#loss = tf.metrics.mean_squared_error(labels=X, predictions=pred_X)
loss = tf.reduce_mean(tf.pow(X - pred_X, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    # Initilization
    sess.run(init)

    # Train
    for i in range(1, num_steps + 1):
        batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((-1, 28, 28, 1))
        batch_x = tf.image.resize_images(batch_x, (32, 32))

        _, error = sess.run([optimizer, loss], feed_dict={X:batch_x})

        if i % display_interval == 0:
            print("Step {step}, minibatch error {error}".format(step=i, error=error))
    
    # Test
    n = 4
    canvas_orig = np.empty([width * n, width * n])
    canvas_recv = np.empty([width * n, width * n])

    for i in range(n):
        batch_x, _ = mnist.test.next_batch(n)
        batch_x = batch_x.reshape((-1, 28, 28, 1))
        batch_x = tf.image.resize_images(batch_x, (32, 32))

        recv_x = sess.run(pred_X, feed_dict={X:batch_x.eval()})
        for j in range(n):
            canvas_orig[i*width:(i+1)*width, j*width:(j+1)*width] = batch_x[j].reshape([width, width])
            canvas_recv[i*width:(i+1)*width, j*width:(j+1)*width] = recv_x[j].reshape([width, width])
    
    # Change from [0, 1] to [0, 255]
    canvas_orig = canvas_orig * 255
    canvas_orig = canvas_orig.astype('int')
    canvas_recv = canvas_recv * 255
    canvas_recv = canvas_recv.astype('int')

    pickle.dump(canvas_orig, open('orig.pkl', 'w'))
    pickle.dump(canvas_recv, open('recv.pkl', 'w'))
    imsave('orig.bmp', canvas_orig)
    imsave('recv.bmp', canvas_recv)

