#!/usr/bin/python
# coding=utf-8

'''
Contrative AutoEncoder
'''

import tensorflow as tf
import numpy as np
from skimage.io import imsave
import cPickle as pickle

# Load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Constants
num_steps = 30000
batch_size = 256
learning_rate = 0.01
lamb = 1e-5
display_interval = 1000

num_hidden = 100
pic_width = 28
num_input = pic_width * pic_width

# Placeholders
X = tf.placeholder(tf.float32, [None, num_input])

# Variables
variables_dict = {
    "encoder_w": tf.Variable(tf.random_normal([num_input, num_hidden]), dtype=tf.float32),
    "encoder_b": tf.Variable(tf.random_normal([num_hidden]), dtype=tf.float32),
    "decoder_w": tf.Variable(tf.random_normal([num_hidden, num_input]), dtype=tf.float32),
    "decoder_b": tf.Variable(tf.random_normal([num_input]), dtype=tf.float32)
}

def encoder(x):
    '''
    $$
    h = \sigma(XW + b)
    $$
    '''
    return tf.nn.sigmoid(tf.matmul(x, variables_dict['encoder_w']) + variables_dict['encoder_b'])

def decoder(h):
    '''
    $$
    x = \sigma(HW + b)
    $$
    '''
    return tf.nn.sigmoid(tf.matmul(h, variables_dict['decoder_w']) + variables_dict['decoder_b'])

def jacobian(x, h):
    '''
    Jacobian of hidden layer with respect to input layer
    '''
    sum_W = tf.reduce_sum(tf.pow(tf.transpose(variables_dict["encoder_w"]), 2), axis=1)
    jacobian = tf.reduce_sum(tf.pow(h * (1 - h), 2) * sum_W)
    return jacobian

# Model
encode_X = encoder(X)
pred_X = decoder(encode_X)

loss = tf.reduce_sum(tf.pow(X - pred_X, 2))
gradient_loss = jacobian(X, encode_X)
loss += lamb * gradient_loss

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Initilization
    sess.run(init)

    # Train
    for i in range(1, num_steps + 1):
        batch_x, _ = mnist.train.next_batch(batch_size)

        _, error = sess.run([optimizer, loss], feed_dict={X:batch_x})

        if i % display_interval == 0:
            print("Step {step}, mnibatch error {error}".format(step = i, error = error))
    
    # Test
    n = 4
    canvas_orig = np.empty([28 * n, 28 * n])
    canvas_recv = np.empty([28 * n, 28 * n])

    # If use ANN, the width is 28, if use CNN, the width will be 32
    width = 28
    for i in range(n):
        batch_x, _ = mnist.test.next_batch(n)

        recv_x = sess.run(pred_X, feed_dict={X:batch_x})
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
    

