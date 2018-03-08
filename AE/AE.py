#!/usr/bin/python
# coding=utf-8

'''
AutoEncoder
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
display_interval = 1000

num_hidden1 = 256
num_hidden2 = 128
pic_width = 28
num_input = pic_width * pic_width

# Placeholders
# For ANN
X = tf.placeholder(tf.float32, [None, num_input])

# Variables
# Variables for ANN
variables_dict = {
    "encoder_w1": tf.Variable(tf.random_normal([num_input, num_hidden1]), dtype=tf.float32),
    "encoder_w2": tf.Variable(tf.random_normal([num_hidden1, num_hidden2]), dtype=tf.float32),
    "encoder_b1": tf.Variable(tf.random_normal([num_hidden1]), dtype=tf.float32),
    "encoder_b2": tf.Variable(tf.random_normal([num_hidden2]), dtype=tf.float32),
    "decoder_w1": tf.Variable(tf.random_normal([num_hidden2, num_hidden1]), dtype=tf.float32),
    "decoder_w2": tf.Variable(tf.random_normal([num_hidden1, num_input]), dtype=tf.float32),
    "decoder_b1": tf.Variable(tf.random_normal([num_hidden1]), dtype=tf.float32),
    "decoder_b2": tf.Variable(tf.random_normal([num_input]), dtype=tf.float32)
}

def encoder(x):
    '''
    Encoder
    '''
    layer1 = tf.nn.sigmoid(tf.matmul(x, variables_dict["encoder_w1"]) + variables_dict["encoder_b1"])
    layer2 = tf.nn.sigmoid(tf.matmul(layer1, variables_dict["encoder_w2"]) + variables_dict["encoder_b2"])
    return layer2

def decoder(x):
    '''
    Decoder
    '''
    layer1 = tf.nn.sigmoid(tf.matmul(x, variables_dict["decoder_w1"]) + variables_dict["decoder_b1"])
    layer2 = tf.nn.sigmoid(tf.matmul(layer1, variables_dict["decoder_w2"]) + variables_dict["decoder_b2"])
    return layer2


# Model

# Fead-Forward NN
encode_X = encoder(X)
pred_X = decoder(encode_X)

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

        _, error = sess.run([optimizer, loss], feed_dict={X:batch_x})

        if i % display_interval == 0:
            print("Step {step}, minibatch error {error}".format(step=i, error=error))
    
    # Test
    n = 4
    width=28
    canvas_orig = np.empty([28 * n, 28 * n])
    canvas_rec = np.empty([28 * n, 28 * n])

    for i in range(n):
        batch_x, _ = mnist.test.next_batch(n)

        rec_x = sess.run(pred_X, feed_dict={X:batch_x})
        for j in range(n):
            canvas_orig[i*width:(i+1)*width, j*width:(j+1)*width] = batch_x[j].reshape([width, width])
            canvas_rec[i*width:(i+1)*width, j*width:(j+1)*width] = rec_x[j].reshape([width, width])
    
    # Change from [0, 1] to [0, 255]
    canvas_orig = canvas_orig * 255
    canvas_orig = canvas_orig.astype('int')
    canvas_rec = canvas_rec * 255
    canvas_rec = canvas_rec.astype('int')

    pickle.dump(canvas_orig, open('orig.pkl', 'w'))
    pickle.dump(canvas_rec, open('rec.pkl', 'w'))
    imsave('orig.bmp', canvas_orig)
    imsave('rec.bmp', canvas_rec)

