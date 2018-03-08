#!/usr/bin/python
# coding=utf-8

'''
Variational Autoencoder
references:
    * Tutorial on Variational Autoencoders
'''

import numpy as np
import tensorflow as tf
import cPickle as pickle
from skimage.io import imsave

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Constants
num_steps = 100000
learning_rate = 0.01
batch_size = 128
display_interval = 1000

num_input = 784
num_hidden = 128
num_noise = 100

# Placeholders
X = tf.placeholder(tf.float32, [None, num_input])


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# \mu(X)
def get_mu(X):
    with tf.variables_scope("mu"):
        W1 = tf.Variable(xavier_init([num_input, num_hidden]), dtype=tf.float32, name="W1")
        b1 = tf.Variable(tf.zeros([num_hidden]), name="b1")
        W2 = tf.Variable(xavier_init([num_hidden, num_noise]), dtype=tf.float32, name="W2")
        b2 = tf.Variable(tf.zeros([num_noise]), name="b2")

        layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
        layer2 = tf.matmul(layer1, W2) + b2

        return layer2

# \Sigma(X)
def get_sigma(X):
    with tf.variables_scope("sigma"):
        W1 = tf.Variable(xavier_init([num_input, num_hidden]), dtype=tf.float32, name="W1")
        b1 = tf.Variable(tf.zeros([num_hidden]), name="b1")
        W2 = tf.Variable(xavier_init([num_hidden, num_noise]), dtype=tf.float32, name="W2")
        b2 = tf.Variable(tf.zeros([num_noise]), name="b2")

        layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
        layer2 = tf.matmul(layer1, W2) + b2

        return layer2

def decoder(z):
    with tf.variables_scope("decoder"):
        W1 = tf.Variable(xavier_init([num_noise, num_hidden]), dtype=tf.float32, name="W1")
        b1 = tf.Variable(tf.zeros([num_hidden]), name="b1")
        W2 = tf.Variable(xavier_init([num_hidden, num_input]), dtype=tf.float32, name="W2")
        b2 = tf.Variable(tf.zeros([num_input]), name="b2")

        layer1 = tf.nn.relu(tf.matmul(z, W1) + b1)
        layer2 = tf.matmul(layer1, W2) + b2

        return layer2


mu = get_mu(X)
log_sigma = get_sigma(X)

kl_loss = 0.5 * tf.reduce_sum(tf.exp(log_sigma) + tf.pow(mu, 2) - log_sigma - 1.0, reduction_indices=1)

epsilon = tf.random_normal(tf.shape(mu), name='epsilon')

z = mu + tf.exp(0.5 * log_sigma) * epsilon

reconstructed_X = decoder(z)

reconstructed_loss = tf.reduce_sum(tf.nn.sigmoid_entropy_with_logits(logits=reconstructed_X, labels=X), reduction_indices=1)

vae_loss = tf.reduce_mean(reconstructed_loss + kl_loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(vae_loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Train
    for i in range(1, 1 + num_steps):
        batch_x, _ = mnist.train.next_batch(batch_size)
        _, loss = sess.run([optimizer, vae_loss], feed_dict={X:batch_x})

        if i % display_interval == 0 or i == 1:
            print("epoch {epoch}, error {error}".format(epoch=i, error=loss))
    # Test
    n = 4
    canvas_orig = np.empty([28 * n, 28 * n])
    canvas_rec = np.empty([28 * n, 28 * n])

    for i in range(n):
        batch_x, _ = mnist.test.next_batch(n)
        new_x = sess.run([reconstructed_X], feed_dict={X:batch_x})

        for j in range(n):
            canvas_orig[i*28:(i+1)*28, j*28:(j+1)*28] = batch_x.reshape(28, 28)
            canvas_rec[i*28:(i+1)*28, j*28:(j+1)*28] = new_x.reshape(28, 28)
    
    # Change from [0, 1] to [0, 255]
    canvas_orig = canvas_orig * 255
    canvas_orig = canvas_orig.astype('int')
    canvas_rec = canvas_rec * 255
    canvas_rec = canvas_rec.astype('int')

    pickle.dump(canvas_orig, open('orig.pkl', 'w'))
    pickle.dump(canvas_rec, open('rec.pkl', 'w'))
    imsave('orig.bmp', canvas_orig)
    imsave('rec.bmp', canvas_rec)