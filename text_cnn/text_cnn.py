#!/usr/bin/python
# coding=utf-8

from model import Model
import tensorflow as tf
import numpy as np

class TextCNN(object):
    def __init__(self,
        sequence_length,
        num_classes,
        vocab_size,
        embedding_size,
        filter_sizes,
        num_filters,
        l2_reg_lambda=0.0):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda

        self.model()
    
    def model(self):
        # Placeholders
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="W")
            self.embedding = tf.nn.embedding_lookup(self.W, self.input_x)
            # Add dimention to self.embedding
            self.embedding_expand = tf.expand_dims(self.embedding, -1)
        
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-{i}".format(i=i)):
                # Convolution layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters], name="b"))
                conv = tf.nn.conv2d(self.embedding_expand,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes) 
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        
        # Output layer
        with tf.name_scope("output"):
            W = tf.get_variable("W",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            # self.scores = tf.matmul(h_drop, W) + b
            self.scores = tf.nn.xw_plus_b(h_drop, W, b)
            self.predictions = tf.argmax(self.scores, 1, name="prediction")
        
        # Loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
        
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
        # Training
        # with tf.name_scope("training"):
        #     self.global_step = tf.Variable(0, name="global_step", trainable=False)
        #     self.optimizer = tf.train.AdamOptimizer(1e-3)
        #     self.train_op = self.optimizer.minimize(self.loss, self.global_step)
        

