# coding=utf-8
import tensorflow as tf

class Word2vec(object):
  def __init__(self, batch_size,\
    vocab_size,\
    embed_size):
    self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    self.center_words = tf.placeholder(tf.int32, shape=batch_size,\
      name='center_words')
    self.target_words = tf.placeholder(tf.int32, shape=[batch_size, 1],\
      name='target_words')
    
    self.embed_matrix = tf.Variable(tf.random_uniform([vocab_size, embed_size],\
      -1.0, 1.0), name='embed_matrix')
    self.embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words,\
      name='embed')