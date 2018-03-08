#!/usr/bin/python
# coding=utf-8

import tensorflow as tf

class Model(object):
    '''
    An abstract model for Tensorflow
    '''

    def __init__(self, *dictArg):
        '''
        Model hyperparameters initialization
        '''
        self.n_steps = 10000
        self.batch_size = 128
        self.checkpoint_interval = 1000

        for key, value in dictArg:
            self.key = value

        self.model_name = None
        # The most recent checkpoint file
        self.recent_store_file = None    

    def load(self, sess, model_checkpoint_path):
        '''
        Load the model

        Args:
            sess: the session to be loaded
            model_checkpoint_path: the path of the model checkpoint
        '''
        saver = tf.train.Saver()
        saver.restore(sess, model_checkpoint_path)
    
    def store(self, sess, model_checkpoint_path, global_step):
        '''
        Store the model

        Args:
            sess: the current session to be stored
            model_checkpoint_path: the path of the model checkpoint
            global_step: current global step
        '''
        saver = tf.train.Saver()
        self.recent_store_file = saver.save(sess, model_checkpoint_path, global_step=global_step)
    
    def model(self):
        '''
        Build the model

        Hint: store the needed operation in the class member
                for example
                self.loss = tf.loss_function(self.y_predict, self.y_label)
        '''
        # Placeholders
        # Model
        raise NotImplementedError('build not implemented')
    
    def train(self, sess):
        '''
        Train the model

        Args:
            sess: the running session
        '''
        raise NotImplementedError('train not implemented')

    def inference(self, sess):
        '''
        Inference on the model

        Args:
            sess: the running session
        '''