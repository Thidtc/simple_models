#!/usr/bin/python
# coding=utf-8
import numpy as np

class Dataset(object):
    '''
    datasets
    '''
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.val_data = None
        self.val_label = None
        self.test_data = None
        self.test_label = None

        self._index_in_epoch = 0
        self._epoch_completed = 0
    
    def next_batch(self, batch_size, data_type='train', shuffle=True):
        '''
        Return the 'batch_size' examples from the dataset
        '''
        start = self._index_in_epoch
        if data_type == 'train':
            data = self.train_data
            labels = self.train_label
            num_data = len(self.train_data)
        elif data_type == 'val':
            data = self.val_data
            labels = self.val_label
            num_data = len(self.val_data)
        elif data_type == 'test':
            data = self.test_data
            labels = self.test_label
            num_data = len(self.test_data)
         
        if self._epoch_completed == 0 and start == 0 and shuffle:
            self.perm0 = np.arange(num_data)
            np.random.shuffle(self.perm0)
        if start + batch_size > num_data:
            self._epoch_completed += 1
            rest_num = num_data - start
            index = self.perm0[start:num_data]
            data_rest_part = data[index]
            labels_rest_part = labels[index]
            if shuffle:
                np.random.shuffle(self.perm0)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num
            index = self.perm0[0:self._index_in_epoch]
            data_new_part = data[index]
            label_new_part = labels[index]
            return np.concatenate((data_rest_part, data_new_part), axis=0),\
                np.concatenate((labels_rest_part, label_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            index = self.perm0[start:end]
            return data[index], labels[index]
        
        

