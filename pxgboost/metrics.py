#!/usr/bin/python
# coding=utf-8

import numpy as np

def accuracy(preds, labels):
    return np.mean(round(preds) == labels)

def error(preds, labels):
    return 1 - accuracy(preds, labels)

def MSE(preds, labels):
    '''
    Mean square error
    '''
    return np.mean(np.square(preds - labels))

def MAE(preds, labels):
    '''
    Mean absolute error
    '''
    return np.mean(np.abs(preds - labels))

def AUC(preds, labels):
    '''
    AUC value
    '''
    def rank(x):
        '''
        Get the ranks of the predictions.
        Note, that the rank will be a float value instead of integer value,
        and items with same score may have different rank.
        But in all
            score_i <= score_j    ->     rank_i < rank_j

        Args:
            x: the predictions
        Returns:
            the rank list
        '''
        # Sort x and attach each item with its index, so that
        # the index can be refered as sorted[i][1]
        sorted_x = sorted(zip(x, range(len(x))))

        r = [0 for k in x]
        cur_val = sorted_x[0][0]
        last_rank = 0
        for i in range(len(sorted_x)):
            if cur_val != sorted_x[i][0]:
                cur_val = sorted_x[i][0]
                for j in range(last_rank, i):
                    # assign rank to the score
                    r[sorted_x[j][1]] = (last_rank + i + 1) / 2.0
                last_rank = i
            if i == len(sorted_x) - 1:
                for j in range(last_rank, i + 1):
                    r[sorted_x[j][1]] = (last_rank + i + 2) / 2.0
        return r

    r = rank(preds)

    num_positive = len([0 for x in labels if labels[x] == 1])
    num_negative = len(labels) - num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if labels[i] == 1])
    
    auc = ((sum_positive - num_positive * (num_positive + 1) / 2.0) /\
        (num_negative * num_positive))
    
    return auc


metrics = {\
    "acc":accuracy,\
    "err":error,\
    "MSE":MSE,\
    "MAE":MAE\
}

def get_metric(metric_name):
    '''
    Get the specified metric function
    Returns:
        the metric function
    '''
    return metrics[metric_name]