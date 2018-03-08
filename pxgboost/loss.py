#!/usr/bin/python
# coding=utf-8

import numpy as np

class BaseLoss(object):
    def __init__(self, lamb=0.0):
        self.lamb = lamb
    
    def grad(self, preds, labels):
        raise NotImplementedError()
    
    def hess(self, preds, labels):
        raise NotImplementedError()

class SquareLoss(BaseLoss):
    def grad(self, preds, labels):
        return preds - labels
    
    def hess(self, preds, labels):
        return np.ones_like(labels)
    
    def transform(self, preds):
        return preds

class LogisticLoss(BaseLoss):
    def grad(self, preds, labels):
        preds = self.transform(preds)
        return (1 - labels) / (1 - preds) - labels / preds
    
    def hess(self, preds, labels):
        preds = self.transform(preds)
        return labels / np.square(preds) + (1 - labels) / np.square(1 - preds)

    def transform(self, preds):
        return 1.0 / (1.0 + np.exp(-preds))