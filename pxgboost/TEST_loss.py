#!/usr/bin/python
# coding=utf-8

import numpy as np
from loss import *

def test_SquareLoss():
    preds = np.array([10.0,7.0,17.0,15.0])
    labels = np.array([10.0,10.0,10.0,10.0])
    loss_obj = SquareLoss()
    print(loss_obj.grad(preds, labels))
    print(loss_obj.hess(preds, labels))


def test_LogisticLoss():
    preds = np.array([0.1,0.9,0.5,0.8])
    labels = np.array([0,0,1,1])
    loss_obj = LogisticLoss()
    print(loss_obj.grad(preds, labels))
    print(loss_obj.hess(preds, labels))

