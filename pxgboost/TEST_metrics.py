#!/usr/bin/python
# coding=utf-8

import numpy as np
from metrics import AUC

def test_auc():
    preds = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([1, 1, 2, 2])
    auc = AUC(preds, labels)
    print(auc)

test_auc()