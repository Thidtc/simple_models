#!/urs/bin/python
#coding=utf-8

import pandas as pd
from pxgboost import PXGBoost

train = pd.read_csv('train.csv')
val = train.iloc[0:5000]
train = train.iloc[5000:]


params = {
    'objective': "square",
    'eta': 0.3,
    'max_depth': 6,
    'boost_round': 500,
    'scale_pos_weight': 1.0,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 1.0,
    'min_child_weight': 2,
    'lamb': 10,
    'gamma': 0,
    'eval_metric': "mse", # "mse"
    'num_thread': 16
}

gbm = PXGBoost()
gbm.fit(train, **params)
