#!/usr/bin/python
# coding=utf-8
import pandas as pd
import numpy as np
from multiprocessing import Pool
from loss import *
from metrics import get_metric
from tree_model import Tree

class PXGBoost(object):
    def __init__(self):
        self.trees = []
        self.eta = None
        self.min_child_weight = None
        self.max_depth = None
        self.gamma = None
        self.max_delta_step = None
        self.subsample = None
        self.colsample_bytree = None
        self.colsample_bylevel = None
        self.lamb = None
        self.scale_pos_weight = None
        self.init_predict = None
        self.loss = None
        self.eval_metric = None
        self.silence = None
        self.num_thread = None
        self.boost_round = None
    
    def fit(self, dtrain, dval=None, eta=0.3, min_child_weight=1,\
        max_depth=6, gamma=0, max_delta_step=0, subsample=1, colsample_bytree=1,\
        colsample_bylevel=1, lamb=1, scale_pos_weight=1,\
        objective="square", eval_metric=None, seed=0, silence=0, num_thread=1,\
        boost_round=1000):
        '''
        Fit the model to the data

        Args:
            dtrain: train data
            dval: valication data
            eta: learning rate
            min_child_weight: mean sum of weight of leaf node
            max_depth: max tree depth
            gamma: L1 regularizer
            max_delta_step: 
            subsample: percentage of subsampling
            colsample_bytree: columns sample percentage in tree build
            colsample_bylevel: columns sample percentage in tree node create
            lamb: L2 regularizer
            scale_pos_weight:
            objective: loss function
            eval_metric: error metric
            silence: voberse
            num_thread: number of threads
            boost_round: nubmer of boost round
        '''
        self.eta = eta
        self.min_child_weight = min_child_weight
        self.max_depth = max_depth
        self.gamma = gamma
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.lamb = lamb
        self.scale_pos_weight = scale_pos_weight
        if objective == 'square':
            self.loss = SquareLoss(lamb)
        else:
            raise NotImplementedError()
        self.eval_metric = eval_metric
        self.silence = silence
        self.num_thread = num_thread
        self.boost_round = boost_round
        

        dtrain.reset_index(drop=True, inplace=True)
        Ytrain = dtrain[['label']]
        Xtrain = dtrain.drop(['label'], axis=1)

        # Initialize predictions
        self.init_predict = Ytrain['label'].mean()

        # Wether do validation or not
        b_validation = dval != None
        if b_validation:
            # Valication check
            dval.reset_index(drop=True, inplace=True)
            Yval = dval['label']
            Xval = dval.drop(['label'], axis=1)
            Yval['pred'] = self.init_predict

        Ytrain['pred'] = self.init_predict
        
        for i in range(boost_round):
            # Calculate gradient and hessian matrix
            Ytrain['grad'] = self.loss.grad(Ytrain['pred'], Ytrain['label'])
            Ytrain['hess'] = self.loss.hess(Ytrain['pred'], Ytrain['label'])

            # Column sample
            data = Xtrain.sample(frac=self.colsample_bytree, axis=1)
            data = pd.concat([data, Ytrain], axis=1)
            data = data.sample(frac=self.subsample, axis=0)

            Y = data[['label', 'pred', 'grad', 'hess']]
            X = data.drop(['label', 'pred', 'grad', 'hess'], axis=1)

            tree = Tree()
            tree.fit(X, Y)

            # Predict the train set
            Ytrain['pred'] += self.eta * tree.predict(X)

            self.trees.append(tree)

            # Training information
            if self.eval_metric == None:
                print("XGBoost iteration-{iter}".format(iter=i))
            else:
                try:
                    metric_func = get_metric(self.eval_metric)
                except Exception, e:
                    raise NotImplementedError()
                train_metric = metric_func(self.loss.transform(Y['pred']), Y['label'])
            
            if not b_validation:
                print("XGBoost iteration-{iter}, Eval metric {eval_metric}-{train_metric}"\
                    .format(iter=i, eval_metric=self.eval_metric, train_metric=train_metric))
            else:
                Yval['pred'] += tree.predict(X)
                val_metric = metric_func(self.loss.transform(Y['pred']), Y['label'])
                print("XGBOOST iteration-{iter}, Eval metric {eval_metric}-{train_metric}, Validation metric {eval_metric}-{val_metric}"\
                    .format(iter=1, eval_metric=self.eval_metric, train_metric=train_metric,\
                    val_metric=val_metric))

    def predict(self, X):
        assert len(self.trees) > 0
        pred = np.zeros((X.shape[0], 0))
        pred += self.init_predict
        pool = Pool(processes=self.num_thread)

        results = []
        for tree in self.trees:
            result = pool.map_async(tree.predict, X)
            results.append(result)
        
        for result in results:
            pred += self.eta * result.get()
        
        return pred
