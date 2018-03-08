#!/usr/bin/python

import numpy as np
import pandas as pd
import multiprocessing
from functools import partial

class Node(object):
    def __init__(self, is_leaf=False, parent=None, cleft=None, cright=None,\
        feature_col=None, leaf_weight=None, split_cond=None, nan_dir=0):
        '''
        Tree node

        :param is_leaf: wether the node is a leaf node or not
        :param parent: the parent node
        :param cleft: the left child
        :param cright: the right child
        :param feature_col: the feature column
        :param leaf_weight: the leaf weight of node (only used when the node is a leaf node)
        :param split_cond: the split condition of node (only used when the node is a internal node)
        :param nan_dir: the direction of NaN data, 0 for left, 1 for right
        '''
        self.is_leaf = is_leaf
        self.parent = parent
        self.cleft = cleft
        self.cright = cright
        self.feature_col = feature_col
        self.leaf_weight = leaf_weight
        self.split_cond = split_cond
        self.nan_dir = nan_dir

class Tree(object):
    def __init__(self):
        self.root = None
        self.min_child_weight = None
        self.gamma = None
        self.lamb = None
        self.min_sample_split = None
        self.num_thread = None
        self.colsample_bylevel = None
    
    def calc_leaf_weight(self, Y):
        '''
        Calculate leaf weight
        '''
        return -Y['grad'].sum() / (Y['hess'].sum() + self.lamb)

    def find_cond(self, X, Y, feature_col):
        '''
        Find the split condition, NaN data direction and corresponding split
        gain on the specified feature

        Args:
            X: X data
            Y: Y data
            feature: split feature
        Returns:
            The feature
            The best split condition
            The best split gain
            The best NaN data direction
        '''
        best_cond = None
        best_gain = -float('inf')
        best_dir = 0

        data = pd.concat([X[feature_col], Y[['grad', 'hess']]], axis=1)

        G = data['grad'].sum()
        H = data['hess'].sum()
        
        # NaN data index
        NaN_ind = data[feature_col].isnull()

        # NaN data
        data_NaN = data[NaN_ind]

        # Not NaN data
        data_not_NaN = data[~NaN_ind]
        data_not_NaN.reset_index(inplace=True)

        ## TODO: precalculate the sort index for different column feature
        sort_index = data_not_NaN[feature_col].argsort()

        def calc_gain(G_L, G_R, G):
            return G_L ** 2 / (H_L + self.lamb) + G_R ** 2 / (H_R + self.lamb)\
                - G ** 2 / (H + self.lamb)

        G_L = 0
        H_L = 0
        for i in range(sort_index.shape[0] - 1):
            # Row index for the item before the split condition
            cur_ind = sort_index[i]
            # Row index for the item after the split condition
            next_ind = sort_index[i + 1]
            cond = (data_not_NaN[feature_col][cur_ind] +\
                    data_not_NaN[feature_col][next_ind]) / 2
            G_L += data_not_NaN['grad'][cur_ind]
            H_L += data_not_NaN['hess'][cur_ind]
            G_R = G - G_L
            H_R = H - H_L
            gain = calc_gain(G_L, G_R, G)
            if gain > best_gain:
                best_cond = cond
                best_gain = gain
                best_dir = 1
        G_R = 0
        H_R = 0
        for i in range(data_not_NaN.shape[0] - 1, -1, -1):
            # Row index for the item before the split condition
            cur_ind = sort_index[i]
            # Row index for the item after the split condition
            next_ind = sort_index[i + 1]
            cond = (data_not_NaN[feature_col][cur_ind] +\
                    data_not_NaN[feature_col][next_ind]) / 2
            G_R += data_not_NaN['grad'][cur_ind]
            H_R += data_not_NaN['hess'][cur_ind]
            G_L = G - G_R
            H_L = H - H_R
            gain = calc_gain(G_L, G_R, G)
            if gain > best_gain:
                best_cond = cond
                best_gain = gain
                best_dir = 0
        
        return feature_col, best_cond, best_gain, best_dir

    def find_feature_cond(self, X, Y):
        '''
        Find the split feature, split condition, split gain and the NaN
        data direction
        
        Args:
            X: X data
            Y: Y data
        Returns:
            The best split feature
            The best split condition
            The best split gain
            The best NaN data direction(0 for left, 1 for right)
        '''
        best_feature = 1
        best_cond = None
        best_gain = -float('inf')
        best_dir = 0
        reses = None

        func = partial(self.find_cond, X, Y)

        feature_cols = list(X.columns)

        #pool = multiprocessing.Pool(1 if self.num_thread == -1 else self.num_thread)
        #reses = pool.map(func, feature_cols)
        #pool.close()
        func(feature_cols[1])

        for res in reses:
            if reses[2] > best_gain:
                best_feature = res[0]
                best_cond = res[1]
                best_gain = res[2]
                best_dir = res[3]
        
        return best_feature, best_cond, best_gain, best_dir

    def split_data(self, X, Y, feature, condition, NaN_dir):
        '''
        split the data according to the feature and split condition, for
        NaN data, use the specified direction

        Args:
            X: X data
            Y: Y data
            feature: the split feature
            condition: the split condition
            NaN_dir: direction for NaN data
        Returns:
            Left X data
            Left Y data
            RIght X data
            Right Y data
        '''
        X_cols = list(X.columns)
        Y_cols = list(Y.columns)
        data = np.concat([X, Y], axis = 1)
        right_data = None
        left_data = None
        if NaN_dir == 0:
            mask = data[feature] >= condition
            right_data = data[mask]
            left_data = data[~mask]
        else:
            mask = data[feature] < condition
            left_data = data[mask]
            right_data = data[~mask]
        
        return left_data[X_cols], left_data[Y_cols], right_data[X_cols],\
            right_data[Y_cols]

    def build(self, X, Y, depth):
        '''
        Build the boost tree

        Args:
            X: X data
            Y: Y data
            depth: depth can be explore to from the current node
        Returns:
            A tree node
        '''
        leaf_weight = self.calc_leaf_weight(Y)
        if depth == 0 or X.shape[0] < self.min_sample_split:
            #leaf_weight <= self.min_child_weight:
            return Node(is_leaf=True, leaf_weight=leaf_weight)

        # Sample columns
        X_sample = X.sample(frac=self.colsample_bylevel, axis=1)

        best_feature, best_cond, best_gain, best_dir = \
            self.find_feature_cond(X_sample, Y)
        
        if best_gain < 0:
            return Node(is_leaf=True, leaf_weight=leaf_weight)

        X_left, Y_left, X_right, Y_right = self.split_data(X, Y, best_feature,\
            best_cond, best_dir)
        left_tree = self.build(X_left, Y_left, depth - 1)
        right_tree = self.build(X_right, Y_right, depth - 1)

        return Node(is_leaf=False, leaf_weight=None, feature_col=best_feature,\
            split_cond=best_cond, cleft=left_tree, cright=right_tree,\
            nan_dir=best_dir)

    def fit(self, X, Y, min_child_weight=1, gamma=0, lamb=1, max_depth=6,\
        min_sample_split=10, num_thread=-1, colsample_bylevel=1):
        '''
        Fit a tree to the given X/Y data

        Args:
            X: X data with type be pd.DataFrame
            Y: Y data with type be pd.DataFrame, with columns 'label', 'grad', 'hess'
            min_child_weight: min sum of weight of leaf node
            gamma: L1 regularizer
            lamb: L2 regularizer
            max_depth: max tree depth
            min_sample_split: minimum of sample to split
            num_thread: number of threads
            colsample_bylevel: columns sample percentage when building nodes
        '''
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.lamb = lamb
        self.min_sample_split = min_sample_split
        self.num_thread = num_thread
        self.colsample_bylevel = colsample_bylevel

        self.root = self.build(X, Y, max_depth)
    
    def _predict(self, node, X):
        '''
        Get the weight of a single data on the specified node
        Args:
            node: the specified node
            X: row data
        Returns:
            The predict value of the data
        '''
        if node.is_leaf:
            return node.leaf_weight
        elif pd.isnull(X[node.feature_col][1]):
            if node.nan_dir == 0:
                return self._predict(node.cleft, X)
            else:
                return self._predict(node.cright, X)
        elif X[node.feature_col][0] < node.split_cond:
            return self._predict(node.cleft, X)
        else:
            return self._predict(node.cright, X)
    
    def predict(self, X):
        '''
        Predict the data
        '''
        rows = X.iterrows()
        func = partial(self._predict, self.root)
        pool = multiprocessing.Pool(self.num_thread)
        preds = pool.map(func, rows)
        pool.close()

        return np.array(preds)

        
