#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 22:05:14 2018

@author: sunan
"""

import scipy.sparse as sp
import numpy as np
from implicit_mf import *
import matplotlib.pyplot as plt

sample_matrix = sp.load_npz('sample_matrix.npz')

sample_array = sample_matrix.toarray()
#Take a look at the sample sparsity
sparsity = float(len(sample_array.nonzero()[0]))
sparsity /= (sample_array.shape[0] * sample_array.shape[1])
sparsity *= 100
print ('Sparsity: {:4.2f}%'.format(sparsity))

def train_test_split(data):
    #data should be ndarray format
    test = np.zeros(data.shape)
    train = data.copy()
    for user in range(data.shape[0]):
        test_index = np.random.choice(data[user, :].nonzero()[0], 
                                        size=10, 
                                        replace=False)
        train[user, test_index] = 0.
        test[user, test_index] = data[user, test_index]
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

train_array, test_array = train_test_split(sample_array)
train = sp.csr_matrix(train_array)
test = sp.csr_matrix(test_array)

##train model
#mf = ImplicitMF(train, alpha=5, num_factors=5, num_iterations=1,
#                 reg_param=0.8)
#mf.fit()
#mf.evaluate(test)
#
#
#
#num_factors = [5, 10, 20, 40]
#regularizations = [0.01, 0.1, 1, 10]
#
#best_params = {}
#best_params['num_factors'] = num_factors[0]
#best_params['regularization'] = regularizations[0]
#best_params['model'] = None
#min_ = np.inf
#
#for num in num_factors:
#    print ('num_factors: {}'.format(num))
#    for reg in regularizations:
#        print ('regularization: {}'.format(reg))
#        mf = ImplicitMF(train, num_factors=num, num_iterations=15, reg_param=reg)
#        mf.fit()        
#        score = mf.evaluate(test)
#        print ('score: {}'.format(score))
#        if score < min_:
#            best_params['num_factors'] = num
#            best_params['regularization'] = reg
#            best_params['model'] = mf
#            min_ = score
        
#{'model': <implicit_mf.ImplicitMF at 0x181047d4e0>,
# 'num_factors': 40,
# 'regularization': 0.01}

num_factors = [20, 40, 60]
regularizations = [0.01, 0.1, 1]
alpha = [0.1, 1, 10, 20]

best_params = {}
best_params['alpha'] = alpha[0]
best_params['num_factors'] = num_factors[0]
best_params['regularization'] = regularizations[0]
best_params['model'] = None
min_ = np.inf

for num in num_factors:
    print ('num_factors: {}'.format(num))
    for reg in regularizations:
        print ('regularization: {}'.format(reg))
        for al in alpha:
            print ('alpha: {}'.format(al))
            mf = ImplicitMF(train, alpha=al, num_factors=num, num_iterations=10, reg_param=reg)
            mf.fit()        
            score = mf.evaluate(test)
            print ('score: {}'.format(score))
            if score < min_:
                best_params['num_factors'] = num
                best_params['regularization'] = reg
                best_params['alpha'] = al
                best_params['model'] = mf
                min_ = score
print ('best parameters: {}'.format(best_params))
print ('best score: {}'.format(min_))






    
