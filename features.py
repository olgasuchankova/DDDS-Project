# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 22:57:53 2021

@author: Lourdes

extract features from each data set
"""
import numpy as np
from statistics import mode
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
import collections

def get_time_features(self):
    mean = []
    minimum = []
    maximum = []
    median_ft = []
    std_ft = []
    # mode_ft = []
    
    for k in range(int(len(self)/15)):
        start = k*15
        stop = (k*15)+16
        for j in range(6):
            current_row = self[start:stop,j]
            mean.append(np.mean(self[start:stop,j]))
            minimum.append(np.min(self[start:stop,j]))
            maximum.append(np.max(self[start:stop,j]))
            median_ft.append(np.median(self[start:stop,j]))
            std_ft.append(np.std(self[start:stop,j]))
            # mode_ft1 = mode(self[start:stop,j]) 
            
            # if isinstance(mode_ft1,collections.Sequence):
            #     mode_ft1 = np.mean(mode_ft1)
                
            # mode_ft.append(mode_ft1)
    
    mean = np.reshape(mean,(k+1,j+1))
    minimum = np.reshape(minimum,(k+1,j+1))
    maximum = np.reshape(maximum,(k+1,j+1))
    median_ft = np.reshape(median_ft,(k+1,j+1))
    std_ft = np.reshape(std_ft,(k+1,j+1))
    # mode_ft = np.reshape(mode_ft,(k+1,j+1))
    time_stats = np.c_[mean, minimum, maximum, median_ft, std_ft]#, mode_ft]
    # print(np.shape(time_stats))

    print(time_stats.shape)
    
    return time_stats

def get_classification_features(X,y,num_feats):
    # define feature selection
    fs = SelectKBest(score_func=f_classif, k=num_feats)
    # apply feature selection
    X_selected = fs.fit_transform(X, y)
    
    return X_selected

def get_regression_features(X,y,num_feats):
    # define feature selection
    fs = SelectKBest(score_func=f_regression, k=num_feats)
    # apply feature selection
    X_selected = fs.fit_transform(X, y)
    
    return X_selected