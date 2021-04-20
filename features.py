# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 22:57:53 2021

@author: Lourdes

extract features from each data set
"""
import numpy as np
from scipy.stats import mode

def get_time_features(self):
    mean = []
    minimum = []
    maximum = []
    median = []
    mode_ft = []
    
    for k in range(int(len(self)/15)):
        start = k*15
        stop = (k*15)+16
        for j in range(6):
            mean.append(np.mean(self[start:stop,j]))
            # minimum.append(np.min(self[start:stop,j]))
            # maximum.append(np.max(self[start:stop,j]))
            # median.append(np.median(self[start:stop,j]))            
            # mode_ft.append(mode(self[start:stop,j]))
    
    np.reshape(mean,(k+1,j+1))
    print(mean)
    # print(minimum)
    # print(maximum)
    # print(median)
    # print(mode)
    # time_stats = np.c_[mean, minimum, maximum, median, mode_ft]
    
    return mean