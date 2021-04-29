# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 22:57:53 2021

@author: Lourdes

extract features from each data set
"""
import numpy as np
from statistics import mode

def get_time_features(self):
    mean = []
    minimum = []
    maximum = []
    median_ft = []
    mode_ft = []
    
    for k in range(int(len(self)/15)):
        start = k*15
        stop = (k*15)+16
        for j in range(6):
            mean.append(np.mean(self[start:stop,j]))
            minimum.append(np.min(self[start:stop,j]))
            maximum.append(np.max(self[start:stop,j]))
            median_ft.append(np.median(self[start:stop,j]))            
            mode_ft.append(mode(self[start:stop,j]))
    
    mean = np.reshape(mean,(k+1,j+1))
    minimum = np.reshape(minimum,(k+1,j+1))
    maximum = np.reshape(maximum,(k+1,j+1))
    median_ft = np.reshape(median_ft,(k+1,j+1))
    mode_ft = np.reshape(mode_ft,(k+1,j+1))
    time_stats = np.c_[mean, minimum, maximum, median_ft, mode_ft]
    # print(np.shape(time_stats))
    
    return time_stats