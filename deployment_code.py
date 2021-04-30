# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 23:30:39 2021

@author: Lourdes
"""
import numpy as np
import csv
from features import get_time_features as gtf
from sklearn.model_selection import train_test_split
import classifier as clfr
from data_compilation import features_loaded

file = open('practicedata.csv')
numpy_array = np.loadtxt(file,delimiter=',')

load_preprocessed_data = features_loaded()

time_features = []
num_time_feat = 4
time_features = load_preprocessed_data[:, (time_features.shape[1] - num_time_feat):]

labels=[]
labels = load_preprocessed_data[0]
X_train, X_test, y_train, y_test = train_test_split(
    time_features, labels, test_size=0.15, shuffle=False)


# with open('practicedata.csv','r') as file:
#     reader = csv.reader(file,quoting=csv.QUOTE_NONNUMERIC)
#     for row in reader:
#         print(row)
        
    # data = gtf(reader)