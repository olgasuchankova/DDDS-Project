# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 23:30:39 2021

@author: Lourdes
"""
import numpy as np
import csv
from features import get_time_features as gtf
from sklearn import train_test_split
import classifier as clfr

file = open('practicedata.csv')
numpy_array = np.loadtxt(file,delimiter=',')

time_features = []
time_features = gtf(numpy_array)

labels=[]
X_train, X_test, y_train, y_test = train_test_split(
    time_features, labels, test_size=0.15, shuffle=False)
# with open('practicedata.csv','r') as file:
#     reader = csv.reader(file,quoting=csv.QUOTE_NONNUMERIC)
#     for row in reader:
#         print(row)
        
    # data = gtf(reader)