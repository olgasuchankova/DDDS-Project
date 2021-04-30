# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 23:30:39 2021

@author: Lourdes
"""
import numpy as np
import csv
from features import get_time_features as gtf
from features import get_classification_
from sklearn.model_selection import train_test_split
import classifier as clfr
from data_compilation import features_loaded

# from features import get_classification_features as gcf

file = open('practicedata.csv')
numpy_array = np.loadtxt(file,delimiter=',')


num_feat = 4
labels, temporal, aggregate = features_loaded(flat=False,f_type='time',num_feats=num_feat)


X_train, X_test, y_train, y_test = train_test_split(
    aggregate, labels, test_size=0.15, shuffle=False)

train_score, test_score, y_pred, gridsearchmodel = clfr.classify(X_train,X_test,y_train,y_test,"Support Vector Machine (SVM)")

# with open('practicedata.csv','r') as file:
#     reader = csv.reader(file,quoting=csv.QUOTE_NONNUMERIC)
#     for row in reader:
#         print(row)
        
    # data = gtf(reader)