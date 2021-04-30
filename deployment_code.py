# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 23:30:39 2021

@author: Lourdes
"""
import numpy as np
from sklearn.model_selection import train_test_split
import classifier as clfr
from data_compilation import features_loaded


labels, temporal, aggregate = features_loaded(flat=False,f_type='time')


X_train, X_test, y_train, y_test = train_test_split(
    aggregate, labels, test_size=0.15, shuffle=True)


train_score, test_score, y_pred, gridsearchmodel = clfr.classify(X_train,X_test,y_train,y_test)
