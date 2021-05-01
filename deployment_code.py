import numpy as np
from sklearn.model_selection import train_test_split
import classifier as clfr
from data_compilation import features_loaded
import random

labels, temporal, aggregate = features_loaded(flat=False,f_type='time')

N_RUNS = 1
np.random.seed(1)
rand_seeds = np.random.randint(1e8,size=N_RUNS)
print(rand_seeds)

accuracy = []

for i, rand_seed in enumerate(rand_seeds):
    X_train, X_test, y_train, y_test = train_test_split(
        temporal, labels, test_size=0.15, random_state=rand_seed, shuffle=True)
    train_score, test_score, y_pred, gridsearchmodel = clfr.classify(
        X_train,X_test,y_train,y_test,False,"SVM")
    accuracy.append(test_score)

accuracy_agg = np.mean(accuracy)
print(accuracy_agg)