import csv
import numpy as np
import json
from features import get_time_features as gtf
from features import get_classification_features as gcf
from features import get_regression_features as grf
# import math

num_datasets = 1
time_series_length = 15
num_forces = 6


def _load_data3d():
    data_matrix = []
    raw_csv_read = []
    pntr = 0
    for k in range(num_datasets, num_datasets + 1):
        class_type = {}
        with open('datafolder/lp' + str(k) + '.json') as json_file:
            class_type = json.load(json_file)

        filename = 'datafolder/lp' + str(k) + '.csv'
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                raw_csv_read.append(row)
                classification = row[0]
                if classification != "":
                    if (class_type.get(classification) == None):
                        print(classification)
                    insert_row = [class_type.get(classification), k, [], [], [], [], [], []]
                    data_matrix.append(insert_row)
                    pntr = pntr + 1
                if row[1] != "":
                    for i in range(2, 8):
                        (data_matrix[pntr - 1])[i].append(float(row[i - 1]))
                # print(row)

    arrx = np.array(data_matrix, dtype=object)
    return arrx


def _clean_3d():
    matrix = _load_data3d()

    labels = np.array(matrix[:, 0], dtype=int)
    num_entries = matrix.shape[0]
    num_signals = matrix.shape[1] - 2
    ret_matrix = np.zeros((num_entries, num_signals, time_series_length))

    for i in range(num_entries):
        for j in range(num_signals):
            ret_matrix[i, j, :] = np.array(matrix[i, j])

    return labels, ret_matrix

def _reshape(clean_data):
    ret = np.zeros((clean_data.shape[0], num_forces*time_series_length))
    for i in range(clean_data.shape[0]):
        ret[i] = clean_data[i].flatten()
    return ret


def _flat_labels(labels, n=time_series_length):
    flat_label_m = np.zeros((len(labels) * n))

    for i in range(len(labels)):
        flat_label_m[i * n:(i + 1) * n] = labels[i] * np.ones((n))

    return flat_label_m


def _load_data2d():
    data_matrix = []
    raw_csv_read = []
    pntr = 0
    for k in range(num_datasets, num_datasets + 1):
        filename = 'datafolder/lp' + str(k) + '.csv'
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                raw_csv_read.append(row)
                classification = row[0]
                if row[1] != "":
                    insert_row = [[], [], [], [], [], []]
                    data_matrix.append(insert_row)
                    pntr = pntr + 1
                    for i in range(2, 8):
                        (data_matrix[pntr - 1])[i - 2].append(float(row[i - 1]))
                elif row[1] != "":
                    for i in range(2, 8):
                        (data_matrix[pntr - 1])[i].append(float(row[i - 1]))
                # print(row)
    arrx = np.array(data_matrix)
    return arrx[:, :, 0]


def features_loaded(flat=False,f_type='time', rshape=False):
    timedat = _load_data2d()     # forcewise
    labels, dataset = _clean_3d() # short labels, 2d x training data (3d matrix)

    if flat:
        labels = _flat_labels(labels)
        dataset = timedat
        
    if rshape:
        dataset = _reshape(dataset)    
    
    features = []

    if f_type == 'time':
        features = gtf(timedat)
    elif f_type == 'classification':
        features = gcf(timedat,labels,4)
    elif f_type == 'regression':
        features = grf(timedat,labels,4)
    
    return labels, dataset, features


_clean_3d()
