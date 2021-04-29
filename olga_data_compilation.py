import csv
import numpy as np
import json

def load_data3d():
    class_type = {}
    with open('datafolder/classtypes.json') as json_file:
        class_type = json.load(json_file)

    data_matrix = []
    raw_csv_read = []
    pntr = 0
    for k in range(1,6):
        filename = 'datafolder/lp'+str(k)+'.csv'
        with open(filename,'r') as file:
            reader = csv.reader(file)
            for row in reader:
                raw_csv_read.append(row)
                classification = row[0]
                if classification != "":
                    if(class_type.get(classification) == None):
                        print(classification)
                    insert_row = [class_type.get(classification), k, [], [], [], [], [], []]
                    data_matrix.append(insert_row)
                    pntr = pntr + 1
                if row[1] != "":
                    for i in range(2,8):
                        (data_matrix[pntr-1])[i].append(float(row[i-1]))
                #print(row)


    arrx = np.array(data_matrix)
    print(arrx)
    return arrx


def load_data2d():
    data_matrix = []
    raw_csv_read = []
    pntr = 0
    for k in range(1, 6):
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
                        (data_matrix[pntr - 1])[i-2].append(float(row[i-1]))
                elif row[1] != "":
                    for i in range(2, 8):
                        (data_matrix[pntr - 1])[i].append(float(row[i - 1]))
                # print(row)
    arrx = np.array(data_matrix)
    return arrx[:,:,0]

load_data2d()