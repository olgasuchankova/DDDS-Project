# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 22:12:04 2021

data compilation

@author: Lourdes
"""

import csv

with open('TempvsGasUsage.data','r') as csvfile:
    reader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC)
    for row in reader
           month.append(row[0])
           cooling.append(row[1])
           heating.append(row[2])
           max_temp.append(row[3])
           min_temp.append(row[4])
           mean_temp.append(row[5])
           gas_usage.append(row[7])
