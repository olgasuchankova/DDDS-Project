# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 22:12:04 2021

data compilation

@author: Lourdes
"""
import csv


for k in range(1,6):
    filename = 'lp'+str(k)+'.csv'
    with open(filename,'r') as file:   
        reader = csv.reader(file)
        for row in reader:
            print(row)
