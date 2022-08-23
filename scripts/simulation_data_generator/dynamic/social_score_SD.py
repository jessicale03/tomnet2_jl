#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get social score SD

@author: elaine
""" 

import numpy as np
import os
import pandas as pd
import csv


DIR_ROOT = os.getcwd()
DIR_CSV_OUTPUT = os.path.join(DIR_ROOT,'SD_summary_1.csv')
df_SD_collect = pd.DataFrame()
for n in range (1,2):
	all_values = np.zeros((5,5),dtype=np.float64)
	FNAME = os.path.join(DIR_ROOT,'dS00'+str(n)+'.csv')
	with open(FNAME, encoding='utf-8') as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		social_score=list(readCSV)
	for i in range (5):
		for j in range (5):
			all_values[i][j]=social_score[i+1][j+1]
	SD = np.std(all_values)
	Var = np.var(all_values)
	#print(SD)
	#SD = SD.astype(str)
	df_SD = pd.DataFrame(data = {'set_n': [n],\
									  'SD': [SD],\
									  'Var':[Var]})
	df_SD_collect = df_SD_collect.append(df_SD)
df_SD_collect.to_csv(DIR_CSV_OUTPUT)
