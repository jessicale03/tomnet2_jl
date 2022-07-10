#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generating social range

@author: elaine
""" 
import random
import statistics as stat
import numpy as np
import os
import pandas as pd

FNAME = 'dS00'

DIR_CSV_OUTPUT = os.getcwd()
for n in range(5,105):    
    flag = 0
    while flag < 1:
        social_score = []
        for i in range (20):
            social_score_next = random.randint(-30,30)
            social_score = np.append(social_score,social_score_next)
            i = i+1
        mean = stat.mean(social_score)
        if mean != 0:
            flag = 0
        else:
            flag+=1
           # print(social_score)
            
    social_score = np.insert(social_score,(0,5,10,15,20),0)
    social_score = social_score.reshape(5,5)
    social_score = social_score.astype(int)
    
    df_social_score = pd.DataFrame(data = {" ":["S","A","B","C","D"],\
                                                          "S":social_score[0],\
                                                          "A":social_score[1],\
                                                          "B":social_score[2],\
                                                          "C":social_score[3],\
                                                          "D":social_score[4]})
    try:
        output = os.path.join(DIR_CSV_OUTPUT,str(FNAME)+str(n)+".csv")
        df_social_score.to_csv(output,index=False)
    except:print('error')
    n+=1

