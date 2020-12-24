#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generating social range

@author: elaine
""" 
import random
import statistics as stat
import numpy as np


    
q = 0
while q < 1:
    a = []
    for i in range (20):
        atemp = random.randint(-30,30)
        a = np.append(a,atemp)
        i = i+1
    amean = stat.mean(a)
    if amean != 0:
        q = 0
    else:
        q = q+1
        print(a)
        
        

