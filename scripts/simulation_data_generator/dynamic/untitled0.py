#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 19:14:27 2020

@author: elaine
"""

import numpy as np
import random

x_random=random.sample(range(1,12), (5))
y_random=random.sample(range(1,12), (5))
p_random=random.sample(range(1,12), (5))
q_random=random.sample(range(1,12), (5))
r_random=random.sample(range(1,12), (5))
s_random=random.sample(range(1,12), (5))

A=[]
if A == []:    
    for i in range(5):    
        A.append([x_random[i],y_random[i]])
        
B=[]
if B == []:    
    for i in range(5):    
        B.append([p_random[i],q_random[i]])
            
C=[]
if C == []:    
    for i in range(5):    
        C.append([r_random[i],s_random[i]])


agent_locations = np.vstack((A,B,C))
for i in range(4,len(agent_locations)-5,5):
    print(agent_locations[i])
    
    
targ=['C','D','E','F']
A=dict((c,i)for i,c in enumerate(targ))
A[C]



goal=np.zeros((5,5))
for i  in range(-5,0,1):
    for j in range(-5,0,1):
        if((agent_locations[i][1]-agent_locations[j][1])**2+(agent_locations[i][0]-agent_locations[j][0])**2) <= 2:
            goal[i][j]=1
char_index=dict((n,i) for n,i in np.ndenumerate(goal))



