#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:48:03 2020

@author: elaine
"""
import numpy as np

file = "/Users/elaine/Desktop/TOMNET/tomnet-project/tomnet2/data/data_dynamic/dS001/dS001_A_1_20.txt"
with open(file) as fp:
    lines = list(fp)
    maze = lines[8:20]
    for i in range(12):
        maze[i]=maze[i][7:18]
    np_maze=np.array(maze)
    agent_locations=lines[27:]
    #agent_locations=np.array(agent_locations)
    #print(maze)
fp.close()

location=[]
for i in agent_locations :
    i = i[2:8]
    tmp = i.split(".")
    try:
        location.append([tmp[0],tmp[1]])
    except:pass
   
#agent_location = 
np_actions_S = np.zeros((12,12,5), dtype=np.int8)
np_obst = np.ones((12,12,1), dtype=np.int8)
np_tensor = np.dstack((np_actions_S,np_obst))
