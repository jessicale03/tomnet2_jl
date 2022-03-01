#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is for generating dynamic simulation query state.

@author: Hsinyi; Chuang, Yun-Shiuang; Yen-Ling
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Dependency imports

#from ai_safety_gridworlds.environments.shared import safety_game

import numpy as np
import csv
import random 
import pdb
import os
import pandas as pd
import itertools  

from queue import *

# pdb.set_trace()
# --------------------------------------------------------------      
# Constant block
# --------------------------------------------------------------      
# Read suject's network
# - param
#FNAME = 'dS001.csv'
VERSION_NAME = 'd_query'
#STEP_TOTAL = 20
AGENT_NAME = "query_test_"
# - dir
DIR_ROOT = os.getcwd()
DIR_TXT_OUTPUT = os.path.join(DIR_ROOT, '..','data','data_preference_predictions',\
                              VERSION_NAME)
if not os.path.exists(DIR_TXT_OUTPUT):
  os.makedirs(DIR_TXT_OUTPUT)

# - file
#FILE_CSV_SUMMARY = os.path.join(DIR_TXT_OUTPUT, 'summary_dynamic.csv')
#FNAME='/bml/Data/Bank5/AI/AI_simulation/S002_familyonly.csv'
#FNAME='/bml/Data/Bank5/AI/ai-safety-gridworlds-master/Robohon/Subj_social_network_paramerter_simulation/S001.csv'
  
# ---------------------------------------------
# A data frame to collect parameters maze by maze
# ---------------------------------------------

GAME_ART = [
['##########################',
 '#                        #',
 '#                        #',
 '#                        #',
 '#                        #',
 '#                        #',  # Environment.
 '#     ##############     #',
 '#     #            #     #',    #[0][7][12]
 '#     #            #     #',
 '#     #            #     #',
 '#     #            #     #',
 '#     #            #     #',
 '#     #            #     #',    #[0][12][7],[0][12][12],[0][12][17]
 '#     #            #     #',
 '#     #            #     #',
 '#     #            #     #',
 '#     #            #     #',
 '#     #            #     #',   #[0][17][12]
 '#     #            #     #', 
 '#     ##############     #',
 '#                        #',
 '#                        #',
 '#                        #',
 '#                        #',
 '#                        #',
 '##########################'],
    ]

# n_chosen_agents = 5
# n_move_agents = n_chosen_agents-1
# Chosen_agents_index = list(range(0, n_chosen_agents))
# Chosen_agents_label = ['S','A','B','C','D']
# center_agent= 0  #index of the pinned-at-center agent
#pdb.set_trace() 
agent_combination_list = list(itertools.permutations('SABCD',5))  #5 agents, generate the combinations

for i in range (len(agent_combination_list)):
    GAME_ART[0][7]=GAME_ART[0][7][0:12]+agent_combination_list[i][0]+GAME_ART[0][7][12+1:]
    GAME_ART[0][12]=GAME_ART[0][12][0:7]+agent_combination_list[i][1]+GAME_ART[0][12][7+1:12]+agent_combination_list[i][2]+GAME_ART[0][12][12+1:17]+agent_combination_list[i][3]+GAME_ART[0][12][17+1:]
    GAME_ART[0][17]=GAME_ART[0][17][0:12]+agent_combination_list[i][4]+GAME_ART[0][7][12+1:]
    # pdb.set_trace()
    
    try:
        output_file  = os.path.join(DIR_TXT_OUTPUT, \
                                         str(AGENT_NAME)+str(i+1)+".txt")
        text_file = open(output_file, "w")
        text_file.write('Maze:\n')
                
        for i in range(26):
            text_file.write(GAME_ART[0][i])
            text_file.write('\n')
        text_file.close()
    except:
        print('error')


                                  
# for i in range (n_chosen_agents):
#     for j in range ()
#     GAME_ART[0][12]=GAME_ART[0][12][0:12]+agent_combination_list[i,j]+GAME_ART[0][12][12+1:] #center agent
#     for j in range (n_chosen_agents):
#         if i==j:
#             pass
#         if i!=j:
            
# #GAME_ART[0][12]=GAME_ART[0][12][0:12]+'S'+GAME_ART[0][12][12+1:]

# for i in range(n_move_agents):
#     while (x_random[i]==12) and (y_random[i]==12):
#             x_random[i]=random.choice(range(7,19))
#             y_random[i]=random.choice(range(7,19))
            
# x_random.insert(center_agent, 12)            
# y_random.insert(center_agent, 12)            

# for i in range(n_chosen_agents):
#     GAME_ART[0][y_random[i]]=GAME_ART[0][y_random[i]][0:x_random[i]]+Chosen_agents_label[Chosen_agents_index[i]]+GAME_ART[0][y_random[i]][x_random[i]+1:]


# #agent start place, distance and energy
# def inputs(agent_place):
#     #place
#     if np.all(agent_place) == 0:    
#         for i in range(n_chosen_agents):    
#             agent_place[i,] = np.array([x_random[i],y_random[i]])
#     else:
#         agent_place=np.copy(agent_place_update)
#     #dist
#     distance=np.zeros((n_chosen_agents,n_chosen_agents))
#     for i in range (n_chosen_agents):
#         for j in range (n_chosen_agents):        
#             distance[i][j]=np.sqrt(np.square(agent_place[i][0]-agent_place[j][0])+np.square(agent_place[i][1]-agent_place[j][1]))
#     #energy
#     #energy=social_reward-distance
#     #force
#     force=social_reward/np.square(distance)

#     return agent_place,distance,force


# try:    
#     output_file  = os.path.join(DIR_TXT_OUTPUT, \
#                                         str(AGENT_NAME)+".txt")
#     text_file = open(output_file, "w")
#     text_file.write('Maze:\n')
                
#     for i in range(26):
#         text_file.write(GAME_ART[0][i])
#         text_file.write('\n')
#     # agent_place_str=str(all_agent_place)
#     # text_file.write(agent_place_str)
#     # text_file.write('\n')
#     text_file.close()
# except:    
#     print('error')            
