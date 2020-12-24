#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is for generating dynamic simulation data.

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

from queue import *

# pdb.set_trace()
# --------------------------------------------------------------      
# Constant block
# --------------------------------------------------------------      
# Read suject's network
# - param
FNAME = 'dS001.csv'
RANDOM_NUM_GOALS = False # If true, the number of goals will vary across mazes
VERSION_NAME = 'dS001'
TARGET_ORDER = np.array(['A','B','C','D'])
STEP_TOTAL = 20
AGENT_NAME = "dS001_S_1"
# - dir
DIR_ROOT = os.getcwd()
DIR_TXT_OUTPUT = os.path.join(DIR_ROOT, '..','data','data_dynamic',\
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
df_collect_final_move = pd.DataFrame()

with open(FNAME, encoding='utf-8') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    agents_list=list(readCSV)


GAME_ART = [
['##########################',
 '#                        #',
 '#                        #',
 '#                        #',
 '#                        #',
 '#                        #',  # Environment.
 '#     ##############     #',
 '#     #            #     #',
 '#     #            #     #',
 '#     #            #     #',
 '#     #            #     #',
 '#     #            #     #',
 '#     #            #     #',
 '#     #            #     #',
 '#     #            #     #',
 '#     #            #     #',
 '#     #            #     #',
 '#     #            #     #',
 '#     #            #     #',
 '#     ##############     #',
 '#                        #',
 '#                        #',
 '#                        #',
 '#                        #',
 '#                        #',
 '##########################'],
    ]

n_chosen_agents = 5
n_move_agents = n_chosen_agents-1
Chosen_agents_index = list(range(0, n_chosen_agents))
Chosen_agents_label = ['S','A','B','C','D']
center_agent= 0  #index of the pinned-at-center agent
 
x_random=random.sample(range(7,19), (n_move_agents))
y_random=random.sample(range(7,19), (n_move_agents))

#GAME_ART[0][12]=GAME_ART[0][12][0:12]+'S'+GAME_ART[0][12][12+1:]

for i in range(n_move_agents):
    while (x_random[i]==12) and (y_random[i]==12):
            x_random[i]=random.choice(range(7,19))
            y_random[i]=random.choice(range(7,19))
            
x_random.insert(center_agent, 12)            
y_random.insert(center_agent, 12)            

for i in range(n_chosen_agents):
    GAME_ART[0][y_random[i]]=GAME_ART[0][y_random[i]][0:x_random[i]]+Chosen_agents_label[Chosen_agents_index[i]]+GAME_ART[0][y_random[i]][x_random[i]+1:]

#social reward    
social_reward=np.zeros((n_chosen_agents,n_chosen_agents))
for i in range (n_chosen_agents):
    for j in range(n_chosen_agents):
        social_reward[i][j]=agents_list[i+1][j+1]


#agent start place, distance and energy
def inputs(agent_place):
    #place
    if np.all(agent_place) == 0:    
        for i in range(n_chosen_agents):    
            agent_place[i,] = np.array([x_random[i],y_random[i]])
    else:
        agent_place=np.copy(agent_place_update)
    #dist
    distance=np.zeros((n_chosen_agents,n_chosen_agents))
    for i in range (n_chosen_agents):
        for j in range (n_chosen_agents):        
            distance[i][j]=np.sqrt(np.square(agent_place[i][0]-agent_place[j][0])+np.square(agent_place[i][1]-agent_place[j][1]))
    #energy
    #energy=social_reward-distance
    #force
    force=social_reward/np.square(distance)

    return agent_place,distance,force

def move():
    #vector of moving force
    force_vector = np.zeros((n_chosen_agents,2*n_chosen_agents))
    for i in range(n_chosen_agents):    
        for j in range(n_chosen_agents):       
            if i!=j:
                x=agent_place[j][0]-agent_place[i][0]
                y=agent_place[j][1]-agent_place[i][1]
                force_vector[i][2*j]=(x/np.sqrt(np.square(x)+np.square(y)))*force[i][j]
                force_vector[i][2*j+1]=(y/np.sqrt(np.square(x)+np.square(y)))*force[i][j]    
            else:
                force_vector[i][2*j]=0
                force_vector[i][2*j+1]=0

    #generate move factor
    force_vector_sum=np.zeros((n_chosen_agents,2))
    move_vector=np.zeros((n_chosen_agents,2))
    for i in range(n_chosen_agents):
        force_vector_sum[i][0]=sum(force_vector[i][range(0,2*n_chosen_agents,2)]) #force vector x coord.
        force_vector_sum[i][1]=sum(force_vector[i][range(1,2*n_chosen_agents,2)]) #force vector y coord.
        if np.sqrt(np.square(force_vector_sum[i][0])+np.square(force_vector_sum[i][1]))<1:
            move_vector[i][0]=force_vector_sum[i][0]
            move_vector[i][1]=force_vector_sum[i][1]
        else:
            move_vector[i][0]=force_vector_sum[i][0]/np.sqrt(np.square(force_vector_sum[i][0])+np.square(force_vector_sum[i][1]))  #unit force vector x
            move_vector[i][1]=force_vector_sum[i][1]/np.sqrt(np.square(force_vector_sum[i][0])+np.square(force_vector_sum[i][1]))  #unit force vector y
    
    move=np.zeros((n_chosen_agents,2))
    agent_place_update=np.array(agent_place)
    #move
    for i in range(n_chosen_agents):
        if move_vector[i][0]>=0.5:
            move[i][0]=1
        if move_vector[i][0]<=-0.5:
            move[i][0]=-1
        if 0.5>move_vector[i][0]>-0.5:
            move[i][0]=0
        if move_vector[i][1]>=0.5:
            move[i][1]=1
        if move_vector[i][1]<=-0.5:
            move[i][1]=-1
        if 0.5>move_vector[i][1]>-0.5:
            move[i][1]=0            
    #change agent coord.          
    for i in range(n_chosen_agents):                    
        if 0<agent_place[i][0]+move[i][0]<24:
            if agent_place[i][0]!=12 or agent_place[i][1]!=12:
                agent_place_update[i][0]=agent_place[i][0]+move[i][0]
        else:
            agent_place_update[i][0]=agent_place[i][0]
        if 0<agent_place[i][1]+move[i][1]<24:
            if agent_place[i][0]!=12 or agent_place[i][1]!=12:
                agent_place_update[i][1]=agent_place[i][1]+move[i][1]        
        else:
            agent_place_update[i][1]=agent_place[i][1]                    
                            
#avoid crashing together                        
    #force_vector_abs=np.sqrt(np.square(force_vector_sum[:,0])+np.square(force_vector_sum[:,1]))                      
                        
                            
    for i in range(n_chosen_agents):                        
        for j in range(n_chosen_agents):                        
            if i!=j:                    
                if agent_place_update[i,0]==agent_place_update[j,0] and agent_place_update[i,1]==agent_place_update[j,1]:                
                    agent_place_update[i]=agent_place[i]                
                    agent_place_update[j]=agent_place[j]                
                                    
                    #if force_vector_abs[i]>=force_vector_abs[j]:                
                    #    agent_place_update[j]=agent_place[j]                
                    #if force_vector_abs[i]<force_vector_abs[j]:
                    #    agent_place_update[i]=agent_place[i]                    
                #else:    
                    #agent_place_update[i]=agent_place_update[i]   
                    #agent_place_update[j]=agent_place_update[j]   
            #else:
               # agent_place_update[i]=agent_place_update[i]   
                #agent_place_update[j]=agent_place_update[j]   
                
    
    return agent_place_update


#def gameart(agent_place_update):
    #update gameart
#    GAME_ART_UPDATE = [
#    ['##########################',
#     '#                        #',
#     '#                        #',
#     '#                        #',
#     '#                        #',
#     '#                        #',  # Environment.
#     '#     ##############     #',
#     '#     #            #     #',
#     '#     #            #     #',
#     '#     #            #     #',
#     '#     #            #     #',
#     '#     #            #     #',
#     '#     #            #     #',
#     '#     #            #     #',
#     '#     #            #     #',
#     '#     #            #     #',
#     '#     #            #     #',
#     '#     #            #     #',
#     '#     #            #     #',
#     '#     ##############     #',
#     '#                        #',
#     '#                        #',
#     '#                        #',
#     '#                        #',
#     '#                        #',
#     '##########################'],
#        ]


#    agent_place_update=agent_place_update.astype(int)
#    for i in range(n_chosen_agents):
#        GAME_ART_UPDATE[0][agent_place_update[i][1]]=GAME_ART_UPDATE[0][agent_place_update[i][1]][0:agent_place_update[i][0]]+Chosen_agents_label[Chosen_agents_index[i]]+GAME_ART_UPDATE[0][agent_place_update[i][1]][agent_place_update[i][0]+1:]
#    return GAME_ART_UPDATE




step=0
agent_place = np.zeros((5,2))                      #recent agent place
#agent_history = np.zeros((n_chosen_agents,2,10))  #for checking settle
all_agent_place = np.zeros((1,2))                  #all coordinates of agent place
flag=0

while step < STEP_TOTAL:   
    agent_place,distance,force=inputs(agent_place)
    agent_place_update=move()
    #GAME_ART_UPDATE=gameart(agent_place_update)
    #stop point
    for i in range(n_chosen_agents):
        for j in range(2):
            if agent_place[i][j] != agent_place_update[i][j]:
                flag=0
            else:
                flag+=1
    
    #agent_history=np.delete(agent_history,0,2)
    #agent_place_update_h=np.reshape(agent_place_update,(5,2,1))
    #agent_history=np.append(agent_history,agent_place_update_h,2)
    

    #for i in range(n_chosen_agents):
        #if agent_place[i][0]==agent_place_update[i][0]:
        #if agent_place_update[i][0]-0.5<=np.mean(agent_history[i][0])<=agent_place_update[i][0]+0.5:      #if it's wobbling
            #flag=flag+1
        #if agent_place[i][1]==agent_place_update[i][1]: 
        #if agent_place_update[i][1]-0.5<=np.mean(agent_history[i][1])<=agent_place_update[i][1]+0.5:      #if it's wobbling
                 
            #flag=flag+1
        #else:
            #flag=0
    if flag==10:
        step=STEP_TOTAL
                    
    else:
        step=step+1
        all_agent_place=np.vstack((all_agent_place,agent_place))              
        # Save the parameters of this maze
        #df_final_move = pd.DataFrame(data = {'agent_S_place': agent_order_name[agent_CDEF_index],\
                                             #'agent_A_place': social_reward_ordered,\
                                             #'agent_B_place': social_reward_ordered,\
                                             #'agent_C_place': social_reward_ordered,\
                                             #'agent_D_place': social_reward_ordered,\})
            
            #pdb.set_trace()              
    #print(flag)    
all_agent_place=np.vstack((all_agent_place,agent_place))              
all_agent_place=np.vstack((all_agent_place,agent_place_update))  
all_agent_place=np.delete(all_agent_place,0,0)
    
#to fit the model, coordinates are 0~12
all_agent_place = all_agent_place -6    
    
try:    
    output_file  = os.path.join(DIR_TXT_OUTPUT, \
                                        str(AGENT_NAME)+"_"+str(step)+".txt")
    text_file = open(output_file, "w")
    text_file.write('Maze:\n')
                
    for i in range(26):
        text_file.write(GAME_ART[0][i])
        text_file.write('\n')
    agent_place_str=str(all_agent_place)
    text_file.write(agent_place_str)
    text_file.write('\n')
    text_file.close()
except:    
    print('error')            
