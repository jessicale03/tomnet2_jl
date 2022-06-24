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
RANDOM_NUM_GOALS = False  # If true, the number of goals will vary across mazes
VERSION_NAME = 'dS001_pred'
STEP_TOTAL = 20           # max steps in each maze
AGENT_NAME = "dS001"
SET_UP_MAZE_TOTAL = 1000  # SET_UP_MAZE_TOTAL=total maze amount
PRINT_FINAL_MAZE = False

# - dir
DIR_ROOT = os.getcwd()
DIR_TXT_OUTPUT = os.path.join(DIR_ROOT, '..', '..', '..','data','data_dynamic',\
                              VERSION_NAME)
if not os.path.exists(DIR_TXT_OUTPUT):
  os.makedirs(DIR_TXT_OUTPUT)

# - file
FILE_CSV_SUMMARY = os.path.join(DIR_TXT_OUTPUT, 'summary.csv')
#FNAME='/bml/Data/Bank5/AI/AI_simulation/S002_familyonly.csv'
#FNAME='/bml/Data/Bank5/AI/ai-safety-gridworlds-master/Robohon/Subj_social_network_paramerter_simulation/S001.csv'
  
# ---------------------------------------------
# A data frame to collect parameters maze by maze
# ---------------------------------------------
#df_collect_final_move = pd.DataFrame()

with open(FNAME, encoding='utf-8') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    agents_list=list(readCSV)

n_chosen_agents = 5
n_move_agents = n_chosen_agents-1
Chosen_agents_index = list(range(0, n_chosen_agents))
Chosen_agents_label = ['S','A','B','C','D']
#center_agent= 0  #index of the pinned-at-center agent


#social reward    
social_reward=np.zeros((n_chosen_agents,n_chosen_agents))
for i in range (n_chosen_agents):
    for j in range(n_chosen_agents):
        social_reward[i][j]=agents_list[i+1][j+1]
        
#social reward check
#if any row of the social score abs() sum = 0 
#the agent won't move in the maze, then fix them in the center of the maze
fix_central_index=[]
for i in range (n_chosen_agents):
    check=sum(abs(social_reward[i]))
    if check == 0:
        fix_central_index.append(i)


def Center_Agent():
    if not fix_central_index:
        center_agent = maze_total%5
    if fix_central_index:        
        center_agent = fix_central_index[maze_total%len(fix_central_index)]
    return center_agent

def game_art():
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

    # n_chosen_agents = 5
    # n_move_agents = n_chosen_agents-1
    # Chosen_agents_index = list(range(0, n_chosen_agents))
    # Chosen_agents_label = ['S','A','B','C','D']
    # center_agent= 0  #index of the pinned-at-center agent
     
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

    return x_random,y_random,GAME_ART
        

#agent start place, distance and energy
def inputs(agent_place,x_random,y_random):
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

def final_maze():            #output the final state of the maze (coords need to be added)
    GAME_FINAL = [
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

    final_agent_place = np.zeros((5,2))
    for i in range(-5,0):
        final_agent_place[i] = all_agent_place[i,]
    final_agent_place = final_agent_place.astype(int)

    for i in range(n_chosen_agents):
        GAME_FINAL[0][final_agent_place[i,1]]=GAME_FINAL[0][final_agent_place[i,1]][0:final_agent_place[i,0]]+Chosen_agents_label[Chosen_agents_index[i]]+GAME_FINAL[0][final_agent_place[i,1]][final_agent_place[i,0]+1:]
    
    
    return GAME_FINAL



maze_total = 0
df_collect_summary = pd.DataFrame()

while maze_total < SET_UP_MAZE_TOTAL:
    #new maze
    step=0
    true_step=999
    agent_place = np.zeros((5,2))                      #recent agent place
    all_agent_place = np.zeros((1,2))                  #all coordinates of agent place
    flag_1=0
    flag_2=0
    
    center_agent=Center_Agent()    
    x_random,y_random,GAME_ART=game_art()
    
    while step < STEP_TOTAL:   
        agent_place,distance,force=inputs(agent_place,x_random,y_random)
        agent_place_update=move()
        #GAME_ART_UPDATE=gameart(agent_place_update)
        
        #stop point_1:settle
        for i in range(n_chosen_agents):
            for j in range(2):
                if agent_place[i][j] != agent_place_update[i][j]:
                    flag_1=0
                else:
                    flag_1+=1
                    
        #stop point_2:wobbles
        if step>0:
            for i in range(n_chosen_agents):
                for j in range(2):
                    if all_agent_place[-5+i][j] == agent_place_update[i][j]:
                        flag_2+=1
                    else:
                        flag_2=0
        else:pass

        if flag_1>=10 or flag_2>=10:   #all of the coords is the same to the previous(or pre-previous) step
            true_step = step
            step = STEP_TOTAL
                        
        else:
            step+=1
            true_step = step
            all_agent_place=np.vstack((all_agent_place,agent_place))              
            # Save the parameters of this maze
            
            
            #pdb.set_trace()

    all_agent_place=np.vstack((all_agent_place,agent_place))              
    all_agent_place=np.vstack((all_agent_place,agent_place_update))  
    all_agent_place=np.delete(all_agent_place,0,0)
    


#print the last maze here
    GAME_FINAL = final_maze()
    
    #to fit the model, coordinates are 0~12
    all_agent_place = all_agent_place -6


    maze_final_stat=np.zeros((5,5))
    for i  in range(-5,0):
        for j in range(-5,0):
            if(int(all_agent_place[i][1])-int(all_agent_place[j][1]))**2+(int(all_agent_place[i][0])-int(all_agent_place[j][0]))**2 <= 2:
                maze_final_stat[i][j]=1
            

    #take 10 units of the 5*5 to represent status
    maze_final_stat_bin=np.zeros(10)  #the binary code of maze_final_stat (matrix to binary code)
    l=0
    for i in range (5):
        for j in range (i):
            maze_final_stat_bin[l]=maze_final_stat[j][i]
            l+=1

    #change into string
    maze_final_stat_bin=maze_final_stat_bin.astype(int)
    maze_final_stat_bin=maze_final_stat_bin.astype(str)
    maze_final_stat_bin="".join(maze_final_stat_bin)

    #bin to dec(0~1023)
    maze_final_stat_dec=int(maze_final_stat_bin,base=2)
    
    maze_final_stat_bin = maze_final_stat_bin.split()
    df_summary = pd.DataFrame(data = {'maze': maze_total+1,\
                                      'steps': true_step,\
                                      'maze_final_binary': maze_final_stat_bin,\
                                      'maze_final_decimal': maze_final_stat_dec,\
                                      'center_agent': Chosen_agents_label[center_agent],\
                                      'flag_1': flag_1,\
                                      'flag_2': flag_2})
    df_collect_summary = df_collect_summary.append(df_summary)
        
    try:    
        #print start maze
        output_file  = os.path.join(DIR_TXT_OUTPUT, \
                                            str(AGENT_NAME)+"_"+str(maze_total+1)+".txt")
        text_file = open(output_file, "w")
        text_file.write('Maze:\n')
                    
        for i in range(26):
            text_file.write(GAME_ART[0][i])
            text_file.write('\n')
        agent_place_str=str(all_agent_place)
        text_file.write(agent_place_str)
        text_file.write('\n')
        text_file.close()
        #print end maze
        if PRINT_FINAL_MAZE == True:
            output_file  = os.path.join(DIR_TXT_OUTPUT, \
                                            str(AGENT_NAME)+"_"+str(maze_total+1)+"FINAL.txt")
            text_file = open(output_file, "w")
            text_file.write('Final_Maze:\n')
                    
            for i in range(26):
                text_file.write(GAME_FINAL[0][i])
                text_file.write('\n')
            text_file.write('\n')
            text_file.close()
        else:pass
    except:    
        print('error')            
    maze_total+=1
    if maze_total == SET_UP_MAZE_TOTAL:
        df_collect_summary.to_csv(FILE_CSV_SUMMARY)
