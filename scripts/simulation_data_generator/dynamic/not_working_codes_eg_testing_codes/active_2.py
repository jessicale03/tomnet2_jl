#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:43:59 2020

@author: elaine
"""

import random 
import numpy as np
import csv
import pandas as pd

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
Chosen_agents_index = list(range(0, n_chosen_agents))
Chosen_agents_label = ['S','A','B','C','D'] 
x_random=random.sample(range(7,18), (n_chosen_agents))
y_random=random.sample(range(7,18), (n_chosen_agents))

for i in range(n_chosen_agents):
    GAME_ART[0][y_random[i]]=GAME_ART[0][y_random[i]][0:x_random[i]]+Chosen_agents_label[Chosen_agents_index[i]]+GAME_ART[0][y_random[i]][x_random[i]+1:]


FNAME = 'dS001.csv'
with open(FNAME, encoding='utf-8') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    agents_list=list(readCSV)

#social reward    
social_reward=np.zeros((n_chosen_agents,n_chosen_agents))
for i in range (n_chosen_agents):
    for j in range(n_chosen_agents):
        social_reward[i][j]=agents_list[i+1][j+1]


agent_place = []
#agent start place, distance and energy
def inputs():
    #place
    if agent_place == []:    
        for i in range(n_chosen_agents):    
            agent_place.append([x_random[i],y_random[i]])
    else:
        agent_place=agent_place_update
    #dist
    distance=np.zeros((n_chosen_agents,n_chosen_agents))
    for i in range (n_chosen_agents):
        for j in range (n_chosen_agents):        
            distance[i][j]=abs(agent_place[i][0]-agent_place[j][0])+abs(agent_place[i][1]-agent_place[j][1])
    #energy
    energy=social_reward-distance

    return agent_place,distance,energy

def move():
    #vector of moving force
    moving_force = np.zeros((n_chosen_agents,2*n_chosen_agents))
    for i in range(n_chosen_agents):    
        for j in range(n_chosen_agents):       
            if i!=j:
                x=agent_place[i][0]-agent_place[j][0]
                y=agent_place[i][1]-agent_place[j][1]
                moving_force[i][2*j]=(x/np.sqrt(np.square(abs(x))+np.square(abs(y))))*energy[i][j]
                moving_force[i][2*j+1]=(y/np.sqrt(np.square(abs(x))+np.square(abs(y))))*energy[i][j]    
            else:
                moving_force[i][2*j]=0
                moving_force[i][2*j+1]=0

    #move
    move_force_sum=np.zeros((n_chosen_agents,2))
    for i in range(n_chosen_agents):
        move_force_sum[i][0]=sum(moving_force[i][range(0,2*n_chosen_agents,2)])
        move_force_sum[i][1]=sum(moving_force[i][range(1,2*n_chosen_agents,2)])
    move=np.zeros((n_chosen_agents,2))
    for i in range(n_chosen_agents):
        if abs(move_force_sum[i][0])>=abs(move_force_sum[i][1]):
            if move_force_sum[i][0]>0:
                move[i][0]=1
            else:
                move[i][0]=-1        
        else:
            if move_force_sum[i][1]>0:
                move[i][1]=1
            else:
                move[i][1]=-1            
    agent_place_update=agent_place+move
    return move,agent_place_update

def gameart():
    #update gameart
    GAME_ART_UPDATE = [
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

    agent_place_update=agent_place_update.astype(int)
    for i in range(n_chosen_agents):
        GAME_ART_UPDATE[0][agent_place_update[i][1]]=GAME_ART_UPDATE[0][agent_place_update[i][1]][0:agent_place_update[i][0]]+Chosen_agents_label[Chosen_agents_index[i]]+GAME_ART_UPDATE[0][agent_place_update[i][1]][agent_place_update[i][0]+1:]
    return GAME_ART_UPDATE

