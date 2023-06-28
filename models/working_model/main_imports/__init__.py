import os
import sys
import time
import datetime
import pandas as pd
import tensorflow as tf
import sys
import argparse
import numpy as np
import pdb # For debugging
import commented_charnet as cn
import commented_prednet as pn
import commented_data_handler as dh
import commented_model_parameters as mp
import commented_batch_generator as bg

# all extracted imports
from __init__.py import *
from train_test_validate.py import *
from optimization.py import *
from _create_graphs.py import _create_graphs
from class_model import Model
  
  
  def __init__(self, args):
    '''
    The constructor for the Model class.
    '''
    # --------------------------------------------------------------
    # Set up constants
    # --------------------------------------------------------------
    path_ckpt = \
    os.path.join(self.path_ckpt,\
                 'cache_dS0054_v1_commit_9cb635')#,\
                 #args.subj_name)

    path_train = \
    os.path.join(self.path_ckpt,\
                 'cache_dS0054_v1_commit_9cb635') #,\
                 #args.subj_name)

    # create the path if not yet existed
    if not os.path.exists(path_ckpt):
      os.makedirs(path_ckpt)
    if not os.path.exists(path_train):
      os.makedirs(path_train)

    path_ckpt = \
    os.path.join(path_ckpt,\
                 'logs','model.ckpt')

    path_train = \
    os.path.join(path_train,\
                'train')

    self.path_ckpt = path_ckpt
    self.path_train = path_train

    # Set up batch generator
    self.batch_generator = bg.BatchGenerator()

    # --------------------------------------------------------------
    # Set up all the placeholders
    # --------------------------------------------------------------
    self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

    # For trajectory data and the corresponding labels
    self.train_data_traj_placeholder = tf.placeholder(dtype = tf.float32,\
                                                      shape = [self.BATCH_SIZE_TRAIN, self.MAX_TRAJECTORY_SIZE, self.MAZE_HEIGHT, self.MAZE_WIDTH, self.MAZE_DEPTH_TRAJECTORY],\
                                                      name = 'train_data_traj_placeholder')
    self.train_labels_traj_placeholder = tf.placeholder(dtype = tf.int32,\
                                                        shape = [self.BATCH_SIZE_TRAIN],\
                                                        name = 'train_labels_traj_placeholder')
    self.vali_data_traj_placeholder = tf.placeholder(dtype = tf.float32,\
                                                     shape = [self.BATCH_SIZE_VAL, self.MAX_TRAJECTORY_SIZE, self.MAZE_HEIGHT, self.MAZE_WIDTH, self.MAZE_DEPTH_TRAJECTORY],\
                                                     name = 'vali_data_traj_placeholder')
    self.vali_labels_traj_placeholder = tf.placeholder(dtype = tf.int32,\
                                                       shape = [self.BATCH_SIZE_VAL],\
                                                       name = 'vali_labels_traj_placeholder')

    # For query state data and the cooresponding labels
    self.train_data_query_state_placeholder = tf.placeholder(dtype = tf.float32,\
                                                             shape = [self.BATCH_SIZE_TRAIN, self.MAZE_HEIGHT, self.MAZE_WIDTH, self.MAZE_DEPTH_QUERY_STATE],\
                                                             name = 'train_data_query_state_placeholder')
    self.train_labels_query_state_placeholder = tf.placeholder(dtype = tf.int32,\
                                                               shape = [self.BATCH_SIZE_TRAIN],\
                                                               name = 'train_labels_query_state_placeholder')
    self.vali_data_query_state_placeholder = tf.placeholder(dtype = tf.float32,\
                                                            shape = [self.BATCH_SIZE_VAL, self.MAZE_HEIGHT, self.MAZE_WIDTH, self.MAZE_DEPTH_QUERY_STATE],\
                                                            name = 'vali_data_query_state_placeholder')
    self.vali_labels_query_state_placeholder = tf.placeholder(dtype = tf.int32,\
                                                              shape = [self.BATCH_SIZE_VAL],\
                                                              name = 'vali_labels_query_state_placeholder')

    # --------------------------------------------------------------
    # Parse the trajectory data and labels
    # train_data_traj = (num_train_files, trajectory_size, height, width, MAZE_DEPTH)
    # train_labels_traj = (num_train_files, )
    # --------------------------------------------------------------
    # Load data
    dir = os.path.join(self.path_txt_data,args.subj_name)
    # pdb.set_trace()
    data_handler = dh.DataHandler()
    # Note that all training examples are NOT shuffled randomly (by defualt)
    # during data_handler.parse_trajectories()
    # pdb.set_trace()
    self.train_data_traj, self.vali_data_traj,\
    self.test_data_traj, self.train_labels_traj,\
    self.vali_labels_traj, self.test_labels_traj,\
    self.all_files_traj, self.train_files_traj, self.vali_files_traj, self.test_files_traj\
    = data_handler.parse_whole_data_set(dir,\
                                        mode=args.mode,\
                                        shuf=args.shuffle,\
                                        subset_size = self.SUBSET_SIZE,\
                                        parse_query_state = False)
    # --------------------------------------------------------------
    # Parse the query state data and labels
    # train_data_traj = (num_train_files, height, width, MAZE_DEPTH_QUERY_STATE)
    # train_labels_traj = (num_train_files, 1)
    # --------------------------------------------------------------
    # pdb.set_trace()
    self.train_data_query_state, self.vali_data_query_state,\
    self.test_data_query_state, self.train_labels_query_state,\
    self.vali_labels_query_state, self.test_labels_query_state,\
    self.all_files_query_state, self.train_files_query_state, self.vali_files_query_state, self.test_files_query_state \
    = data_handler.parse_whole_data_set(dir,\
                                        mode=args.mode,\
                                        shuf=args.shuffle,\
                                        subset_size = self.SUBSET_SIZE,\
                                        parse_query_state = True)

    #print('End of __init__-----------------')
    #pdb.set_trace()
