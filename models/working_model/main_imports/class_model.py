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

class Model(mp.ModelParameter):
  # --------------------------------------
  # Constant block
  # --------------------------------------
  # --------------------------------------
  # Constant: Model parameters
  # --------------------------------------
  # Use inheretance to share the model constants across classes

  # --------------------------------------
  # Constant: Training parameters
  # --------------------------------------
  #Batch size = 16, same in the paper A.3.1. EXPERIMENT 1: SINGLE PAST MDP)
  BATCH_SIZE = 16
  # BATCH_SIZE = 10 # for human data with less than 160 files
  BATCH_SIZE_TRAIN = BATCH_SIZE # size of the batch for traning (number of the steps within each batch)
  BATCH_SIZE_VAL = BATCH_SIZE # size of the batch for validation
  BATCH_SIZE_TEST = BATCH_SIZE # size of batch for testing

  # for testing on a GPU machine with 10000 files
  SUBSET_SIZE = -1 # use all files
  # tota number of minibatches used for training
  # (Paper: 2M minibatches, A.3.1. EXPERIMENT 1: SINGLE PAST MDP)
  TRAIN_STEPS = 10000
  REPORT_FREQ = 100 # the frequency of writing the error to error.csv
  #path_txt_data = os.getcwd() + '/S002a/'
  # TRUE: use the full data set for validation
  # (but this would not be fair because a portion of the data has already been seen)
  # FALSE: data split using train:vali:test = 8:1:1
  FULL_VALIDATION = False
  USE_CKPT = False
  # the version of the training
  TRAINING_VERSION = 'v1'

  # --------------------------------------
  # Variable: Training parameters
  # --------------------------------------
  path_mode =  os.getcwd()  # Necessary when the output dir and script dir is different
  # for simulation data
  # ckpt_fname = 'test_on_simulation_data/training_result/caches/cache_dS001_v99_commit_?'
  # train_fname = 'test_on_simulation_data/training_result/caches/cache_dS001_v99_commit_?'
  # path_txt_data ='../../data/data_simulation/dS001/'

  # for **human/simulation data
  #use panda df to store these values
  path_ckpt = os.path.join('test_on_simulation_data','training_result','caches')
  path_train = os.path.join('test_on_simulation_data','training_result','caches')
  path_txt_data = os.path.join('..','..','data','data_dynamic')

  path_ckpt = os.path.join(path_mode,path_ckpt)
  path_train = os.path.join(path_mode,path_train)
  path_txt_data = os.path.join(path_mode,path_txt_data)
