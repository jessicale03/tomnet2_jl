
# -*- coding: utf-8 -*-
"""
class Model(mp.ModelParameter):

The class for training the ToMNET model.

Note:
  Inherit mp.ModelParameter to share model constants.

@author: Chuang, Yun-Shiuan; Edwinn
"""
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
import charnet as cn
import prednet as pn
import data_handler as dh
import model_parameters as mp
import batch_generator as bg

# all extracted imports
from __init__.py import *
from train_test_validate.py import *
from optimization.py import *
from _create_graphs.py import _create_graphs
from class_model import Model


if __name__ == "__main__":
  # --------------------------------------------------------
  # Constants
  # --------------------------------------------------------
  # LIST_SUBJECTS = ["S0" + str(i) for i in ["35","50","51","52"]]
  LIST_SUBJECTS = ["dS0054"]

  # LIST_SUBJECTS = ["S0" + str(i) for i in ["24","33","35","50","51","52"]]

  # --------------------------------------------------------
  # Iterate through the subject list
  # --------------------------------------------------------
  for subj_index, subj_name in enumerate(LIST_SUBJECTS):
    print("\n================================= \n"+
          "Start working on "+ subj_name+'\n'+
          "================================= \n")

    # reseting the graph is necessary for running the script via spyder or other
    # ipython intepreter
    tf.reset_default_graph()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all', help='all: train and test, train: only train, test: only test')
    parser.add_argument('--shuffle', type=str, default=False, help='shuffle the data for more random result')
    parser.add_argument('--subj_name',type = str,default=subj_name) # the subject name
    args = parser.parse_args()
    model = Model(args)
    # pdb.set_trace()

    # specify arguments - training specifying parameters - usuage instructions

    if args.mode == 'train' or args.mode == 'all': 
      model.train()
    if args.mode == 'test' or args.mode == 'all':
      model.test()

    print("------------------------------------")
    print("Congratultions! You have reached the end of the script.")
    print("------------------------------------")

