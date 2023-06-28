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


from __init__.py import *
from _create_graphs.py import _create_graphs
from train_test_validate.py import *
from optimization.py import *
from class_model import Model

  def _create_graphs(self, with_prednet):
    '''
    Create the graph that includes all tensforflow operations and parameters.

    Args:
      :with_prednet:
        If `with_prednet = True`, then construct the complete model includeing
        both charnet and prednet.
        If `with_prednet = False`, then construct the partial model including
        only the charnet.

    '''

    # > for step in range(self.TRAIN_STEPS):
    # The "step" values will be input to
    # (1)"self.train_operation(global_step, self.full_loss, self.train_top1_error)",
    # and then to
    # (2)"tf.train.ExponentialMovingAverage(self.TRAIN_EMA_DECAY, global_step)"
    # - decay = self.TRAIN_EMA_DECAY
    # - num_updates = global_step #this is where 'global_step' goes

    global_step = tf.Variable(0, trainable=False)
    validation_step = tf.Variable(0, trainable=False)

    #pdb.set_trace()

    # --------------------------------------------------------------
    # Build the model for training and validation
    # --------------------------------------------------------------
    if not with_prednet:
      # The charnet
      # def build_charnet(input_tensor, n, num_classes, reuse, train):
      # - Add n residual layers
      # - Add average pooling
      # - Add LSTM layer
      # - Add a fully connected layer
      # The output of charnet is "logits", which will be feeded into
      # the softmax layer to make predictions

      # "logits" is the output of the charnet (including ResNET and LSTM)
      # and is the input for a softmax layer (see below)
      charnet = cn.CharNet()

      logits = charnet.build_charnet(self.train_data_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=False, train=True)
      # - Use train=True for batch-wise validation along training to make the error metric
      # - comparable to training error
      vali_logits = charnet.build_charnet(self.vali_data_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=True, train=True)

      # Define the placeholder for final targets
      self.train_final_target_placeholder = self.train_labels_traj_placeholder
      self.vali_final_target_placeholder = self.vali_labels_traj_placeholder

    else:
      charnet = cn.CharNet()
      prednet = pn.PredNet()
      length_e_char = length_e_char = self.LENGTH_E_CHAR


      # model for training
      # pdb.set_trace()
      train_e_char = charnet.build_charnet(self.train_data_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=length_e_char, reuse=False, train=True)
      logits = prednet.build_prednet(train_e_char, self.train_data_query_state_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes = self.NUM_CLASS, reuse=False )

      # model for batch-validation along training
      # - Use train=True for batch-wise validation along training to make the error metric
      # - comparable to training error
      vali_e_char = charnet.build_charnet(self.vali_data_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=length_e_char, reuse=True, train=True)
      vali_logits = prednet.build_prednet(vali_e_char, self.vali_data_query_state_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes = self.NUM_CLASS, reuse=True )

      # Define the placeholder for final targets
      self.train_final_target_placeholder = self.train_labels_query_state_placeholder
      self.vali_final_target_placeholder = self.vali_labels_query_state_placeholder

    # --------------------------------------------------------------
    # Define the regularization operation for training
    # --------------------------------------------------------------
    # REGULARIZATION_LOSSES: regularization losses collected during graph construction.
    # See: https://www.tensorflow.org/api_docs/python/tf/GraphKeys
    regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    # --------------------------------------------------------------
    # Define training loss and error
    # Note that for training, regularization is required;
    # however, for validation, regularization is not needed.
    # --------------------------------------------------------------
    #  loss: the cross entropy loss given logits and true labels
    #  > loss(logits, labels)
    # Note:
    # (1) To compute loss, it is important to use the output from NN before entering the softmax function
    # https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
    # WARNING: This op expects unscaled logits,
    # since it performs a softmax on logits internally for efficiency.
    # Do not call this op with the output of softmax, as it will produce incorrect results.
    loss = self.loss(logits, self.train_final_target_placeholder)

    #  tf.add_n: Adds all input tensors element-wise.
    #  - Using sum or + might create many tensors in graph to store intermediate result.
    self.full_loss = tf.add_n([loss] + regu_losses)

    #Validation loss and error
    self.vali_loss = self.loss(vali_logits, self.vali_final_target_placeholder)

    # --------------------------------------------------------------
    # Make prediction based on the output of the model
    # --------------------------------------------------------------
    predictions = tf.nn.softmax(logits, name = 'train_predictions_array')
    vali_predictions = tf.nn.softmax(vali_logits, name = 'vali_predictions_array')

    # --------------------------------------------------------------
    # Define performace metric: prediction error
    # - Note that, by comparison,  the loss function 'def loss(self, logits, labels):'
    # - use the cross entropy loss.
    # --------------------------------------------------------------
    self.train_top1_error = self.top_k_error(predictions, self.train_final_target_placeholder, 1)
    self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_final_target_placeholder, 1)

    # --------------------------------------------------------------
    # Define optimizer
    # --------------------------------------------------------------
    self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss, self.train_top1_error)
    self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)

    return
