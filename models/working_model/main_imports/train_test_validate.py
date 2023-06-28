"""
This file contains functions - 

def train(self):

def test(self):

def evaluate_on_test_set(self):

def evaluate_on_validation_set(self):

def evaluate_on_training_set(self):

def evaluate_whole_data_set(self, files_traj, data_traj, labels_traj, data_query_state, labels_query_state, batch_size, mode, with_prednet):

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


 def train(self):

    print('Start training-----------------')
    # pdb.set_trace()

    #Build graphs
    self._create_graphs(with_prednet = self.WITH_PREDNET)


    # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
    # summarizing operations by running summary_op. Initialize a new session
    saver = tf.train.Saver(tf.global_variables()) # <class 'tensorflow.python.training.saver.Saver'>
    summary_op = tf.summary.merge_all() # <class 'tensorflow.python.framework.ops.Tensor'>

    # initialize_all_variables (from tensorflow.python.ops.variables)
    # is deprecated and will be removed after 2017-03-02.
    # Instructions for updating:
    # Use `tf.global_variables_initializer` instead.

    init = tf.initialize_all_variables() # <class 'tensorflow.python.framework.ops.Operation'>
    # -----------------------
    # Session: This is the start of the tf session
    # -----------------------
    sess = tf.Session()

    # If you want to load from a checkpoint
    if self.USE_CKPT:
      saver.restore(sess, self.ckpt_path)
      print('Restored from checkpoint...')
    else:
      # -----------------------
      # Session: Initialize all the parameters in the sess.
      # See above: "init = tf.initialize_all_variables()"
      # -----------------------
      sess.run(init)

    # This summary writer object helps write summaries on tensorboard
    # this is irrelevant to the error.csv file

    # pdb.set_trace()
    summary_writer = tf.summary.FileWriter(self.path_train, sess.graph)

    # These lists are used to save a csv file at last
    # This is the data for error.csv
    step_list = []
    train_error_list = []
    val_error_list = []

    print('Start training batch by batch...')
    print('----------------------------')
    #pdb.set_trace()

    for step in range(self.TRAIN_STEPS):
      # pdb.set_trace()
      # --------------------------------------------------------------
      # Generate batches for training data
      # --------------------------------------------------------------
      # pdb.set_trace()
      train_batch_data_traj, train_batch_labels_traj,\
      train_batch_data_query_state, train_batch_labels_query_state\
      = self.batch_generator.generate_train_batch(self.train_data_traj,\
                                  self.train_labels_traj,\
                                  self.train_data_query_state,\
                                  self.train_labels_query_state,\
                                  self.BATCH_SIZE_TRAIN)

      # --------------------------------------------------------------
      # Generate batches for validation data
      # --------------------------------------------------------------
      vali_batch_data_traj, vali_batch_labels_traj,\
      vali_batch_data_query_state, vali_batch_labels_query_state\
      = self.batch_generator.generate_vali_batch(self.vali_data_traj,\
                                  self.vali_labels_traj,\
                                  self.vali_data_query_state,\
                                  self.vali_labels_query_state,\
                                  self.BATCH_SIZE_TRAIN)

      #Validate first?
      if step % self.REPORT_FREQ == 0:

        # Comment the block for 'FULL_VALIDATION' as it will not be run anyways
#        if self.FULL_VALIDATION:
#          validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss, top1_error=self.vali_top1_error, vali_data=vali_data, vali_labels=vali_labels, session=sess, batch_data=train_batch_data, batch_label=train_batch_labels)
#
#          vali_summ = tf.Summary()
#          vali_summ.value.add(tag='full_validation_error', simple_value=validation_error_value.astype(np.float))
#          summary_writer.add_summary(vali_summ, step)
#          summary_writer.flush()
#
#        else:
        #pdb.set_trace()
        _, validation_error_value, validation_loss_value = sess.run([self.val_op, self.vali_top1_error, self.vali_loss],\
                                                                    {self.vali_data_traj_placeholder: vali_batch_data_traj,\
                                                                     self.vali_final_target_placeholder: vali_batch_labels_traj,\
                                                                     self.vali_data_query_state_placeholder: vali_batch_data_query_state,\
                                                                     self.vali_labels_query_state_placeholder: vali_batch_labels_query_state,\
                                                                     self.lr_placeholder: self.INIT_LR})

        val_error_list.append(validation_error_value)

      start_time = time.time()

      # Actual training
      # -----------------------------------------------
      # This is where the train_error_value comes from
      # -----------------------------------------------
      # sess.run(
      #     fetches = [self.train_op,
      #                self.train_ema_op,
      #                self.full_loss,
      #                self.train_top1_error],
      #     feed_dict = {self.train_data_traj_placeholder: train_batch_data,
      #                  self.train_final_target_placeholder: train_batch_labels,
      #                  self.vali_data_traj_placeholder: validation_batch_data,
      #                  self.vali_final_target_placeholder: validation_batch_labels,
      #                  self.lr_placeholder: self.INIT_LR})
      # Parameters:
      # -----------------------------
      # fetches
      # -----------------------------
      # (1,2) self.train_op, self.train_ema_op
      # - (1) These define the optimization operation.
      # - (2) come from: def _create_graphs(self):
      #       (1) come from: self.train_operation(global_step, self.full_loss, self.train_top1_error)
      #         - return: two operations.
      #           - Running train_op will do optimization once.
      #           - Running train_ema_op will generate the moving average of train error and
      #             train loss for tensorboard
      #         - param: global_step
      #         - param: self.full_loss:
      #             - The loss that includes both the loss and the regularized loss
      #             - comes from: self.full_loss = tf.add_n([loss] + regu_losses)
      #         - param: self.train_top1_error:
      #             def _create_graphs(self):
      #                self.train_top1_error = self.top_k_error(predictions, self.train_final_target_placeholder, 1)
      #                   def top_k_error(self, predictions, labels, k):
      #                        The Top-1 error is the percentage of the time that the classifier
      #                        did not give the correct class the highest score.
      #
      # (3) self.full_loss
      # - (1) The loss that includes both the loss and the regularized loss
      # - (2) comes from: self.full_loss = tf.add_n([loss] + regu_losses)
      #
      # (4) self.train_top1_error
      # - (1) comes from:
      # - def _create_graphs(self):
      # --- self.train_top1_error = self.top_k_error(predictions, self.train_final_target_placeholder, 1)
      # --- def top_k_error(self, predictions, labels, k):
      # - (2) The Top-1 error is the percentage of the time that the classifier
      #       did not give the correct class the highest score.
      #
      # -----------------------------
      # feed_dict
      # -----------------------------
      # self.train_data_traj_placeholder: train_batch_data
      # - feed in the trajectories of the training batch
      # self.train_final_target_placeholder: train_batch_labels
      # - feed in the labels of the training batch
      # self.vali_data_traj_placeholder: validation_batch_data
      # - feed in the trajectories of the validation batch
      # self.vali_final_target_placeholder: validation_batch_labels
      # - feed in the labels of the validation batch
      # self.lr_placeholder: self.INIT_LR
      # - feed in the initial learning rate
      # pdb.set_trace()
      _, _, train_loss_value, train_error_value = sess.run([self.train_op, self.train_ema_op, self.full_loss, self.train_top1_error],\
                                                           {self.train_data_traj_placeholder: train_batch_data_traj,\
                                                            self.train_final_target_placeholder: train_batch_labels_traj,\
                                                            self.train_data_query_state_placeholder: train_batch_data_query_state,\
                                                            self.train_labels_query_state_placeholder: train_batch_labels_query_state,\
                                                            self.lr_placeholder: self.INIT_LR})
      duration = time.time() - start_time

      if step % self.REPORT_FREQ == 0:
        summary_str = sess.run(summary_op,\
                               {self.train_data_traj_placeholder: train_batch_data_traj,\
                                self.train_final_target_placeholder: train_batch_labels_traj,\
                                self.train_data_query_state_placeholder: train_batch_data_query_state,\
                                self.train_labels_query_state_placeholder: train_batch_labels_query_state,\
                                self.vali_data_traj_placeholder: vali_batch_data_traj,\
                                self.vali_final_target_placeholder: vali_batch_labels_traj,\
                                self.vali_data_query_state_placeholder: vali_batch_data_query_state,\
                                self.vali_labels_query_state_placeholder: vali_batch_labels_query_state,\
                                self.lr_placeholder: self.INIT_LR})
        summary_writer.add_summary(summary_str, step)

        num_examples_per_step = self.BATCH_SIZE_TRAIN # trajectoris per step = trajectoris per batch = batch size
        examples_per_sec = num_examples_per_step / duration # trajectories per second
        sec_per_batch = float(duration) # seconds for this step

        format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
        # pdb.set_trace()
        print(format_str % (datetime.datetime.now(), step, train_loss_value, examples_per_sec, sec_per_batch))
        print('Train top1 error = ', train_error_value)
        print('Validation top1 error = %.4f' % validation_error_value)
        print('Validation loss = ', validation_loss_value)
        print('----------------------------')

        # This records the training steps and the corresponding training error
        step_list.append(step)
        train_error_list.append(train_error_value)

        #print('End of training report-----------------')
        #pdb.set_trace()

      #if step == self.DECAY_STEP_0 or step == self.DECAY_STEP_1:
      #  self.INIT_LR = 0.1 * self.INIT_LR
      #  print('Learning rate decayed to ', self.INIT_LR)

      # Save checkpoints every 10000 steps and also at the last step
      if step % 10000 == 0 or (step + 1) == self.TRAIN_STEPS:
          checkpoint_path = os.path.join(self.path_train, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

          df = pd.DataFrame(data={'step':step_list,\
                                  'train_error':train_error_list,\
                                  'validation_error': val_error_list})
          # overwrite the csv
          df.to_csv(os.path.join(self.path_train, '_error.csv'))



  def test(self):
    '''
    This function is used to evaluate the model based on traing, validation and test data.
    Please finish pre-precessing in advance
    It will write a csv file with both validation and test performance.
    '''
    # --------------------------------------------------------------
    # Evaluate the model on the training set
    # --------------------------------------------------------------
    # pdb.set_trace()
    df_train_all = self.evaluate_on_training_set()

    # --------------------------------------------------------------
    # Evaluate the model on the whole validation set
    # --------------------------------------------------------------
    # pdb.set_trace()
    df_vali_all = self.evaluate_on_validation_set()

    # --------------------------------------------------------------
    # Evaluate the model on the whole test set
    # --------------------------------------------------------------
    # pdb.set_trace()
    df_test_all = self.evaluate_on_test_set()

    # --------------------------------------------------------------
    # My codes
    # Combine all dfs into one
    # --------------------------------------------------------------
    #pdb.set_trace()
    df_train_and_vali = df_train_all.append(df_vali_all)
    df_train_vali_and_test = df_train_and_vali.append(df_test_all)

    df_train_vali_and_test.to_csv(os.path.join(self.path_train,\
                                               '_train_test_and_validation_accuracy.csv'))

    return df_train_vali_and_test




  def evaluate_on_test_set(self):
      '''
      Evaluate a model with the test data (instead of a single batch).
      It will evaluate the data batch-by-batch and summarize the performance.
      It will return a dataframe with model accuracy.

      Returns:
        :df_accuracy_all: a dataframe with model accuracy.
      '''

      df_accuracy_all = self.evaluate_whole_data_set(files_traj = self.test_files_traj,\
                                                     data_traj = self.test_data_traj,\
                                                     labels_traj= self.test_labels_traj,\
                                                     data_query_state = self.test_data_query_state,\
                                                     labels_query_state = self.test_labels_query_state,\
                                                     batch_size = self.BATCH_SIZE_TEST,\
                                                     mode = 'test',
                                                     with_prednet = self.WITH_PREDNET)

      return df_accuracy_all

  def evaluate_on_validation_set(self):
      '''
      Evaluate a model with the validation data (instead of a single batch).
      It will evaluate the data batch-by-batch and summarize the performance.
      It will return a dataframe with model accuracy.

      Returns:
        :df_accuracy_all: a dataframe with model accuracy.
      '''
      df_accuracy_all = self.evaluate_whole_data_set(files_traj = self.vali_files_traj,\
                                                     data_traj = self.vali_data_traj,\
                                                     labels_traj= self.vali_labels_traj,\
                                                     data_query_state = self.vali_data_query_state,\
                                                     labels_query_state = self.vali_labels_query_state,\
                                                     batch_size = self.BATCH_SIZE_VAL,\
                                                     mode = 'vali',
                                                     with_prednet = self.WITH_PREDNET)

      return df_accuracy_all


def evaluate_on_training_set(self):
      '''
      Evaluate a model with the training data (instead of a single batch).
      It will evaluate the data batch-by-batch and summarize the performance.
      It will return a dataframe with model accuracy.

      Returns:
        :df_accuracy_all: a dataframe with model accuracy.
      '''

      df_accuracy_all = self.evaluate_whole_data_set(files_traj = self.train_files_traj,\
                                                     data_traj = self.train_data_traj,\
                                                     labels_traj= self.train_labels_traj,\
                                                     data_query_state = self.train_data_query_state,\
                                                     labels_query_state = self.train_labels_query_state,\
                                                     batch_size = self.BATCH_SIZE_TRAIN,\
                                                     mode = 'train',
                                                     with_prednet = self.WITH_PREDNET)

      return df_accuracy_all

  def evaluate_whole_data_set(self, files_traj, data_traj, labels_traj, data_query_state, labels_query_state, batch_size, mode, with_prednet):
      '''
      Evaluate a model with a set of data (instead of a single batch).
      It will evaluate the data batch-by-batch and summarize the performance.
      It will return a dataframe with model accuracy.

      Args:
        :param files_traj:
          the txt files_traj to be test (only used to compute the number of trajectories)
        :param data_traj:
          the data_traj to be test the model on
          (num_files, MAX_TRAJECTORY_SIZE, height, width, depth_trajectory)
        :param labels_traj:
          If `with_prednet = False`, they are the ground truth labels to
          be test the model on (num_files, 1).
          If `with_prednet = True`, they are ignored.
        :param data_query_state:
          If `with_prednet = True`, it is the query state
          of the new maze (num_files, height, width, depth_query_state).
          If `with_prednet = False`, they are ignored.
        :param labels_query_state:
          If `with_prednet = True`, they are the ground truth labels to
          be test the model on (num_files, 1).
          If `with_prednet = False`, they are ignored.
        :param batch_size:
          the batch size
        :param mode:
          should be either 'vali' or 'test'
        :param with_prednet:
          If `with_prednet = True`, then construct the complete model includeing
          both charnet and prednet.
          If `with_prednet = False`, then construct the partial model including
          only the charnet.
      Returns:
        :df_accuracy_all:
          a dataframe with model accuracy.
      '''
      # pdb.set_trace()

      num_vali_files = len(files_traj)
      num_batches = num_vali_files // batch_size

      print('%i' %num_batches, mode, 'batches in total...')

      if with_prednet:
        # --------------------------------------------------------------
        # Reverse the query state data to break the correspondence
        # between files_query_state and files_trajectory (when using the same
        # set of files) for the model with both charnet and prednet.
        # Otherwise, the performance would be overestimated.
        # --------------------------------------------------------------
        # pdb.set_trace()
        data_query_state = np.flip(data_query_state, 0)
        labels_query_state = np.flip(labels_query_state, 0)

      # --------------------------------------------------------------
      # Model with only charnet
      # --------------------------------------------------------------
      if not with_prednet:
        # Create the image and labels_traj placeholders
        data_traj_placeholder = tf.placeholder(dtype=tf.float32,\
                                          shape=[batch_size,\
                                                 self.MAX_TRAJECTORY_SIZE,\
                                                 self.MAZE_HEIGHT,\
                                                 self.MAZE_WIDTH,\
                                                 self.MAZE_DEPTH_TRAJECTORY])
        # --------------------------------------------------------------
        # Build the graph
        # --------------------------------------------------------------
        charnet = cn.CharNet()
        # train=False -> Not dropout for LSTM
        logits = charnet.build_charnet(data_traj_placeholder,\
                                       n=self.NUM_RESIDUAL_BLOCKS,\
                                       num_classes=self.NUM_CLASS,\
                                       reuse=True,\
                                       train=False)
        # logits = (batch_size, num_classes)
        predictions = tf.nn.softmax(logits)
        # predictions = (batch_size, num_classes)

        # --------------------------------------------------------------
        # Initialize a new session and restore a checkpoint
        # --------------------------------------------------------------
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()
        saver.restore(sess, os.path.join(self.path_train, 'model.ckpt-' + str(self.TRAIN_STEPS-1)))
        print('Model restored from ', os.path.join(self.path_train, 'model.ckpt-' + str(self.TRAIN_STEPS-1)))

        # collecting prediction_array for each batch
        # will be size of (batch_size * num_batches, num_classes)
        data_set_prediction_array = np.array([]).reshape(-1, self.NUM_CLASS)

        # collecting ground truth labels for each batch
        # will be size of (batch_size * num_batches, 1)
        data_set_ground_truth_labels = np.array([]).reshape(-1, )

        # Test by batches

        #pdb.set_trace()
        for step in range(num_batches):
          if step % 10 == 0:
              print('%i batches finished!' %step)
          # pdb.set_trace()
          file_index = step * batch_size

          batch_data_traj, batch_labels_traj,\
          batch_data_query_state, batch_labels_query_state\
          = self.batch_generator.generate_vali_batch(data_traj,\
                                     labels_traj,\
                                     data_query_state,\
                                     labels_query_state,\
                                     batch_size,\
                                     file_index = file_index)
  #        batch_data, batch_labels = self.batch_generator.generate_vali_batch(data, labels, batch_size, file_index)
          # pdb.set_trace()
          batch_prediction_array = sess.run(predictions,\
                                            feed_dict={data_traj_placeholder: batch_data_traj})
          # batch_prediction_array = (batch_size, num_classes)
          data_set_prediction_array = np.concatenate((data_set_prediction_array, batch_prediction_array))
          # vali_set_prediction_array will be size of (batch_size * num_batches, num_classes)
          data_set_ground_truth_labels = np.concatenate((data_set_ground_truth_labels, batch_labels_traj))

      # --------------------------------------------------------------
      # Model with both charnet and prednet
      # --------------------------------------------------------------
      else:
        #pdb.set_trace()
        # Create the image and labels_traj placeholders
        data_traj_placeholder = tf.placeholder(dtype=tf.float32,\
                                          shape=[batch_size,\
                                                 self.MAX_TRAJECTORY_SIZE,\
                                                 self.MAZE_HEIGHT,\
                                                 self.MAZE_WIDTH,\
                                                 self.MAZE_DEPTH_TRAJECTORY])
        data_query_state_placeholder = tf.placeholder(dtype=tf.float32,\
                                                      shape=[batch_size,\
                                                             self.MAZE_HEIGHT,\
                                                             self.MAZE_WIDTH,\
                                                             self.MAZE_DEPTH_QUERY_STATE])
        # --------------------------------------------------------------
        # Build the graph
        # --------------------------------------------------------------
        charnet = cn.CharNet()
        prednet = pn.PredNet()
        length_e_char = mp.ModelParameter.LENGTH_E_CHAR

        # train=False -> Not dropout for LSTM
        e_char = charnet.build_charnet(input_tensor = data_traj_placeholder,\
                                       n = self.NUM_RESIDUAL_BLOCKS,\
                                       num_classes = length_e_char,\
                                       reuse=True,\
                                       train=False)
        logits = prednet.build_prednet(e_char,\
                                       data_query_state_placeholder,\
                                       n=self.NUM_RESIDUAL_BLOCKS,\
                                       num_classes = self.NUM_CLASS,\
                                       reuse=True)
        # logits = (batch_size, num_classes)
        predictions = tf.nn.softmax(logits)
        # predictions = (batch_size, num_classes)

        # --------------------------------------------------------------
        # Initialize a new session and restore a checkpoint
        # --------------------------------------------------------------
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()

        saver.restore(sess, os.path.join(self.path_train, 'model.ckpt-' + str(self.TRAIN_STEPS-1)))
        print('Model restored from ', os.path.join(self.path_train, 'model.ckpt-' + str(self.TRAIN_STEPS-1)))

        # collecting prediction_array for each batch
        # will be size of (batch_size * num_batches, num_classes)
        data_set_prediction_array = np.array([]).reshape(-1, self.NUM_CLASS)

        # collecting ground truth labels for each batch
        # will be size of (batch_size * num_batches, 1)
        data_set_ground_truth_labels = np.array([]).reshape(-1, )

        # Test by batches
        #pdb.set_trace()
        for step in range(num_batches):
          if step % 10 == 0:
              print('%i batches finished!' %step)
          # pdb.set_trace()
          file_index = step * batch_size

          batch_data_traj, batch_labels_traj,\
          batch_data_query_state, batch_labels_query_state\
          = self.batch_generator.generate_vali_batch(data_traj,\
                                                     labels_traj,\
                                                     data_query_state,\
                                                     labels_query_state,\
                                                     batch_size,\
                                                     file_index = file_index)
  #        batch_data, batch_labels = self.batch_generator.generate_vali_batch(data, labels, batch_size, file_index)
          # pdb.set_trace()
          batch_prediction_array = sess.run(predictions,\
                                            feed_dict={data_traj_placeholder: batch_data_traj,\
                                                       data_query_state_placeholder: batch_data_query_state})
          # batch_prediction_array = (batch_size, num_classes)
          data_set_prediction_array = np.concatenate((data_set_prediction_array, batch_prediction_array))
          # vali_set_prediction_array will be size of (batch_size * num_batches, num_classes)
          data_set_ground_truth_labels = np.concatenate((data_set_ground_truth_labels, batch_labels_query_state))
        # Model with both charnet and prednet

      # --------------------------------------------------------------
      # My codes
      # Test accuracy by definition
      # --------------------------------------------------------------
      # pdb.set_trace()
      # vali_set_prediction_array = (num_batches*batch_size, num_classes)
      # vali_set_ground_truth = (num_batches*batch_size, 1)

      df_accuracy_proportion = self.proportion_accuracy(data_set_prediction_array, data_set_ground_truth_labels, mode)

      # --------------------------------------------------------------
      # My codes
      # Combine all dfs into one
      # --------------------------------------------------------------
      #pdb.set_trace()

#      df_vali_all = df_vali_proportion.append(df_vali_match_estimation, ignore_index = True)
      df_accuracy_all = df_accuracy_proportion
      #pdb.set_trace()
      return df_accuracy_all