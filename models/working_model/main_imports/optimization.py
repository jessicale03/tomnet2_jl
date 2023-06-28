"""
This file contains functions - 

def proportion_accuracy(self, prediction_array, labels, mode):
def loss(self, logits, labels):
def top_k_error(self, predictions, labels, k):
def train_operation(self, global_step, total_loss, top1_error):
def validation_op(self, validation_step, top1_error, loss):

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



def proportion_accuracy(self, prediction_array, labels, mode):
    '''
    Evaluate model accuracy defined by proportion (num_matches/num_total).
    Return a df that contains the accuracy metric.

    Args:
      :param prediction_array: a tensor with (num_batches * batch_size, num_classes).
      :param labels: in-batch labels. Note that only in-batch labels (size = length)
        are tested because they have corresponding predicted labels.
      :param mode: should be either 'vali' or 'test'
    Returns:
      :df_summary: a a dataframe that stores the acuuracy metrics
    '''
    total_predictions = len(prediction_array)
    # match_predictions
    predicted_labels = np.argmax(prediction_array,1)

    # Retrieve corresponding labels
    groud_truth_labels = labels.astype(int)
    # pdb.set_trace()
    match_predictions = sum(predicted_labels == groud_truth_labels)

    matches_percentage = str(match_predictions) + '/' + str(total_predictions)
    accuracy = str(round(match_predictions*100/total_predictions, 2)) + '%'

    print('\n' + str(mode)+ ': proportion_accuracy()')
    print('Matches: ' + matches_percentage)
    print('Accuracy: ' + accuracy)

    df_summary = pd.DataFrame(data={'matches':matches_percentage,
                                    'accurary':accuracy,
                                    'mode': str(mode + '_proportion')},
                        index = [0])
    return df_summary


      def loss(self, logits, labels):
    '''
    Calculate the cross entropy loss given logits and true labels
    :param logits: 2D tensor with shape [batch_size, num_labels]
    :param labels: 1D tensor with shape [batch_size]
    :return: loss tensor with shape [1]
    '''
    labels = tf.cast(labels, tf.int64)

    # Note
    # (1) https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
    # WARNING: This op expects unscaled logits,
    # since it performs a softmax on logits internally for efficiency.
    # Do not call this op with the output of softmax, as it will produce incorrect results.
    # (2) The ToMNET paper also uses softmax cross entropy for loss function
    # https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-softmax-crossentropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


  def top_k_error(self, predictions, labels, k):
    '''
    Calculate the top-k error
    :param predictions: 2D tensor with shape [batch_size, num_labels]
    :param labels: 1D tensor with shape [batch_size, 1]
    :param k: int
    :return: tensor with shape [1]
    '''
    # -----------
    # The Top-1 error is the percentage of the time that the classifier
    # did not give the correct class the highest score. The Top-5 error
    # is the percentage of the time that the classifier did not include
    # the correct class among its top 5 guesses.
    # -----------

    # predictions:
    # Tensor("Softmax_1:0", shape=(16, 4), dtype=float32)
    batch_size = predictions.get_shape().as_list()[0]

    # in_top1
    # Tensor("ToFloat_1:0", shape=(16,), dtype=float32)
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))

    # num_correct
    # Tensor("Sum_1:0", shape=(), dtype=float32)
    num_correct = tf.reduce_sum(in_top1)
    # print('predictions:')
    # print(predictions)
    # print('in_top1')
    # print(in_top1)
    # print('num_correct')
    # print(num_correct)
    return (batch_size - num_correct) / float(batch_size)


  def train_operation(self, global_step, total_loss, top1_error):
    '''
    Defines train operations

    :param global_step: tensor variable with shape [1]
    :param total_loss: tensor with shape [1]
    :param top1_error: tensor with shape [1]
    :return: two operations. Running train_op will do optimization once. Running train_ema_op
      will generate the moving average of train error and train loss for tensorboard
    '''
    # Add train_loss, current learning rate and train error into the tensorboard summary ops
    tf.summary.scalar('learning_rate', self.lr_placeholder)
    tf.summary.scalar('train_loss', total_loss)
    tf.summary.scalar('train_top1_error', top1_error)

    # The ema object help calculate the moving average of train loss and train error
    ema = tf.train.ExponentialMovingAverage(self.TRAIN_EMA_DECAY, global_step)
    train_ema_op = ema.apply([total_loss, top1_error])
    tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
    tf.summary.scalar('train_loss_avg', ema.average(total_loss))

    opt = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder)
    train_op = opt.minimize(total_loss)
    return train_op, train_ema_op




  def validation_op(self, validation_step, top1_error, loss):
    '''
    Defines validation operations

    :param validation_step: tensor with shape [1]
    :param top1_error: tensor with shape [1]
    :param loss: tensor with shape [1]
    :return: validation operation
    '''

    # This ema object help calculate the moving average of validation loss and error

    # ema with decay = 0.0 won't average things at all. This returns the original error
    ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
    ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)


    val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]), ema2.apply([top1_error, loss]))
    top1_error_val = ema.average(top1_error)
    top1_error_avg = ema2.average(top1_error)
    loss_val = ema.average(loss)
    loss_val_avg = ema2.average(loss)

    # Summarize these values on tensorboard
    tf.summary.scalar('val_top1_error', top1_error_val)
    tf.summary.scalar('val_top1_error_avg', top1_error_avg)
    tf.summary.scalar('val_loss', loss_val)
    tf.summary.scalar('val_loss_avg', loss_val_avg)

    return val_op
