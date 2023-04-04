# This file contains modules to evaluate the partially trained model.
# According to the paper the MAP of PHOCNet for IAM dataset In QbE is 72.51.
#
# Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from load_data import load_data
from save_load_weight import *
from datetime import datetime

from scipy.spatial.distance import pdist, squareform


def mean_average_precision(model, x_test, y_test, transcripts):
    """This module evaluates the partially trained model using Test Data
  Args:
    model: Instance of Sequential Class storing Neural Network
    x_test: Numpy storing the Test Images
    y_test: Numpy storing the PHOC Labels of Test Data
    transcripts: String storing the characters in the Image.
  Returns:
    map: Floating number storing the Mean Average Precision.
  """
    start = datetime.now()
    y_pred = model.predict(x_test)
    y_pred = np.where(y_pred < 0.5, 0, 1)
    print("Time taken to predict all data: ", datetime.now() - start)
    start = datetime.now()
    N = len(transcripts)
    precision = {}
    count = {}
    for i in range(N):
        if transcripts[i] not in precision.keys():
            precision[transcripts[i]] = 1
            count[transcripts[i]] = 0
        else:
            precision[transcripts[i]] += 1

    for i in range(N):
        pred = y_pred[i]
        acc = np.sum(abs(y_test - pred), axis=1)
        tmp = np.argmin(acc)
        if transcripts[tmp] == transcripts[i]:
            count[transcripts[tmp]] += 1

    mean_avg_prec = [0, 0]
    for i in range(N):
        if precision[transcripts[i]] <= 1:
            continue
        mean_avg_prec[0] += count[transcripts[i]] * 1.0 / precision[transcripts[i]]
        mean_avg_prec[1] += 1

    print("Time taken to calculate l2 dist: ", datetime.now() - start)
    print("The Mean Average Precision = ", mean_avg_prec[0] * 1. / mean_avg_prec[1])
    print("Total test cases = ", N)


# load data and corresponding transcripts
#


def map_from_feature_matrix(features, labels, metric, drop_first):
    '''
    Computes mAP and APs from a given matrix of feature vectors
    Each sample is used as a query once and all the other samples are
    used for testing. The user can specify whether he wants to include
    the query in the test results as well or not.

    Args:
        features (2d-ndarray): the feature representation from which to compute the mAP
        labels (1d-ndarray or list): the labels corresponding to the features (either numeric or characters)
        metric (string): the metric to be used in calculating the mAP
        drop_first (bool): whether to drop the first retrieval result or not
    '''
    # argument error checks
    if features.shape[0] != len(labels):
        raise ValueError('The number of feature vectors and number of labels must match')
    # compute the pairwise distances from the
    # features
    dists = pdist(X=features, metric=metric)
    dists = squareform(dists)
    inds = np.argsort(dists, axis=1)
    retr_mat = np.tile(labels, (features.shape[0], 1))

    # compute two matrices for selecting rows and columns
    # from the label matrix
    # -> advanced indexing
    row_selector = np.transpose(np.tile(np.arange(features.shape[0]), (features.shape[0], 1)))
    retr_mat = retr_mat[row_selector, inds]

    # create the relevance matrix
    rel_matrix = retr_mat == np.atleast_2d(labels).T
    if drop_first:
        rel_matrix = rel_matrix[:, 1:]

    # calculate mAP and APs
    map_calc = MeanAveragePrecision()
    avg_precs = np.array([map_calc.average_precision(row) for row in rel_matrix])
    mAP = np.mean(avg_precs)
    return mAP, avg_precs

class IterativeMean(object):
    '''
    Class for iteratively computing a mean. With every new value (@see: _add_value)
    the mean will be updated
    '''

    def __init__(self, mean_init=0.0):
        self.__mean = mean_init
        self.__N = 0.0

    def add_value(self, value):
        '''
        Updates the mean with respect to value

        Args:
            value (float): The value that will be incorporated in the mean
        '''
        self.__mean = (self.__N / (self.__N + 1)) * self.__mean + (1.0 / (self.__N + 1)) * value
        self.__N += 1

    def get_mean(self):
        return self.__mean

    def reset(self):
        self.__mean = 0.0
        self.__N = 0.0


class MeanAveragePrecision(IterativeMean):
    '''
    Computes average precision values and iteratively updates their mean
    '''
    def __init__(self):
        super(MeanAveragePrecision, self).__init__()

    def average_precision(self, ret_vec_relevance, gt_relevance_num=None):
        '''
        Computes the average precision and updates the mean average precision

        Args:
            ret_vec_relevance (1d-ndarray): array containing ground truth (gt) relevance values
            gt_relevance_num (int): The number of relevant samples in retrieval. If None the sum
                                    over the retrieval gt list is used.
        '''
        ret_vec_cumsum = np.cumsum(ret_vec_relevance, dtype=float)
        ret_vec_range = np.arange(1, ret_vec_relevance.size + 1)
        ret_vec_precision = ret_vec_cumsum / ret_vec_range

        if gt_relevance_num is None:
            n_relevance = ret_vec_relevance.sum()
        else:
            n_relevance = gt_relevance_num

        if n_relevance > 0:
            ret_vec_ap = (ret_vec_precision * ret_vec_relevance).sum() / n_relevance
        else:
            ret_vec_ap = 0.0

        super(MeanAveragePrecision, self).add_value(ret_vec_ap)

        return ret_vec_ap



