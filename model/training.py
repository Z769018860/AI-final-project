# coding: utf-8
# ============================================================================
#   Copyright (C) 2017 All rights reserved.
#
#   filename : Assignment_template.py
#   author   : chendian / okcd00@qq.com
#   date     : 2018-11-15
#   desc     : Tensorflow Tuple Extraction Tutorial
# ============================================================================

from __future__ import print_function
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ["CUDA_VISIBLE_DEVICES"] = "2" #(if you want to use GPU2)

import sys
import math

import numpy as np
import tensorflow as tf
'''
import random

import frontend

### load sentences and vocabulary
import json
import pickle

sentence_list_fname = './assignment_training_data_word_segment.json'
sentence_list = json.load(open(sentence_list_fname , 'r'))

voc_dict_fname = './voc-d.pkl'
voc_dict = pickle.load(open(voc_dict_fname, 'rb'))
idx2word, word2idx = voc_dict['idx2word'], voc_dict['word2idx']

# idx2word:
# 0 $TIME$
# 1 $ATTRIBUTE$
# 2 $VALUE$
# ...

# Output of the network uses another vocabulary in this form:
# 0 $TIME$
# 1 $ATTRIBUTE$
# 2 $VALUE$
# 3 $OTHER$

### convert training data

train_data_x = [data['indexes'] for data in sentence_list]
train_data_y = []
for data in sentence_list:
    indexes = [3]*len(data['indexes'])
    useful_times            = [triple[0] for triple in data['results']]
    useful_attributes       = [triple[1] for triple in data['results']]
    useful_values           = [triple[2] for triple in data['results']]
    for i in useful_times:
        indexes[i] = 0
    for i in useful_attributes:
        indexes[i] = 1
    for i in useful_values:
        indexes[i] = 2
    train_data_y.append(indexes)

### default parameters
default_params = {
    'vocsize': len(idx2word),
    'max_seq_size': 256,
    'embedsize': 256,
    'n_hidden': 256,
    'learning_rate': 0.1,
    'optimizer': 'adam'
}

def train(model, n_epochs):
    epoch = 0
    print("Now training.")
    while epoch < n_epochs:
        test_set_x = train_data_x[2500:3000]
        test_set_y = train_data_y[2500:3000]
        train_set = list(zip(train_data_x[0:2500], train_data_y[0:2500]))
        train_set_x, train_set_y = zip(*random.sample(train_set, 200))

        # print error/cost per epoch
        train_pred, loss = model.call_model(train_set_x, train_set_y, 'train')
        train_error = model.errors(y_pred=train_pred, y_truth=train_set_y)

        test_pred = model.call_model(
            test_set_x,  test_set_y, 'test')
        test_error = model.errors(
            y_pred=test_pred, y_truth=test_set_y)

        print ("epoch is %d, train error %f, test error %f" % (
            epoch, train_error, test_error))
        epoch += 1
    return train_pred

if __name__=="__main__":
    model = frontend.MyModel(default_params)
    train(model, 100)
    model.save_model('./default.ckpt')
