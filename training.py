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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ["CUDA_VISIBLE_DEVICES"] = "2" #(if you want to use GPU2)

import sys
import math
import json
import pickle
import numpy as np
import tensorflow as tf

### load sentences and vocabulary

sentence_list_fname = './assignment_training_data_word_segment.json'
sentence_list = json.load(open(sentence_list_fname , 'r'))

voc_dict_fname = './voc-d.pkl'
voc_dict = pickle.load(open(voc_dict_fname, 'rb'))
idx2word, word2idx = voc_dict['idx2word'], voc_dict['word2idx']

vocsize = len(idx2word)

# idx2word:
# 0 $TIME$
# 1 $ATTRIBUTE$
# 2 $VALUE$
# ...

# Output of the network uses an extended vocabulary in this form:
# 0 $TIME$
# 1 $ATTRIBUTE$
# 2 $VALUE$
# ...
# vocsize+0 $REDUNDANT_TIME$
# vocsize+1 $REDUNDANT_ATTRIBUTE$
# vocsize+2 $REDUNDANT_VALUE$

exvocsize = vocsize+3

### convert training data

import copy

train_set_x = [data['indexes'] for data in sentence_list]
train_set_y = []
for data in sentence_list:
    indexes = copy.copy(data['indexes'])
    useful_times            = [triple[0] for triple in data['results']]
    useful_attributes       = [triple[1] for triple in data['results']]
    useful_values           = [triple[2] for triple in data['results']]
    redundant_times         = [i for i in data['times'] if i not in useful_times]
    redundant_attributes    = [i for i in data['attributes'] if i not in useful_attributes]
    redundant_values        = [i for i in data['values'] if i not in useful_values]
    for i in redundant_times:
        indexes[i] = vocsize+0
    for i in redundant_attributes:
        indexes[i] = vocsize+1
    for i in redundant_times:
        indexes[i] = vocsize+2
    train_set_y.append(indexes)

max_seq_size = max([len(x) for x in train_set_x])

test_set_x = train_set_x[200:400]
test_set_y = train_set_y[200:400]
train_set_x = train_set_x[0:200]
train_set_y = train_set_y[0:200]

### Parameters for network

# word embedding size
embedsize = 256
# num of LSTM cells
n_hidden = 256

learning_rate = 0.1
n_epochs = 50

class MyNetwork():
    def __init__(self, other_params=None):
        # Define your placeholders with type, size and name.
        # variable sequence count, variable sequence length
        self.X          = tf.placeholder(tf.int32, [None, max_seq_size], name='X')
        self.seq_len    = tf.placeholder(tf.int32, [None], name='seq_len')
        self.y          = tf.placeholder(tf.int32, [None, max_seq_size], name='y')
        self.hidden = None
        self.init_variables()
        
    def init_variables(self):
        # Define your variables
        # HINT: or you can directly use existed functions.
        self.wordvec = tf.Variable(tf.random_normal([vocsize, embedsize]))
    
    def sigmoid(self, logits):
        return tf.nn.sigmoid(logits)
    
    def softmax(self, logits):
        # softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
        return tf.nn.softmax(logits, -1)
    
    def prob_layer(self, hidden, mask=None, expand_dim=False):
        # calculate probability from hidden
        prob = self.sigmoid(hidden)# * mask
        if expand_dim:
            # twins_prob = tf.concat([prob, 1.0 - prob], -1)
            return tf.expand_dims(prob, -1)
        return prob
    
    def get_network(self):
        if self.hidden is not None:
            return self.y_pred, self.hidden
        
        # 1. word embedding
        x_embed = tf.one_hot(self.X, vocsize, dtype=tf.float32)     # [None, max_seq_size, vocsize]
        x_embed = tf.einsum('abi,ij->abj', x_embed, self.wordvec)   # [None, max_seq_size, embedsize]
        
        # 2. bidirectional LSTM
        cell_fw = tf.nn.rnn_cell.LSTMCell(n_hidden)
        cell_bw = tf.nn.rnn_cell.LSTMCell(n_hidden)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x_embed, self.seq_len, dtype=tf.float32)
        # [2, None, None, n_hidden]
        
        # 3. softmax
        softmax_w = tf.get_variable("softmax_w", [2*n_hidden, vocsize], dtype=tf.float32)
        hidden = tf.concat(outputs, -1)                         # [None, max_seq_size, 2*n_hidden]
        hidden = tf.einsum('abi,ij->abj', hidden, softmax_w)    # [None, max_seq_size, vocsize]

        probs = self.prob_layer(hidden)
        self.y_pred = tf.argmax(probs, -1, output_type=tf.int32)
        self.hidden = hidden
        return self.y_pred, self.hidden
    
    def get_loss(self, hidden):
        y_onehot = tf.one_hot(self.y, vocsize, dtype=tf.float32)
        self.loss = tf.losses.softmax_cross_entropy(y_onehot, hidden)
        return tf.reduce_mean(self.loss)
    
    def gen_input(self, data_x, data_y):
        feed_dict = {}
        # padding
        feed_dict[self.X] = [x+[0]*(max_seq_size-len(x)) for x in data_x]
        feed_dict[self.y] = [y+[0]*(max_seq_size-len(y)) for y in data_y]
        feed_dict[self.seq_len] = [len(x) for x in data_x]
        return feed_dict

def get_optimizer(learning_rate, optim=None):
    # get optimizer for training, AdamOptimizer as default
    #optim = self.options.get('optimizer', 'N/A')
    if optim == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optim == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif optim == 'adadelta':
        return tf.train.AdadeltaOptimizer()
    else:  # if optim == 'adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate)

class MyModel():
    def __init__(self, other_params=None):
        self.sess = tf.Session()
        self.network = MyNetwork()  # Classifier
        self.train_output = self.get_model_train()  # y_pred, loss
        self.test_output = self.get_model_test()  # y_pred
        
        self.optimizer = get_optimizer(learning_rate, 'adam').minimize(self.train_output[-1])
        self.sess.run(tf.global_variables_initializer())
        self.saver = None
        self.init_saver()
    
    def errors(self, y_pred, y_truth=None):
        err = 0.0
        for i in range(len(y_truth)):
            t_indexes = [[], [], []]
            p_indexes = [[], [], []]
            for j in range(len(y_truth[i])):
                if y_truth[i][j] < 3:
                    t_indexes[y_truth[i][j]].append(j)
                if y_pred[i][j] < 3:
                    p_indexes[y_pred[i][j]].append(j)
            if t_indexes != p_indexes:
                err += 1
        err /= len(y_truth)
        return err
    
    def get_model_train(self):
        with tf.name_scope('train'):
            y_pred, hidden = self.network.get_network()
            loss = self.network.get_loss(hidden)
            return y_pred, loss
    
    def get_model_test(self):
        with tf.name_scope('test'):
            y_pred, hidden = self.network.get_network()
            return y_pred
    
    def init_saver(self, var_list=None):
        if self.saver is not None:
            return
        self.saver = tf.train.Saver(
            var_list = tf.global_variables(),
            reshape=True,
            sharded=False,
            restore_sequentially=True,
            write_version=tf.train.SaverDef.V2)
    
    def save_model(self, path):
        if self.saver is None:
            self.init_saver()
        # save session in file
        self.saver.save(self.sess, save_path=path)
    
    def load_model(self, path):
        if self.saver is None:
            self.init_saver()
        self.saver.restore(self.session, path)
    
    def call_model(self, data_x, data_y, mode='train'):
        # generate data for placeholder
        if mode == 'test':
            ret = self.sess.run(  # return y_pred
                self.test_output,
                feed_dict=self.network.gen_input(data_x, data_y))
        else:  # mode == 'train'
            _, ret = self.sess.run(  # return y_pred, loss
                [self.optimizer, self.train_output], 
                feed_dict=self.network.gen_input(data_x, data_y))
        return ret

def run_epoch(model, n_epochs):
    epoch = 0
    print("Now training.")
    while epoch < n_epochs:
        # draw a figure every 'draw_freq' times

        # print error/cost per epoch
        train_pred, loss = model.call_model(train_set_x, train_set_y, 'train')
        train_error = model.errors(y_pred=train_pred, y_truth=train_set_y)

        test_pred = model.call_model(
            test_set_x,  test_set_y, 'test')
        test_error = model.errors(
            y_pred=test_pred, y_truth=test_set_y)

        print ("epoch is %d, train error %f, test error %f" % (
            epoch, train_error, test_error))
        #print ("epoch is %d, train error %f" % (epoch, train_error))
        epoch += 1
    return train_pred


if __name__=="__main__":
    # call model class for a instance
    model = MyModel()

    # train_set_x, train_set_y, test_set_x,  test_set_y = ...
    # HINT: X-Fold Cross Validation.

    run_epoch(model, n_epochs)

