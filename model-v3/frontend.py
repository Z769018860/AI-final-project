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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2" #(if you want to use GPU2)

import sys
import math
import numpy as np
import tensorflow as tf

class MyNetwork():
    def __init__(self, params):
        # Define your placeholders with type, size and name.
        self.vocsize        = params['vocsize']         # vocabulary size
        self.max_seq_size   = params['max_seq_size']    # maximum of sequence length
        self.embedsize      = params['embedsize']       # word embedding size
        self.n_hidden       = params['n_hidden']        # num of LSTM cells
        self.X          = tf.placeholder(tf.int32, [None, self.max_seq_size], name='X')
        self.seq_len    = tf.placeholder(tf.int32, [None], name='seq_len')
        self.y          = tf.placeholder(tf.int32, [None, self.max_seq_size], name='y')
        self.hidden = None
        self.init_variables()
        
    def init_variables(self):
        # Define your variables
        # HINT: or you can directly use existed functions.
        self.wordvec = tf.Variable(tf.random_normal([self.vocsize, self.embedsize]))
    
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
        x_embed = tf.one_hot(self.X, self.vocsize, dtype=tf.float32)     # [None, max_seq_size, vocsize]
        x_embed = tf.einsum('abi,ij->abj', x_embed, self.wordvec)   # [None, max_seq_size, embedsize]
        
        # 2. bidirectional LSTM
        cell_fw = tf.contrib.rnn.LSTMCell(self.n_hidden)
        cell_bw = tf.contrib.rnn.LSTMCell(self.n_hidden)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x_embed, self.seq_len, dtype=tf.float32)
        # [2, None, max_seq_size, n_hidden]
        
        # 3. softmax
        self.softmax_w = tf.get_variable("softmax_w", [2*self.n_hidden, 4], dtype=tf.float32)
        self.softmax_b = tf.get_variable("softmax_b", [4], dtype=tf.float32)
        hidden = tf.concat(outputs, -1)                             # [None, max_seq_size, 2*n_hidden]
        hidden = tf.einsum('abi,ij->abj', hidden, self.softmax_w) + self.softmax_b
        # [None, max_seq_size, 4]

        probs = self.prob_layer(hidden)
        self.y_pred = tf.argmax(probs, -1)
        self.hidden = hidden
        return self.y_pred, self.hidden
    
    def get_loss(self, hidden):
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.y, hidden)
        return self.loss
    
    def gen_input(self, data_x, data_y):
        feed_dict = {}
        # padding
        feed_dict[self.X] = [x+[0]*(self.max_seq_size-len(x)) for x in data_x]
        feed_dict[self.y] = [y+[0]*(self.max_seq_size-len(y)) for y in data_y]
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

### default parameters for model
default_params = {
    'vocsize': 2840, #len(idx2word),
    'max_seq_size': 256,
    'embedsize': 256,
    'n_hidden': 512,
    'learning_rate': 0.001,
    'optimizer': 'adam'
}

class MyModel():
    def __init__(self, params=default_params):
        self.sess = tf.Session()
        self.network = MyNetwork(params)  # Classifier
        self.train_output = self.get_model_train()  # y_pred, loss
        self.test_output = self.get_model_test()  # y_pred
        
        self.optimizer = get_optimizer(params['learning_rate'], params['optimizer']).minimize(self.train_output[-1])
        self.sess.run(tf.global_variables_initializer())
        self.saver = None
        self.init_saver()
    
    def errors(self, y_pred, y_truth):
        err = 0.0
        for truth, pred in zip(y_truth, y_pred):
            if not (truth == pred[:len(truth)]).all():
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
        self.saver.restore(self.sess, path)
    
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
