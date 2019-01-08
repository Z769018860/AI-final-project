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
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" (if you want to use GPU2)

import sys
import math
import json
import pickle
import numpy as np
import tensorflow as tf
import operator


cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128) # state_size = 128

print(cell.state_size) # 128



inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size

h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)

output, h1 = cell.__call__(inputs, h0) #调用call函数



print(h1.shape) # (32, 128)