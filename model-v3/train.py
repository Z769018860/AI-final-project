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

import numpy as np

import frontend

### load sentences
import json

sentence_list_fname = './assignment_training_data_word_segment.json'

# idx2word is in the following form:
# 0 $TIME$
# 1 $ATTRIBUTE$
# 2 $VALUE$
# ...

# Output of the network uses a vocabulary in this form:
# 0 $TIME$
# 1 $ATTRIBUTE$
# 2 $VALUE$
# 3 $OTHER$
# where 0, 1, 2 represent useful time, attribute and value respectively

def load_train_data(path):
    sentence_list = json.load(open(path , 'r'))
    data_x = [data['indexes'] for data in sentence_list]
    data_y = []
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
        data_y.append(indexes)
    return list(zip(data_x, data_y))

def train(model, train_set, test_set, n_epochs=20, batchsize=500, iterations=5):
    print("Now training.")
    test_set_x, test_set_y = zip(*test_set)

    for epoch in range(n_epochs):
        # shuffle train set in each epoch
        per = np.random.permutation(train_set)

        for i in range(0, len(train_set), batchsize):
            train_batch_x, train_batch_y = zip(*per[i:i+batchsize])

            for j in range(iterations):
                train_pred, loss = model.call_model(train_batch_x, train_batch_y, 'train')
                train_error = model.errors(y_pred=train_pred, y_truth=train_batch_y)

                test_pred = model.call_model(
                    test_set_x,  test_set_y, 'test')
                test_error = model.errors(
                    y_pred=test_pred, y_truth=test_set_y)

                print ("epoch = %d, batch_start = %d, iteration = %d, train error %f, test error %f" % (
                    epoch, i, j, train_error, test_error))
                
                # skip this batch if train error is 0
                if train_error == 0:
                    break

    return train_pred

train_file_name = './assignment_training_data_word_segment_0-2499.json'
test_file_name = 'assignment_training_data_word_segment_2500-2999.json'

if __name__=="__main__":
    model = frontend.MyModel()
    train_set = load_train_data(train_file_name)
    test_set = load_train_data(test_file_name)
    train(model, train_set, test_set)
    model.save_model('./default.ckpt')
