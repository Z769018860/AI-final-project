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

import random

import frontend
import backend

from evaluation import evaluate

from copy import deepcopy

### load sentences
import json

def predict(model, pred_list):
    print("Testing.")
    print("Processing through front end ...")
    num = 0
    # convert to index sequences
    test_data_x = [data['indexes'] for data in pred_list]
    # call frontend
    front_output = model.call_model(test_data_x,  test_data_x, 'test')
    print("Processing through back end ...")
    # convert to fit input format
    input_list = []
    for x in front_output:
        y = {'times':[], 'attributes':[], 'values':[], 'results':[]}
        for i in range(len(x)):
            if x[i] == 0:
                y['times'].append(i)
            elif x[i] == 1:
                y['attributes'].append(i)
            elif x[i] == 2:
                y['values'].append(i)
        input_list.append(y)
    # call backend
    result = backend.my_simple_model(input_list)
    # convert to fit output format
    pred_result = deepcopy(pred_list)
    for r, pr in zip(pred_result, result):
        r['results'] = pr
    print("Predication complete.")
    return pred_result

def checkerror(truth_list, pred_list):
    errlist = []
    for i in range(len(truth_list)):
        if sorted(truth_list[i]['results']) != sorted(pred_list[i]['results']):
            errlist.append(i)
    return errlist

test_file_name = 'assignment_training_data_word_segment_2500-2999.json'
result_file_name = 'result.json'

if __name__=="__main__":
    model = frontend.MyModel()
    model.load_model('./default.ckpt')
    test_list = json.load(open(test_file_name , 'r'))
    pred_result = predict(model, test_list)
    json.dump(pred_result, open(result_file_name, 'w'))
