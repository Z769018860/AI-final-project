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

### load sentences and vocabulary
import json
import pickle

sentence_list_fname = './assignment_training_data_word_segment.json'
sentence_list = json.load(open(sentence_list_fname , 'r'))

voc_dict_fname = './voc-d.pkl'
voc_dict = pickle.load(open(voc_dict_fname, 'rb'))
idx2word, word2idx = voc_dict['idx2word'], voc_dict['word2idx']

### default parameters
default_params = {
    'vocsize': len(idx2word),
    'max_seq_size': 256,
    'embedsize': 256,
    'n_hidden': 256,
    'learning_rate': 0.1,
    'optimizer': 'adam'
}

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

if __name__=="__main__":
    model = frontend.MyModel(default_params)
    model.load_model('./default.ckpt')
    test_list = sentence_list[2500:3000]
    pred_result = predict(model, test_list)
    p, r, A, BC = evaluate(test_list, pred_result)
