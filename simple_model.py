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

#-----------------------------------my lod data--------------------------------
# How to load training data in Python:
import json
sentence_list_fname = 'assignment_training_data_word_segment.json'  # ‘assignment_training_data_word_segment.json’ here
sentence_list = json.load(open(sentence_list_fname , 'r'))

# How to load vocabulary in Python:

import pickle #(in python 3.x)
#import cPickle as pickle # (in python 2.x)

#voc_dict_fname = 'assignment_training_data_word_segment.json'  # provided file is ‘voc.pkl’ here
#voc_dict = pickle.load(open(voc_dict_fname, 'rb'))
#idx2word, word2idx = voc_dict['idx2word', 'word2idx'] 
# idx2word[index] is a word
# word2idx[word] is an index

def my_load_data(path="assignment_training_data_word_segment.json", maxlen=None):
    sentence_list_fname = path
    sentence_list = json.load(open(sentence_list_fname,'r'))
    sentences = []
    results=[]
    times = []
    attributes = []
    values = []
    ids=[]
    for dic in sentence_list[:]:
        ids.append(dic['sentenceId'])
        sentences.append(dic['indexes'])
        times.append(dic['times'])
        attributes.append(dic['attributes'])
        values.append(dic['values'])
        t_list = []
        for t in dic['times']:
            for a in dic['attributes']:
                for v in dic ['values']:
                    t_list.append([t,a,v])
        results.append(t_list)
    train_set = (sentences,results,times,attributes,values,ids)

    return train_set

#-----------------------------------my lod data--------------------------------

test_data=my_load_data

test_result=[]
for dic in sentence_list[:]:
    test_result.append(dic['results'])

def my_simple_model():
    i=0
    result=[]
    count=0;
    for dic in sentence_list[:]:  
    #for j in range(0,3000):
        my_result=[]
        last_v=0
        last_a=0
        training_count=0
        for a in dic['attributes']:
            for t in dic['times']:
                if t<a :
                    #if i<len(dic['values']):
                        #if dic['values'][i]>a:
                    for v in dic ['values']:
                        if v>last_v and v>a:    #my_result.append([t,a,v])
                            my_result.append([t,a,v])
                            last_v=v
                            last_a=a
                            break
                        #print("value:",dic['values'][i])
                        #i+=1
        result.append(my_result)
        #这里做一个调整：如果有多个attributes，而且存在冗余,排除冗余情况
#        while sorted(result)!=sorted(test_result[count]) and training_count<len(dic['attributes'])-1:
#            for a in dic['attributes']:
#                if a>last_a:
#                    for t in dic['times']:
#                        if t<a :
#                            #if i<len(dic['values']):
#                            #if dic['values'][i]>a:
#                            for v in dic ['values']:
#                                if v>last_v and v>a:    #my_result.append([t,a,v])
#                                    my_result.append([t,a,v])
#                                    last_v=v
#                                    last_a=a
#                                    break    
#            result.append(my_result)
#            training_count+=1 

#        count+=1     

    return result
#一一对应[t,a,v]
#(time1 time2 time3 attribute1 value11 value12 value13 attribute2 value21 value22 value23)
#对应的result为[time1,attributes1,values1],[time2,attributes1,values2]

#for i in range(0,3000):
#    test_result.append(test_data[i]["results"])

my_result=my_simple_model()

def my_predict(refdata,mydata):
    passcount=0
    f = open (r'E:/school/computer/人工智能/project/simple_out.txt','w')
    for count in range(len(mydata)):
        if sorted(refdata[count])==sorted(mydata[count]):
            print("[%d]PASS!!!!!\n" % count,file=f)
            passcount+=1
        else:
            print("[%d]ERROR!!!!" % count,file=f)
            print("[reference]\n" ,refdata[count],file=f)
            print("[result]\n",mydata[count],file=f)
            print("\n",file=f)
        count+=1
    
    print("%d"%passcount,"/3000 PASS!!!!!!\n",file=f)
    f.close()

if __name__ == '__main__':
    my_predict(test_result,my_result)