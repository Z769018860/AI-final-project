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

#----------------------准确率、召回率、F1--------------------------------
'''
with tf.name_scope("accuracy"):
      correct_predictions = tf.equal(self.predictions,self.input_y)
      self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

with tf.name_scope("tn"):
        tn = tf.metrics.true_negatives(labels=self.input_y, predictions=self.predictions)
        self.tn = tf.reduce_sum(tf.cast(tn, "float"), name="tn")
with tf.name_scope("tp"):
        tp = tf.metrics.true_positives(labels=self.input_y, predictions=self.predictions)
        self.tp = tf.reduce_sum(tf.cast(tn, "float"), name="tp")
with tf.name_scope("fp"):
        fp = tf.metrics.false_positives(labels=self.input_y, predictions=self.predictions)
        self.fp = tf.reduce_sum(tf.cast(fp, "float"), name="fp")
with tf.name_scope("fn"):
       fn = tf.metrics.false_negatives(labels=self.input_y, predictions=self.predictions)
       self.fn = tf.reduce_sum(tf.cast(fn, "float"), name="fn")

with tf.name_scope("recall"):
       self.recall = self.tp / (self.tp + self.fn)
with tf.name_scope("precision"):
       self.precision = self.tp / (self.tp + self.fp)
with tf.name_scope("F1"):
       self.F1 = (2 * self.precision * self.recall) / (self.precision + self.recall)
'''
#-----------------------------------------------------------------------

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
    rule1_count=0
    rule2_count=0
    rule1_pass=0
    rule2_pass=0
    other_pass=0
    for dic in sentence_list[:]:  
    #for j in range(0,3000):
        my_result=[]
        #获得每个的元素个数，从而应用规则
        a_n=len(dic['attributes'])
        t_n=len(dic['times'])
        v_n=len(dic['values'])
        #------rule2:t=a=v--------
        if t_n==a_n==v_n:
            i=0
            rule1_count+=1
            for i in range(t_n):
                my_result.append([dic['times'][i],dic['attributes'][i],dic['values'][i]])
            if sorted(my_result)==sorted(dic['results']):
                rule1_pass+=1
            #print(my_result)
        #----rule1：t*a=v------
        elif t_n*a_n==v_n:   
            i=0    
            rule2_count+=1    
            for a in dic['attributes']:
                for t in dic['times']:
                        #if i<len(dic['values']):
                        #if dic['values'][i]>a:
                    if i<len(dic['values']):
                        my_result.append([t,a,dic['values'][i]])
                        i+=1
                    else:
                        break
                        #print("value:",dic['values'][i])
                        #i+=1
            if sorted(my_result)==sorted(dic['results']):
                rule2_pass+=1
            #print(my_result)
        #-------other----------
        else:
            last_a=0
            last_t=0
            last_v=0
            for a in dic['attributes']:
                for t in dic['times']:
                    if last_t>0 and last_t<a:
                        if t<a and t>last_t:
                        #if i<len(dic['values']):
                        #if dic['values'][i]>a:
                            for v in dic ['values']:
                                if v>last_v and v>a:    #my_result.append([t,a,v])
                                    my_result.append([t,a,v])
                                    last_v=v
                                    last_a=a
                                    break
                        else:
                            last_t=t
                    else:
                        if t<a:
                            #if i<len(dic['values']):
                        #if dic['values'][i]>a:
                            for v in dic ['values']:
                                if v>last_v and v>a:    #my_result.append([t,a,v])
                                    my_result.append([t,a,v])
                                    last_v=v
                                    last_a=a
                                    break
                        else:
                            last_t=t
            if sorted(my_result)==sorted(dic['results']):
                other_pass+=1

        result.append(my_result)

    print("rule1 total:",rule1_count,"\nrule1 pass:",rule1_pass," \nrule2 total:",rule2_count,"\nrule2 pass:",rule2_pass,"\nother total:",3000-rule1_count-rule2_count,"\nother pass:",other_pass)
    return result
#一一对应[t,a,v]
#(time1 time2 time3 attribute1 value11 value12 value13 attribute2 value21 value22 value23)
#对应的result为[time1,attributes1,values1],[time2,attributes1,values2]

#for i in range(0,3000):
#    test_result.append(test_data[i]["results"])

my_result=my_simple_model()

def my_predict(refdata,mydata):
    passcount=0
    f = open (r'E:/school/computer/人工智能/project/rule_out.txt','w')

    for count in range(len(mydata)):
        if sorted(refdata[count])==sorted(mydata[count]):
            print("[%d]PASS!!!!!\n" % count,file=f)
            passcount+=1
        else:
            print("[%d]ERROR!!!!" % count,file=f)
            print("[reference]\n" ,refdata[count],file=f)
            print("[result]\n",mydata[count],file=f)
            print("\n",file=f)
    
    print("%d"%passcount,"/3000 PASS!!!!!!\n",file=f)
    f.close()

if __name__ == '__main__':
    my_predict(test_result,my_result)