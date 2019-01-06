# coding=utf8
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

import json

sentence_list_fname = 'assignment_training_data_word_segment.json'
sentence_list = json.load(open(sentence_list_fname , 'r'))

for i in range(len(sentence_list)):
  data = sentence_list[i]
  sentence = data['sentence']
  word = data['words']
  print(i)
  print(sentence)
  for result in data['results']:
    print(word[result[0]]+', '+word[result[1]]+', '+word[result[2]])
  print('')