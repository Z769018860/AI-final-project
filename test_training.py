import logging
import multiprocessing
import os.path
import sys
import gensim 

#from gensim.models import Word2Vec
#from gensim.models.word2vec import PathLineSentences

model=gensim.models.Word2Vec.load('word2vec.model')
print(model['收入'])
print(type(model['收入']))
print(model.most_similar(['收入']))