import logging
import multiprocessing
import os.path
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
from sklearn.decomposition import PCA
from matplotlib import pyplot
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
    words=[]
    results=[]
    times = []
    attributes = []
    values = []
    ids=[]
    f = open (r'words.txt','w',encoding='utf-8')
    for dic in sentence_list[:]:
        ids.append(dic['sentenceId'])
        sentences.append(dic['indexes'])
        words.append(dic['words'])
        times.append(dic['times'])
        attributes.append(dic['attributes'])
        values.append(dic['values'])
        t_list = []
        for t in dic['times']:
            for a in dic['attributes']:
                for v in dic ['values']:
                    t_list.append([t,a,v])
        results.append(t_list)
    
        #将所有words转换成字符串输入
        for word in dic['words']:
            print(word," ",end="",file=f)
        print("")

    f.close()

#-----------------------------------my lod data--------------------------------
## draw picture
def plot_with_labels(low_dim_embs, labels, filename='tsne_gensim.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)


if __name__ == '__main__':
    my_load_data()

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # check and process input arguments
    # if len(sys.argv) < 4:
    #     print(globals()['__doc__'] % locals())
    #     sys.exit(1)
    # input_dir, outp1, outp2 = sys.argv[1:4]
    input_dir = 'words.txt'
    outp1 = 'model/word2vec.model'
    outp2 = 'model/word2vec_format'
    fileName = 'words.txt'
    # 训练模型 输入语料目录 embedding size 256,共现窗口大小10,去除出现次数5以下的词,多线程运行,迭代10次
    model = Word2Vec(PathLineSentences(input_dir),
                     size=256, window=10, min_count=5,
                     workers=multiprocessing.cpu_count(), iter=10)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)

    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.style.use('ggplot') #使用'ggplot'风格美化显示的图表
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # 可视化展示
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    count=0
    for i, word in enumerate(words):
	    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()



