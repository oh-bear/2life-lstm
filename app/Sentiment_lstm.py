# -*- coding: utf-8 -*-
import yaml
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from sklearn.cross_validation import train_test_split
import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
import keras
# For Reproducibility
np.random.seed(1337)
import jieba
import pandas as pd
import sys
sys.setrecursionlimit(1000000)

# set parameters:
vocab_dim = 100
maxlen = 100
n_iterations = 1
n_exposures = 10
window_size = 7
batch_size = 32
n_epoch = 4
input_length = 100
cpu_count = multiprocessing.cpu_count()
magic_num = ['100']

# 加载训练文件
def loadfile():
    pos=pd.read_excel('data/pos.xls',header=None,index=None)
    neg=pd.read_excel('data/neg.xls',header=None,index=None)
    combined=np.concatenate((pos[0], neg[0]))
    y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))
    return combined,y

# 对句子经行分词，并去掉换行符
def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    # text = [jieba.lcut(document.replace('\n', '')) for document in text]
    # return text
    t_list = []
    for document in text:
        if type(document) is not float and type(document) is not int:
            t_list += [list(jieba.lcut(document.replace('\n', '')))]
        else:
            t_list += magic_num
    return t_list

# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.vocab.keys(),
                            allow_update=True)
        # 所有频数超过10的词语的索引
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        # 所有频数超过10的词语的词向量
        w2vec = {word: model[word] for word in w2indx.keys()}
        def parse_dataset(combined):
            # Words become integers
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        combined= sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec,combined
    else:
        print 'No data provided...'

# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):

    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(combined)
    model.train(combined)
    model.save('lstm_data/Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined

def get_data(index_dict,word_vectors,combined,y):

    # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    n_symbols = len(index_dict) + 1
    #索引为0的词语，词向量全为0
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    #从索引为1的词语开始，对每个词语对应其词向量
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    print x_train.shape,y_train.shape
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test

##定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print 'Defining a Simple Keras Model...'
    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print 'Compiling the Model...'
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    print "Train..."
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch,verbose=1, validation_data=(x_test, y_test))

    print "Evaluate..."
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    yaml_string = model.to_yaml()
    with open('lstm_data/lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('lstm_data/lstm.h5')
    print 'Test score:', score

#训练模型，并保存
def train():
    print 'Loading Data...'
    combined,y=loadfile()
    print 'Loading Data:',len(combined),len(y)
    print 'Tokenising...'
    combined = tokenizer(combined)
    print 'Tokenising:',len(combined),len(y)
    print 'Training a Word2vec model...'
    index_dict, word_vectors,combined=word2vec_train(combined)
    print 'Setting up Arrays for Keras Embedding Layer...'
    n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
    print x_train.shape,y_train.shape
    train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)

# run prediction function
def input_transform(string, filePath):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load(filePath+'/Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined

def lstm_predict(string):
    keras.backend.clear_session()
    return_data = dict()
    ans = np.array([0, 0, 0, 0, 0])
    for i in range(0, 5):
        for j in range(i, 5):
            if i != j:
                print 'loading model......for', i, "to", j
                filePath = "app/" + str(i)+"_"+str(j)       # chz
                with open(filePath+'/lstm.yml', 'r') as f:
                    yaml_string = yaml.load(f)
                model = model_from_yaml(yaml_string)
                print 'loading weights......for', i, "to", j
                model.load_weights(filePath+'/lstm.h5')
                model.compile(loss='binary_crossentropy',
                              optimizer='adam',metrics=['accuracy'])
                data=input_transform(string, filePath)
                data.reshape(1,-1)
                result=model.predict_classes(data)
                if result[0][0] == 1:
                    ans[i]+=1
                else:
                    ans[j]+=1

    print 'loading model......'
    with open('app/lstm_data/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)
    print 'loading weights......'
    model.load_weights('app/lstm_data/lstm.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    data=input_transform(string, "app/lstm_data")
    data.reshape(1,-1)
    # print pos or neg
    result=model.predict_classes(data)
    if result[0][0]==1:
        print string,' positive'
        return_data.update({'mood_type': 'positive'})
    else:
        print string,' negative'
        return_data.update({'mood_type': 'negative'})
    return_data.update({
        'mood_sub_type':{
            "E": ans[0]/10.0,
            "A": ans[1]/10.0,
            "C": ans[2]/10.0,
            "O": ans[3]/10.0,
            "N": ans[4]/10.0
        }
    })

    # print EACON
    print ans
    print "E: ", ans[0]/10.0
    print "A: ", ans[1]/10.0
    print "C: ", ans[2]/10.0
    print "O: ", ans[3]/10.0
    print "N: ", ans[4]/10.0
    if ans.argmax() == 0:
        return_data.update({'mood_sub_result': 'E'})
        print string,'Result is E: 喜悦'
    elif ans.argmax() == 1:
        return_data.update({'mood_sub_result': 'A'})
        print string,'Result is A: 温和'
    elif ans.argmax() == 2:
        return_data.update({'mood_sub_result': 'C'})
        print string,'Result is C: 厌恶'
    elif ans.argmax() == 3:
        return_data.update({'mood_sub_result': 'O'})
        print string,'Result is O: 低落'
    else:
        return_data.update({'mood_sub_result': 'N'})
        print string,'Result is N: 愤怒'

    return return_data

# main is used for model train
if __name__=='__main__':
    train()
    string='酒店的环境非常好，价格也便宜，值得推荐'
    string='屏幕较差，拍照也很粗糙。'
    string='质量不错，是正品 ，安装师傅也很好，才要了83元材料费'
    string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    #lstm_predict(string)
