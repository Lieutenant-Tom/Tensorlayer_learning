#_*_coding:utf8_*_
'''
Created on 2017年2月23日

@author: Tom
'''

import re
import itertools
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import merge,Input
from keras.models import Model
from gensim.models import word2vec
from os.path import join, exists, split
import os
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

#将句子中的特殊符号进行相应处理
def pre_sentence(sentence):
        sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
        sentence = re.sub(r"\'s", " \'s", sentence)
        sentence = re.sub(r"\'ve", " \'ve", sentence)
        sentence = re.sub(r"n\'t", " n\'t", sentence)
        sentence = re.sub(r"\'re", " \'re", sentence)
        sentence = re.sub(r"\'d", " \'d", sentence)
        sentence = re.sub(r"\'ll", " \'ll", sentence)
        sentence = re.sub(r",", " , ", sentence)
        sentence = re.sub(r"!", " ! ", sentence)
        sentence = re.sub(r"\(", " \( ", sentence)
        sentence = re.sub(r"\)", " \) ", sentence)
        sentence = re.sub(r"\?", " \? ", sentence)
        sentence = re.sub(r"\s{2,}", " ", sentence)
        return sentence.strip().lower()


def pad_sentences(sequence_length, sentences, tests, padding_word=" "):
    padded_sentences = []
    padded_test = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence.split(" "))
        new_sentence = sentence + padding_word * num_padding
        padded_sentences.append(new_sentence)
    for i in range(len(tests)):
        test = tests[i]
        num_padding = sequence_length - len(test.split(" "))
        new_test = test + padding_word * num_padding
        padded_test.append(new_test)
    print("pad_sentences finished")
    return padded_sentences, padded_test


def load_data_and_labels(train_name, test_name):
    dict = {"Other": 0, 
            "Cause-Effect(e1,e2)": 1, "Cause-Effect(e2,e1)": 2,
            "Product-Producer(e1,e2)": 3, "Product-Producer(e2,e1)": 4, 
            "Entity-Origin(e1,e2)": 5, "Entity-Origin(e2,e1)": 6,
            "Instrument-Agency(e1,e2)": 7, "Instrument-Agency(e2,e1)": 8,
            "Component-Whole(e1,e2)": 9, "Component-Whole(e2,e1)": 10,
            "Content-Container(e1,e2)": 11, "Content-Container(e2,e1)": 12,
            "Entity-Destination(e1,e2)": 13, "Entity-Destination(e2,e1)": 14,
            "Member-Collection(e1,e2)": 15, "Member-Collection(e2,e1)": 16,
            "Message-Topic(e1,e2)": 17, "Message-Topic(e2,e1)": 18}
    data_ori = open(train_name).readlines()
    data_ori = [s.strip() for s in data_ori]
    data_ori_x=[]
    data_ori_y=[]
    for s in data_ori:
        s = s.split(" ", 1)
        data_ori_x.append(s[1])
        data_ori_y.append(dict[s[0]])
    
    test_ori = open(test_name).readlines()
    test_ori = [s.strip() for s in test_ori]
    test_ori_x=[]
    test_ori_y=[]
    for s in test_ori:
        s = s.split(" ", 1)
        test_ori_x.append(s[1])
        test_ori_y.append(dict[s[0]])
    print("load_data_and_labels finished")
    return data_ori_x,data_ori_y,test_ori_x,test_ori_y


#构建所有词的集合，以及字典
def build_vocab(sentences):
    vocab_set = set()
    for s in sentences:
        vocab_set = vocab_set.union(set(s.split(" ")))
    vocabulary_inv = [x for x in vocab_set]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv


#根据词字典，返回数据集相对应的数字集合，返回标签集
def build_input_data(sentences, labels, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence.split(" ")] for sentence in sentences])
    y = np.array(labels)
    print("build_input_data finished")
    return x,y


#生成训练集，标签，所有词的集合， 所有词的字典
def load_data(train_name, test_name):
    sentences, labels, test, key = load_data_and_labels(train_name, test_name)
    sentences_length = max(len(x.split(" ")) for x in sentences)
    test_length = max(len(x.split(" ")) for x in test)
    sequence_length = max(sentences_length, test_length)
    sentences_padded, test_padded = pad_sentences(sequence_length, sentences, test)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded + test_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    test, key = build_input_data(test_padded, key, vocabulary)
    return x,y,test, key,vocabulary,vocabulary_inv, sequence_length


#将所有单词转换为词向量
def train_word2vec(vocabulary_inv):
    embedding_model = word2vec.Word2Vec.load('text8.model')
    embedding_weight = [np.array([embedding_model[w] if w in embedding_model \
                                        else np.random.uniform(-0.25, 0.25, embedding_model.vector_size) \
                                        for w in vocabulary_inv])]
    return embedding_weight

class_num = 19
output_dim = 300
hiddenlayer_num = 256
dropout = 0.5
batch_size = 32
nb_epoch = 1


if __name__=='__main__':
    #预处理数据集等
    X_train, y_train, X_test, y_test, vocabulary, vocabulary_inv, sequence_length = load_data("train_labels_and_features.txt", "test_labels_and_features.txt")
    embedding_weights = train_word2vec(vocabulary_inv)
    y_train = np_utils.to_categorical(y_train, class_num)
    y_test_pre = y_test.tolist()
    y_test = np_utils.to_categorical(y_test, class_num)
    main_input= Input(shape=(sequence_length,), dtype='int32',name='main_input')
    embedded=Embedding(output_dim=output_dim, input_dim=len(vocabulary_inv),input_length=sequence_length,weights=embedding_weights, mask_zero=True)(main_input)
    forwards = LSTM(hiddenlayer_num)(embedded)
    backwards = LSTM(hiddenlayer_num, go_backwards=True)(embedded)
    merged = merge([forwards, backwards], mode='concat')
    after_dp = Dropout(dropout)(merged)
    output = Dense(class_num, activation='softmax')(after_dp)
    model = Model(input=main_input, output=output)
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
 
    checkpointer = ModelCheckpoint(filepath="model/train_labels_and_features.{epoch:02d}-{val_acc:.4f}.hdf5",monitor='val_acc', verbose=1,save_best_only=True,mode='auto',save_weights_only=False)  
    EarlyStopping(monitor='val_acc', patience=3, verbose=1,mode='auto')
    model.fit(X_train, y_train, batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(X_test,y_test), callbacks=[checkpointer])
#     model.load_weights()
    model.save_weights("weights.hdf5")
    model.load_weights("weights.hdf5")
    predict_classes_ori = model.predict(X_test)
#     print(predict_classes_ori)
    predict_classes = [] 
    for sig in predict_classes_ori:
        predict_classes.append(np.argmax(sig))
    print(predict_classes)
    print(y_test_pre)
    print('F1 score : ')
    print(f1_score(y_test_pre,predict_classes,average='macro'))
    print('accuracy_score : ')
    print(accuracy_score(y_test_pre,predict_classes))
    print('recall_score : ')
    print(recall_score(y_test_pre,predict_classes,average='weighted'))
    
#     plt.subplot(1, 1, 1)
#     plt.plot(y_test_pre)
#     plt.plot(predict_classes)
#     plt.show()




#     model = Sequential()
#     model.add(Embedding(output_dim=output_dim, input_dim=len(vocabulary_inv),input_length=sequence_length,weights=embedding_weights, mask_zero=True))
#     model.add(Bidirectional(GRU(hiddenlayer_num)))
#     model.add(Dropout(dropout))
#     model.add(Dense(class_num))
#     model.add(Activation('softmax'))
#     model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1,mode='auto')
#     history = model.fit(X_Val_train, y_Val_train, batch_size=batch_size,nb_epoch=nb_epoch,verbose=2,validation_data=(X_Val_test,y_Val_test))



























































































