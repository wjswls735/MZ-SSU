# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:15:44 2018

@author: jbk48
@editor: lumyjuwon
"""

import os
import tensorflow as tf
import Bi_LSTM as Bi_LSTM
import Word2Vec as Word2Vec
import gensim
import numpy as np
import csv

Line=[]
CLine=[]
file=open("dev.txt",'r')
while True:
    nline = file.readline()
    if not nline: break
    Line.append(nline.rstrip('\n'))

for i in Line:
    li = i.split('\t')
    CLine.append(li[1])


def Convert2Vec(model_name, sentence):
    word_vec = []
    sub = []
    model = gensim.models.word2vec.Word2Vec.load(model_name)
    for word in sentence:
        if (word in model.wv.vocab):
            sub.append(model.wv[word])
        else:
            sub.append(np.random.uniform(-0.25, 0.25, 300))  # used for OOV words
    word_vec.append(sub)
    return word_vec

def Grade(sentence):
    tokens = W2V.tokenize(sentence)
    embedding = Convert2Vec('../post.embedding', tokens)
    zero_pad = W2V.Zero_padding(embedding, Batch_size, Maxseq_length, Vector_size)
    global sess
    result = sess.run(prediction, feed_dict={X: zero_pad, seq_len: [len(tokens)]}) # tf.argmax(prediction, 1)이 여러 prediction 값중 max 값 1개만 가져옴
    point = result.ravel().tolist()

    biggest_idx=0
    idx=0
    Tag = key
    for t, i in zip(Tag, point):
#        print(t, round(i * 100, 2),"%")
#        percent = t + str(round(i * 100, 2)) + "%"
        if point[biggest_idx] < i :
            biggest_idx = idx
        idx=idx+1
    print(cnt , Tag[biggest_idx])
    f2.write(Tag[biggest_idx])
    f2.write("\n")
W2V = Word2Vec.Word2Vec()

f = open("../output/train_tag.txt",'r',encoding='utf-8')
key=[]
while True:
    nline = f.readline()
    if not nline: break
    key.append(nline.rstrip('\n'))


Batch_size = 1
Vector_size = 300
Maxseq_length = 500  # Max length of training data
learning_rate = 0.001
lstm_units = 128
num_class = 785
keep_prob = 1.0

X = tf.placeholder(tf.float32, shape = [None, Maxseq_length, Vector_size], name = 'X')
Y = tf.placeholder(tf.float32, shape = [None, num_class], name = 'Y')
seq_len = tf.placeholder(tf.int32, shape = [None])

BiLSTM = Bi_LSTM.Bi_LSTM(lstm_units, num_class, keep_prob)

with tf.variable_scope("loss", reuse = tf.AUTO_REUSE):
    logits = BiLSTM.logits(X, BiLSTM.W, BiLSTM.b, seq_len)
    loss, optimizer = BiLSTM.model_build(logits, Y, learning_rate)

prediction = tf.nn.softmax(logits)  # softmax
saver = tf.train.Saver()
init = tf.global_variables_initializer()
modelName = "../Bi_LSTM"

f2 = open("result.txt","wt",encoding='utf-8')
sess = tf.Session()
sess.run(init)
saver.restore(sess, modelName)

cnt=0
"""
while(cnt<2):
    try:
        s = input("문장을 입력하세요 : ")		
#    if cnt == len(CLine) : break
#    s = CLine[cnt]
        cnt=cnt+1
        Grade(s)
    except:
        pass
"""
while(cnt < len(CLine)):
    try:
#        s = input("문장을 입력하세요 : ")		
        s = CLine[cnt]
        cnt=cnt+1
        Grade(s)
    except:
        pass
