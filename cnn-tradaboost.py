#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
from gensim import corpora, models, similarities
from keras.preprocessing import sequence
from sklearn.svm import SVC
from collections import defaultdict
from keras.models import Sequential, Model
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input, merge, Merge, Flatten
from keras.layers import Embedding, Masking
from keras.layers import Convolution1D, GlobalMaxPooling1D, MaxPooling1D, LSTM, GRU
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
import random
from gensim.models import word2vec
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json
early_stopping = EarlyStopping(monitor='val_acc', patience=15)
import csv
np.random.seed(1337)

# emotion_dict = {'anger':0, 'disgust':1, 'fear':2, 'guilt':3, 'joy':4, 'sadness':5, 'shame':6}
# nb_classes = 7
patience = 15
num_topics = 8
maxlen = 600
train_num = 1000
folder = ['books', 'dvd', 'electronics', 'kitchen']
unlabel = [4465, 3586, 5681, 5945]
all_words = set([])
DIV = 1000
num_Ts = 50
TRAIN = 2000
diff = 2000
global_maxn = 0

# 读入
def read_data(filename, num):
    corpus, label = [], []
    fr = open(filename, 'r')
    n = num
    for i in xrange(n):
        line = fr.readline().strip().split()
        words = []
        for j in xrange(len(line)-1):
          x = line[j].split(':')
          word = x[0]
          num = int(x[1])
          # if word not in stoplist:
          all_words.add(word)
          # for k in xrange(num):
          words.append(word)
        x = line[len(line)-1].split(':')
        polarity = x[1]
        if polarity == 'positive':
          label.append(1)
        elif polarity == 'negative':
          label.append(0)
        corpus.append(' '.join(words))
    return corpus, label

def read_isear_data(filename):
  fr = csv.reader(open(filename+'.csv', 'rb'))
  data, label = [], []
  for line in fr:
    label.append(emotion_dict[line[0]])
    data.append(line[1])
  return data, label

def Doc(filename, num):
  books_corpus_neg, books_label_neg = read_data(filename+'/negative.review', train_num)
  books_corpus_pos, books_label_pos = read_data(filename+'/positive.review', train_num)
  books_corpus_unlabel, books_label_unlabel = read_data(filename+'/unlabeled.review', num)

  documents = books_corpus_neg + books_corpus_pos + books_corpus_unlabel
  labels = books_label_neg + books_label_pos + books_label_unlabel
  # print len(documents)
  # corpus, label = read_data('books/positive.review', train_num, unlabel[0], 'neg')
  # print len(all_words)
  # print len(documents)
  return documents, labels

def Sequence(documents):
  stoplist = set('for a of the and to i in not'.split())
  texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
  # print texts
  frequency = defaultdict(int)
  
  for text in texts:
    for token in text:
      frequency[token] += 1
  texts = [[token for token in text if frequency[token] > 10 and len(token) > 2] # 根据词频停词
        for text in texts]

  # Embedding = word2vec.Word2Vec(texts, size=size, window=5, min_count=1, workers=4)
  # print Embedding

  # dictionary = corpora.Dictionary(texts)
  # corpus = [dictionary.doc2bow(text) for text in texts]
  # print corpus
  # lda = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
  # topics = list(lda.get_document_topics(corpus))
  # print doc_transformed
  # m = 4000
  # topics = np.zeros((m, num_topics))
  # i = 0
  # for doc in doc_transformed:
  #   for word in doc:
  #     data_mat[i][word[0]] = word[1]
  #   i += 1
  # print lda.print_topics(num_topics)
  # print len(dictionary)

  words = set([])
  for text in texts:
    for token in text:
      words.add(token)
  # print texts
  print len(list(words))
  words = list(words)
  dictionary = defaultdict(int)
  for i in xrange(len(words)):
    dictionary[words[i]] = i+1

  
  # doc = np.zeros((len(texts), len(words), size))
  # print doc.shape
  # for i in xrange(len(texts)):
  #   text = texts[i]
  #   for j in xrange(len(text)):
  #       if dictionary[text[j]] > 0:
  #         for k in xrange(size):
  #           doc[i, j, k] = Embedding[text[j]][k]
  

  doc = []
  for text in texts:
    vec = []
    for word in text:
      vec.append(dictionary[word])
    doc.append(vec)
  maxn = 0
  for i in xrange(len(doc)):
    if maxn < len(doc[i]):
      maxn = len(doc[i])
  print maxn,
  global maxlen
  maxlen = maxn
  print maxlen
  # doc = np.array(doc)
  # # print list(doc[0]).count(1)
  # # print doc.shape
  doc = sequence.pad_sequences(doc, maxlen=maxlen)
  print doc.shape
  # print doc
 
  # np.save('isear_document.npy', doc)
  # doc = np.load('isear_document.npy')
  return doc, words

def LDA(topics, X_train, X_test, random_list1, random_list2):
  # print topics
  m = 4000
  print m
  doc_topics = np.zeros((m, num_topics))
  i = 0
  for doc in topics:
    for word in doc:
      doc_topics[i][word[0]] = word[1]
    i += 1
  print doc_topics.shape
  data_train = []
  data_test = []
  for i in xrange(2000):
  	vec = []
  	for j in X_train[i]:
  		vec.append(j)
  	for j in doc_topics[i]:
  		vec.append(j)
  	data_train.append(vec)
  for i in random_list1:
  	vec = []
  	for j in X_test[i]:
  		vec.append(j)
  	for j in doc_topics[2000+i]:
  		vec.append(j)
  	data_train.append(vec)
  for i in random_list2:
    vec = []
    for j in X_test[i]:
      vec.append(j)
    for j in doc_topics[2000+i]:
      vec.append(j)
    data_train.append(vec)

  for i in xrange(2000):
  	vec = []
  	for j in X_test[i]:
  		vec.append(j)
  	for j in doc_topics[2000+i]:
  		vec.append(j)
  	data_test.append(vec)

  data_train = np.array(data_train)
  data_test = np.array(data_test)
  
  print data_train.shape, data_test.shape
  return data_train, data_test


def conv_block(x, nb_filter, kernel_size=3):
	k = nb_filter
	out = Convolution1D(k, kernel_size, subsample_length=2, border_mode='same')(x)
	# out = BatchNormalization(axis=1)(out)
	
	out = Activation('relu')(out)
	out = Dropout(0.5)(out)
	out = Convolution1D(k, kernel_size, subsample_length=1, border_mode='same')(out)
	# out = BatchNormalization(axis=1)(out)
	
	out = Activation('relu')(out)
	out = Dropout(0.5)(out)

	x = Convolution1D(k, kernel_size, subsample_length=2, border_mode='same')(x)

	out = merge([out, x], mode='sum')
	out = Activation('relu')(out)

	return out

def identity_block(x, nb_filter, kernel_size=3):
	k = nb_filter  

	out = Convolution1D(k, kernel_size, border_mode='same', subsample_length=1)(x)
	# out = BatchNormalization(axis=1)(out)
	
	out = Activation('relu')(out)
	out = Dropout(0.5)(out)
	out = Convolution1D(k, kernel_size, border_mode='same', subsample_length=1)(out)
	# out = BatchNormalization(axis=1)(out)
	
	out = Activation('relu')(out)
	out = Dropout(0.5)(out)

	out = merge([out, x], mode='sum')
	out = Activation('relu')(out)
	return out

def CNN_Model(X_train, Y_train, X_test, Y_test, words):
  inp = Input(shape=(maxlen,))
  # out = Masking(mask_value=0)(inp)
  out = Embedding(input_dim=len(words)+1, output_dim=64, input_length=maxlen, dropout=0.2)(inp)
  # out = Bidirectional(LSTM(128))(out)
  # out = LSTM(64)(out)
  out = Convolution1D(256, 64, border_mode='same', subsample_length=1)(out)
  # out = BatchNormalization(axis=1)(out)
  out = Activation('relu')(out)
  # out = PReLU()(out)
  # out = MaxPooling1D(64)(out)
  # # out = LSTM(128)(out)
  # out = Convolution1D(64, 48, border_mode='same', subsample_length=1)(out)
  # # out = BatchNormalization(axis=1)(out)
  # out = Activation('relu')(out)
  # out = MaxPooling1D(3)(out)
  # out = Dropout(0.5)(out)
  # out = identity_block(out,256)
  # out = identity_block(out,64)
  # out = identity_block(out,64)
  
  # out = conv_block(out,512)
  # out = identity_block(out,512)
  # out = identity_block(out,128)
  # out = identity_block(out,128)
  
  # out = conv_block(out,256)
  # out = identity_block(out,256)
  # out = identity_block(out,256)
  # out = identity_block(out,256)
  # out = BatchNormalization()(out) # the val_loss is good, amazing
  # out = Activation('relu')(out)
  # out = Convolution1D(250, 3, border_mode='valid', subsample_length=1)(out)
  # out = Activation('relu')(out)
  # out = MaxPooling1D(2)(out)
  # out = Convolution1D(250, 3, border_mode='valid', subsample_length=1)(out)
  # out = Activation('relu')(out)
  out = GlobalMaxPooling1D()(out)
  out = Dropout(0.5)(out)
  # out = Dropout(0.2)(out)
  # out = Activation('relu')(out)
  # out = Dense(512, activation='relu')(out)
  # out = Dropout(0.5)(out)
  out = Dense(1, activation='sigmoid')(out)
  model = Model(inp,out)
  # get_avg_layer = Model(input=inp, output=avg)
  sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)

  best_weights_filepath = './best_multi_domain_weights.hdf5'
  saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, shuffle=True, validation_data=(X_test, Y_test), verbose=2, callbacks=[saveBestModel])

  json_string = model.to_json()
  open('multi_domain_archi.json', 'w').write(json_string)
  model.save_weights('final_multi_domain_weights.h5')

  return model

clf = SVC(kernel='linear')
def buildStump(dataArr, classLabels, D, epoch):  
	 
	m = len(dataArr)
	numSteps = 10.0; bestStump = {}; bestClassEst = np.mat(np.zeros((m,1)))  
	minError = 1000000 
	
	for i in xrange(1):
		random_list1 = random.sample(range(DIV), 500 )
		random_list2 = random.sample(range(DIV, TRAIN), 500)
		random_list3 = random.sample(range(TRAIN, TRAIN+num_Ts), 40)
		train = []
		label = []
		for j in random_list1:
			train.append(dataArr[j])
			label.append(classLabels[j])
		for j in random_list2:
			train.append(dataArr[j])
			label.append(classLabels[j])
		for j in random_list3:
			train.append(dataArr[j])
			label.append(classLabels[j])

		clf.fit(train, label)
		# clf.fit(dataArr, classLabels)
		predictedVals = clf.predict(dataArr)

		errArr = np.mat(np.ones((m,1)))  
		errArr[predictedVals == classLabels] = 0 
		s = sum(D[diff:m])
		wi = D[diff:m] / s
		# print errArr[diff:m]
		error_t = sum(wi * errArr[diff:m])
		# print error_t
		if error_t > 0.5:
			error_t = 0.5
		# print error_t
		if error_t < minError:  
			# print threshVal
			minError = error_t 
			bestStump['clf'] = clf
			bestClassEst = predictedVals.copy()  
	return bestStump, minError, bestClassEst

# 基于svm的adaboost训练过程  
def adaBoostTrainDS(dataArr, classLabels, data_test, label_test, numIt, proba):
	# print diff
	weakClassArr = []  
	m = dataArr.shape[0]
	D = np.ones(m)

	# D[diff:m] = 10
	# D = D / sum(D)
	print m
	# hf = mat(ones((m, 1)))
	for i in range(numIt):  
		print 'numIt: %s' % str(i+1),
		bestStump, error, classEst = buildStump(dataArr, classLabels, D, i)  
		
		beta_t = float(error / (1.0 - error) + 1e-16)
		beta = float(1.0 / (1.0 + np.sqrt(2 * np.log(diff) / 100)))

		bestStump['beta_t'] = beta_t

		weakClassArr.append(bestStump)  

		for x in xrange(0, diff):
			D[x] *= pow(beta, np.fabs((classEst[x] - classLabels[x]) / 2))
		for x in xrange(diff, m):
			D[x] *= pow(beta_t, -np.fabs((classEst[x] - classLabels[x]) / 2))

		# D = D / D.sum()
		# print D

		adaClassify(data_test, label_test, bestStump, proba)
		
	return weakClassArr

aggClassEst = np.zeros(2000)
maxn = 0
def adaClassify(datToClass, label_test, classifierArr, proba):  
	
	m = len(datToClass)
	
	classEst = classifierArr['clf'].predict(datToClass)
		
	for x in xrange(m):
		aggClassEst[x] += (classifierArr['beta_t']) * classEst[x]
	
	hf = np.sign(aggClassEst)
	
	cnt = 0
	for x in xrange(m):
		if hf[x] == 0:
			hf[x] = -1
		if hf[x] == label_test[x]:
			cnt += 1

	acc = cnt * 1.0 / m
	print 'Tradaboost: ', acc
	global global_maxn
	# global_maxn = 0
	pred = sigmoid(aggClassEst)
	alpha = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
	for j in alpha:
		print j, 
		p = []
		# local_maxn = 0
		for i in xrange(len(label_test)):
			p.append(j*proba[i] + (1-j)*pred[i])
		for i in xrange(len(p)):
			if p[i] >= 0.5:
				p[i] = 1
			else:
				p[i] = -1
		cnt = 0
		for i in xrange(len(p)):
			if p[i] == label_test[i]:
				cnt += 1
		accuracy = cnt * 1.0 / len(p)
		print 'Accuracy: ', accuracy
		if global_maxn < accuracy:
			global_maxn = accuracy
	print global_maxn
	print 
	# global maxn
	# if maxn < acc:
	# 	maxn = acc

def eval(ans, pred):
  TP, FN, FP, TN = 0, 0, 0, 0
  for i in xrange(len(ans)):
    if ans[i] == 1 and pred[i] == 1:
      TP += 1
    elif ans[i] == 1 and pred[i] == 0:
      FN += 1
    elif ans[i] == 0 and pred[i] == 1:
      FP += 1
    elif ans[i] == 0 and pred[i] == 0:
      TN += 1
  return TP, FN, FP, TN


def precision(TP, FN, FP, TN):
  return TP * 1.0 / (TP + FP)

def recall(TP, FN, FP, TN):
  return TP * 1.0 / (TP + FN)

def F1_measure(TP, FN, FP, TN):
  return 2 * TP * 1.0 / (2 * TP + FP + FN)

def TransNet(train_doc, train_labels, test_doc, test_labels, random_list1, random_list2):
  doc, words = Sequence(train_doc[0:2000] + test_doc[0:2000])
  # doc, words = Sequence(train_doc0[0:2000] + train_doc1[0:2000] + train_doc2[0:2000] + test_doc[0:2000])
  X_train = doc[0:2000]
  X_test = doc[2000:2000+2000]
  print doc
  
  Ts = []
  l = []
  for i in random_list1:
    Ts.append(X_test[i])
    l.append(test_labels[i])
  for i in random_list2:
    Ts.append(X_test[i])
    l.append(test_labels[i])
  Ts = np.array(Ts)
  l = np.array(l)
  # print Ts.shape
  X_train = np.concatenate((X_train, Ts))
  Y_train = np.concatenate((train_labels[0:2000], l))
  # Y_train = np.concatenate(((train_labels0[0:2000] + train_labels1[0:2000] + train_labels2[0:2000]), l))
  Y_test = np.array(test_labels[0:2000])
  # # print X_train.shape
  # # print X_test.shape
  model = CNN_Model(X_train, Y_train, X_test, Y_test, words)

  model = model_from_json(open('multi_domain_archi.json').read())
  model.load_weights('best_multi_domain_weights.hdf5')
  
  best_weights_filepath = './best_multi_domain_weights.hdf5'
  saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

  for i in xrange(len(model.layers)):
  	print model.layers[i].name, i, model.layers[i].trainable
  for i in xrange(4):
  	model.layers[i].trainable = False
  sgd = SGD(lr=1e-4, decay=1e-4, momentum=0.9, nesterov=True)
  model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
  model.fit(X_train[2000:2000+50], Y_train[2000:2000+50], batch_size=1, nb_epoch=10, shuffle=True, validation_data=(X_test, Y_test), verbose=2, callbacks=[saveBestModel])

  model = model_from_json(open('multi_domain_archi.json').read())
  model.load_weights('best_multi_domain_weights.hdf5')

  proba = model.predict(X_test)
  
  return model, proba

def Format(documents):
	
	stoplist = set('for a of the and to i in not'.split())	# 停词表
	texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
	frequency = defaultdict(int)
	for text in texts:
		for token in text:
			frequency[token] += 1
	texts = [[token for token in text if frequency[token] > 30 and len(token) > 2]	# 根据词频停词
				for text in texts]
	dictionary = corpora.Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts]

	# tfidf = models.TfidfModel(corpus)
	# corpus_tfidf = tfidf[corpus]
	lda = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
	# lda.save('model.lda')
	# lda = models.LdaModel.load('model.lda')
	doc_transformed = list(lda.get_document_topics(corpus))

	n = len(dictionary)

	# print n
	m = len(texts)
	# print m

	data_mat_topic = np.zeros((m, n+num_topics))
	print data_mat_topic.shape

	i = 0
	for doc in corpus:
		for word in doc:
			data_mat_topic[i][word[0]] = word[1]
		i += 1
	
	i = 0
	for doc in doc_transformed:
		for word in doc:
			data_mat_topic[i][word[0]+n] = word[1]
		i += 1
	# print data_mat_topic
	return data_mat_topic

def TrAdaBoost(X_train, Y_train, X_test, Y_test, random_list1, random_list2, proba):
  for i in xrange(len(Y_train)):
    if Y_train[i] == 0:
      Y_train[i] = -1
  for i in xrange(len(Y_test)):
    if Y_test[i] == 0:
      Y_test[i] = -1
  doc = Format(X_train[0:2000] + X_test[0:2000])
  X_train = doc[0:2000]
  X_test = doc[2000:2000+2000]

  Ts = []
  l = []
  for i in random_list1:
    Ts.append(X_test[i])
    l.append(Y_test[i])
  for i in random_list2:
    Ts.append(X_test[i])
    l.append(Y_test[i])
  Ts = np.array(Ts)
  l = np.array(l)
  # print Ts.shape
  X_train = np.concatenate((X_train, Ts))
  Y_train = np.concatenate((Y_train[0:2000], l))
  Y_test = np.array(Y_train[0:2000])
  
  numIt = 100
  print 'numIt: %s ' % numIt
  print 'num_topics: %s' % str(num_topics)
  classifierArr = adaBoostTrainDS(X_train, Y_train, X_test, Y_test, numIt, proba)	# 30

  pred = sigmoid(aggClassEst)

  # cnt = 0
  # for x in xrange(m):
  #   if hf[x] == 0:
  #     hf[x] = -1
  #   if hf[x] == label_test[x]:
  #     cnt += 1

  return classifierArr, pred, Y_test

def sigmoid(X):  
	return 1.0 / (1 + np.exp(-X)) 

# def Merged_Model(X_test, Y_test, classifierArr, model):


def main():
  # X_train, Y_train = read_data('train')
  # X_test, Y_test = read_data('test')
  # train_num = len(Y_train)
  # test_num = len(Y_test)
  # doc, words = Sequence(X_train + X_test)
  # print train_num, test_num
  # X_train = doc[0:train_num]
  # X_test = doc[train_num:train_num+test_num]
  # Y_train_new = np_utils.to_categorical(Y_train, nb_classes)
  # Y_test_new = np_utils.to_categorical(Y_test, nb_classes)
  # model = CNN_Model(X_train, Y_train_new, X_test, Y_test_new, words)
  
  domain_id = [0, 1, 2, 3]
  train_id = 1
  test_id = 0
  train_doc0, train_labels0 = Doc(folder[train_id], unlabel[train_id])
  # train_doc1, train_labels1 = Doc(folder[domain_id[1]], unlabel[domain_id[1]])
  # train_doc2, train_labels2 = Doc(folder[domain_id[3]], unlabel[domain_id[3]])

  test_doc, test_labels = Doc(folder[test_id], unlabel[test_id])
  
  random_list1 = random.sample(range(0, 1000), 25)
  random_list2 = random.sample(range(1000, 2000), 25)

  model, proba = TransNet(train_doc0, train_labels0, test_doc, test_labels, random_list1, random_list2)
  classifierArr, pred, Y_test = TrAdaBoost(train_doc0, train_labels0, test_doc, test_labels, random_list1, random_list2, proba)

  

  # alpha = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  # for j in alpha:
  # 	  print j, 
  # 	  p = []
	 #  for i in xrange(len(Y_test)):
	 #  	p.append(j*proba[i] + (1-j)*pred[i])
	 #  # print p
	 #  # print len(p)
	 #  for i in xrange(len(p)):
	 #  	if p[i] >= 0.5:
	 #  		p[i] = 1
	 #  	else:
	 #  		p[i] = -1

	 #  cnt = 0
	 #  for i in xrange(len(p)):
	 #    if p[i] == Y_test[i]:
	 #      cnt += 1
	 #  print 'Accuracy: ', cnt * 1.0 / len(p)
  # print Y_test
  # Merged_Model(X_test, Y_test, classifierArr, model)
  # X_train = (X_train - np.mean(X_train))/np.std(X_train)
  # X_test = (X_test - np.mean(X_test))/np.std(X_test)

  # X_train, X_test = LDA(topics, X_train, X_test, random_list1, random_list2)

  # for i in xrange(len(Y_train)):
  # 	if Y_train[i] == 0:
  # 		Y_train[i] == -1
  # for i in xrange(len(Y_test)):
  # 	if Y_test[i] == 0:
  # 		Y_test[i] == -1

  # print 'TrAdaBoost'
  # numIt = 50
  # print 'numIt: %s ' % numIt
  # print 'num_topics: %s' % str(num_topics)
  # classifierArr = adaBoostTrainDS(X_train, Y_train, X_test, Y_test, numIt)
  # bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=100)
  # bdt.fit(X_train, Y_train)
  # pred = bdt.predict(X_test)
  # cnt = 0
  # for i in xrange(len(Y_test)):
  # 	if pred[i] == Y_test[i]:
  # 		cnt += 1
  # print 'Accuracy: ', cnt * 1.0 / len(Y_test)
  


if __name__ == '__main__':
	main()
 
