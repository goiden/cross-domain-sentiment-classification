#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from numpy import *
from gensim import corpora, models, similarities
from sklearn.svm import SVC
from collections import defaultdict
import random

f = open('out.txt', 'w')
# Td
# 主题数量
num_topics = 4

stoplist = []
with open('stopwords.txt') as fr:
    for word in fr.readlines():
        stoplist.append(word.strip())
stoplist = set(stoplist)

folder = ['books', 'dvd', 'electronics', 'kitchen']
train_num = 1000
TRAIN = 2000
DIV = 1000
diff = 2000
unlabel = [4465, 3586, 5681, 5945]
all_words = set([])

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
        	if word not in stoplist:
    			all_words.add(word)
        	for k in xrange(num):
        		words.append(word)
        x = line[len(line)-1].split(':')
        polarity = x[1]
        if polarity == 'positive':
        	label.append(1)
        elif polarity == 'negative':
        	label.append(-1)
        corpus.append(' '.join(words))
    return corpus, label


def Format(documents):
	# stoplist = set('for a of the and to in i not\' his he <num>\' not was their who <num> one more you\' not you  all if my her\' about one what how they we which some so very no only other just me out like when time first even\' her she your many'.split())
	stoplis = set('for a of the and to i in not'.split())	# 停词表

	# print len(all_words)
	# print documents[2000]

	texts = [[word for word in document.lower().split() if word not in stoplis] for document in documents]
	# print len(texts)

	frequency = defaultdict(int)
	for text in texts:
		for token in text:
			frequency[token] += 1
	texts = [[token for token in text if frequency[token] > 300 and len(token) > 2]	# 根据词频停词
				for text in texts]

	dictionary = corpora.Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts]
	# print corpus

	n = len(dictionary)
	print n
	m = len(texts)
	print m


	lda = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
	doc_transformed = list(lda.get_document_topics(corpus))
	# # print doc_transformed
	data_mat = zeros((m, num_topics))
	i = 0
	for doc in doc_transformed:
		for word in doc:
			data_mat[i][word[0]] = word[1]
		i += 1
	print lda.print_topics(num_topics)
	# # print len(dictionary)
	# print lda.get_topic_terms(0, 20)

	# 
	# data_mat = zeros((m, n))
	# i = 0
	# for corp in corpus:
	# 	if corp != []:
	# 		for word in corp:
	# 			data_mat[i][word[0]] = word[1]
	# 	i += 1

	# print data_mat
	return data_mat

	# return lda, dictionary, corpus

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

def buildStump(dataArr, classLabels, D):  
	dataMatrix = mat(dataArr); labelMat = mat(classLabels).T  
	# dataArr = array(dataArr); classLabels = array(classLabels)
	m,n = shape(dataMatrix)  
	numSteps = 10.0; bestStump = {}; bestClassEst = mat(zeros((m,1)))  
	minError = inf 
	# svm by scikitlearn
	clf = SVC()
	# print dataArr

	'''
	# k次迭代找出最低误差率的一次svm basic learner，可调参数
	'''
	for i in xrange(1):
		'''
		# 每次随机选2*n个训练svm basic learner，不同标签各n个，可调参数
		'''
		# random_list1 = random.sample(range(DIV), 10)
		# random_list2 = random.sample(range(DIV, TRAIN), 10)
		random_list1 = random.sample(range(TRAIN), 200)
		random_list2 = random.sample(range(TRAIN), 200)
		random_list3 = range(TRAIN, TRAIN+10)
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

		# print train, label

		'''
		# svm by scikitlearn
		'''
		clf.fit(train, label)
		# clf.fit(dataArr, classLabels)
		predictedVals = clf.predict(dataArr)


		# print predictedVals
		errArr = mat(ones((m,1)))  
		errArr[predictedVals == classLabels] = 0 
		s = sum(D[diff:m])
		wi = D[diff:m] / s
		error_t = wi.T * errArr[diff:m]
		if error_t > 0.5:
			error_t = 0.5

		if error_t < minError:  
			# print threshVal
			minError = error_t 
			bestStump['clf'] = clf
			bestClassEst = predictedVals.copy()  
	
	del clf
	return bestStump, minError, bestClassEst


# 基于svm的adaboost训练过程  
def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
	# print diff
	weakClassArr = []  
	m = shape(dataArr)[0]  
	D = ones(m)
	print m
	# hf = mat(ones((m, 1)))
	for i in range(numIt):  
		print 'numIt: %s' % str(i+1)
		bestStump, error, classEst = buildStump(dataArr, classLabels, D)  
		# bestStump = {}
		
		

		# beta and beta_t, update the new weight vector
		# error = error_t
		beta_t = float(error / (1.0 - error) + 1e-16)
		beta = float(1.0 / (1.0 + sqrt(2 * log(diff) / numIt)))

		bestStump['beta_t'] = beta_t

		# print bestStump

		weakClassArr.append(bestStump)  

		# D[0:diff] *= (classEst[0:diff] - classLabels[0:diff]) / 2
		# D[diff:m] *= (classEst[diff:m] - classLabels[diff:m]) / 2
		for x in xrange(0, diff):
			D[x] *= pow(beta, fabs((classEst[x] - classLabels[x]) / 2))
		for x in xrange(diff, m):
			D[x] *= pow(beta_t, -fabs((classEst[x] - classLabels[x]) / 2))

		print error
		# print D
		# if error == 0.0:
		# break

	return weakClassArr


def predict(data_test, label_test, classifierArr):  
	dataMatrix = mat(data_test)
	m = shape(dataMatrix)[0]  
	hf = mat(ones((m, 1)))
	numIt = len(classifierArr)
	for x in xrange(m):
		l, r = 1, 1
		for t in xrange((numIt/2), numIt):	# floor(numIt/2)	
	# 		classEst = stumpClassify(dataMatrix, classifierArr[t]['dim'], classifierArr[t]['thresh'], classifierArr[t]['ineq'])
			classEst = classifierArr[t]['clf'].predict(dataMatrix[x])
			# print classEst
			if classEst == 1:
				l *= pow(classifierArr[t]['beta_t'], -1.0)
			r *= pow(classifierArr[t]['beta_t'], -0.5)
		if l < r:
			hf[x] = -1
	# print hf
	# print label_test
	cnt = 0
	for x in xrange(m):
		if int(hf[x]) == label_test[x]:
			cnt += 1
	print 'tsvm: ', cnt * 1.0 / m

def main():
	print 'lda...'
	# 重复试验次数
	for k in xrange(1):
		
		'''
		0: books
		1: dvd
		2: electronics
		3: kitchen

		'''
		train_id = 0
		test_id = 2
		train_doc, train_labels = Doc(folder[train_id], unlabel[train_id])
		test_doc, test_labels = Doc(folder[test_id], unlabel[test_id])
		# electronics_doc, electronics_labels = doc(folder[2], unlabel[2])
		# kitchen_doc, kitchen_labels = doc(folder[3], unlabel[3])
		# print len(books_doc[0:2000])
		# print len(dvd_doc[2000:unlabel[1]])
		
		# test_list1 = random.sample(range(1000), 200)
		# test_list2 = random.sample(range(1000, 2000), 200)
		# test_list = test_list1 + test_list2
		# doc = []
		# label = []
		# for i in xrange(2000):
		# 	if i not in test_list:
		# 		doc.append(train_doc[i])
		# 		label.append(train_labels[i])
		# for i in test_list:
		# 	doc.append(train_doc[i])
		# 	label.append(train_labels[i])
		
		data_mat = Format(train_doc[0:2000] + test_doc[2000:2000+unlabel[test_id]])
		# data_mat = Format(test_doc[0:2000])
		# print doc
		# data_mat = Format(doc)
		# data_train = data_mat[0:1600]
		# label_train = label[0:1600]
		# data_test = data_mat[1600:2000]
		# label_test = label[1600:2000]

		data_train = data_mat[0:2000]
		label_train = train_labels[0:2000]
		
		data_test = data_mat[2000:2000+unlabel[test_id]]
		label_test = test_labels[2000:2000+unlabel[test_id]]

		'''
		选取带标签的测试数据用来训练
		'''
		random_list = random.sample(range(unlabel[test_id]), 50)
		# random_list = [3629, 4392, 1445, 705, 3202, 3520, 468, 5023, 2380, 5660, 944, 873, 4623, 5028,4323, 2281, 4451, 2404, 1118, 1657, 4279, 4206, 2014, 998, 544, 2173, 3682, 1645, 1188, 1273, 4421, 3234, 4797, 3284, 508, 1294, 5445, 4735, 5535, 2091, 848, 3968, 2202, 3610, 4622, 5037, 662, 989, 1647, 409]
		# random_list = [1977, 5626, 1544, 5440, 4838, 5126, 2835, 2962, 208, 3609, 2000, 2478, 741, 3611, 3592, 4221, 4422, 1882, 4921, 829, 1985, 1230, 1561, 4101, 3168, 135, 5536, 2613, 843, 502, 5165, 1572, 5059, 4476, 1122, 1859, 1302, 911, 282, 1054, 4390, 871, 24, 1747, 4504, 5100, 2481, 4624, 5566, 271, 2633, 2917, 2057, 366, 380, 2637, 1952, 800, 3389, 4376, 4682, 25, 3998, 3559, 4314, 2322, 4743, 3298, 1092, 2037, 5124, 1643, 4157, 967, 2394, 3945, 4855, 1789, 5011, 5043, 1565, 4020, 4341,1874, 1955, 4570, 5491, 2166, 783, 1154, 868, 1330, 3508, 2813, 440, 827, 5037,3047, 2676, 3209]
		# random_list = [1728, 3117, 1066, 1454, 2997, 3506, 480, 3036, 3058, 2145, 1578, 2939, 1555, 2758, 9, 1815, 1228, 3436, 753, 218, 2629, 567, 3407, 1905, 939, 1180, 1875, 1178,716, 1594, 1425, 3163, 742, 1002, 1574, 1317, 964, 411, 3253, 565, 2221, 2761,1085, 2824, 921, 849, 1674, 1135, 801, 3171]
		print random_list
		Ts = []
		l = []
		for i in random_list:
			Ts.append(data_test[i])
			l.append(label_test[i])
		Ts = array(Ts)
		l = array(l)
		data_train = concatenate((data_train, Ts))
		label_train = concatenate((label_train, l))
		
		# print data_train

		# for i in xrange(len(data_train)):
		# 	for j in data_train[i]:
		# 		print >> f, j,
		# 	print >> f

		'''
		svm by scikitlearn
		'''
		clf = SVC()
		clf.fit(data_train, label_train)
		predictedVals = clf.predict(data_test)

		cnt = 0
		for i in xrange(len(label_test)):
			if label_test[i] == predictedVals[i]:
				cnt += 1
		print 'svm: ', cnt * 1.0 / len(label_test)

		# '''
		# boosting
		# '''
		# numIt = 4
		# print 'numIt: %s' % numIt
		# classifierArr = adaBoostTrainDS(data_train, label_train, numIt)	# 30
		# predict(data_test, label_test, classifierArr)
		# print type(data_train[1])
		# print type(label_train[1])
		# classifierArr = adaBoostTrainDS(data_train, label_train, numIt)	# 30
		# predict(data_test, label_test, classifierArr)
		# print len(data_mat)
		# lda_books, dictionary_books, corpus_books = LDA(books_doc)
		# lda_dvd, dictionary_dvd, corpus_dvd = LDA(dvd_doc)

		# print lda_books.print_topics(num_topics)
		# print lda_dvd.print_topics(num_topics)
		# index = similarities.MatrixSimilarity(lda[corpus])
		# k = 7
		# print books_corpus[0]
		# doc = "dvd dvd dvd"
		# predict = []
		# for doc in dvd_doc[2000:]:
		# # doc = dvd_doc[0]
		# # # print doc

		# 	vec_bow = dictionary.doc2bow(doc.lower().split())
		# 	vec_lda = lda[vec_bow]
		# 	# print vec_lda
		# 	sims = index[vec_lda]
		# 	sims = sorted(enumerate(sims), key=lambda item: -item[1])
		# 	l = []
		# 	for i in xrange(k):
		# 		l.append(books_labels[sims[i][0]])
		# 	if l.count(1) > l.count(0):
		# 		predict.append(1)
		# 	else:
		# 		predict.append(0)

		# # print (predict)
		# cnt = 0
		# for i in xrange(2000, len(dvd_labels)):
		# 	if dvd_labels[i] == predict[i-2000]:
		# 		cnt += 1
		# print cnt * 1.0 / len(dvd_labels[2000:])
			# # print sims
			# cnt = 0
			# for i in sims:
			# 	if cnt == 1000:
			# 		break
			# 	cnt += 1
			# 	print i


		# for i in xrange(num_topics):
			# print lda.print_topics(i)
			# print lda.get_topic_terms(i, topn=10)
		# # 
		# print random_list
		# for i in xrange(10):
		# 	# t = random.randint(0, TEST-1)
		# 	# print t,
		# 	t = random_list[i]
		# 	# t = random_index[i]
		# 	data_train.append(data_test[t])
		# 	label_train.append(label_test[t])
		# # tradaboost迭代次数
		# numIt = 10
		# classifierArr = adaBoostTrainDS(data_train, label_train, w, numIt)	# 30
		# predict(data_test, label_test, classifierArr)

if __name__ == '__main__':
    main()


    

	