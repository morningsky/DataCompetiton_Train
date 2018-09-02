# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 15:01:32 2017

@author: sky
"""

from sklearn.datasets import fetch_20newsgroups #需要联网下载
news = fetch_20newsgroups(subset='all')

from sklearn.cross_validation import train_test_split #以3：1划分训练集测试集
X_train,X_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25)

#将文本转化为特征向量
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.fit_transform(X_test)

#使用朴素贝叶斯算法训练
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)

#查看效果
from sklearn.metrics import classification_report
print 'score:',model.score(X_test,y_test)
print classification_report(y_test,y_predict,target_names=news.target_names)

