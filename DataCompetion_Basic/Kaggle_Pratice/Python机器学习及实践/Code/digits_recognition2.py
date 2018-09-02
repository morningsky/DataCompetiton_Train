# -*- coding: utf-8 -*-
"""
Created on Tue Jan 02 20:37:27 2018

@author: sky
"""

import pandas as pd
import numpy as np

data_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header = None)
data_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header = None)

X_train = data_train[np.arange(64)] #图片为64*64的大小 前64列为像素信息
y_train = data_train[64] #第65列为数字label

X_test = data_test[np.arange(64)]
y_test = data_test[64]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)

from sklearn import metrics 
print metrics.adjusted_rand_score(y_test,y_pred) #利用聚类结果的准确率进行评估

                                 