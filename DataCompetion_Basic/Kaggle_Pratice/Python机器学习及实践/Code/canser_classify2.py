# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 19:25:46 2017

@author: sky
"""

import pandas as pd
import numpy as np

# 创建特征列表
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
data = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-wisconsin.data.txt', names = column_names )
data.replace(to_replace='?',value=np.nan,inplace=True) #将？替换为标准缺失值
data.dropna(how='any',inplace=True) #只要有一个维度数据有缺失则丢弃数据

from sklearn.cross_validation import train_test_split #按3:1划分训练集与测试集
X_train,X_test,y_train,y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)#第1列为编号 2-9列为特征 第10列为类别(一共2类：2，4)

from sklearn.preprocessing import StandardScaler #将数据标准化为正态分布
from sklearn.linear_model import SGDClassifier,LogisticRegression #分类模型

#标准化特征
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

#创建模型并训练、预测
lr = LogisticRegression()
sgd = SGDClassifier()

lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test) #lr分类结果

sgd.fit(X_train,y_train)
sgd_y_predict = lr.predict(X_test) #sgd分类结果

from sklearn.metrics import classification_report
print 'Accuracy of LR Classifier:',lr.score(X_test,y_test)
print classification_report(y_test,lr_y_predict)                          

print 'Accuracy of SGD Classifier:',sgd.score(X_test,y_test)
print classification_report(y_test,sgd_y_predict) 




