# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 15:29:20 2017

@author: sky
"""
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier #knn分类模型
from sklearn.metrics import classification_report


iris = load_iris() #获取iris数据
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25)#划分数据

ss = StandardScaler()  #标准化
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

model = KNeighborsClassifier()#模型训练、预测
model.fit(X_train,y_train)
y_predict = model.predict(X_test)

print 'score:',model.score(X_test,y_test) #正确率
print classification_report(y_test,y_predict,target_names=iris.target_names)#平均精确率 召回率 F1值
