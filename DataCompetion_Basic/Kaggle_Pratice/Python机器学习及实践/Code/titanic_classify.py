# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:19:16 2017

@author: sky
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report


titanic = pd.read_csv('../Datasets/Titanic/train.csv')
X = titanic[['Pclass','Age','Sex']]
y = titanic['Survived']

#使用平均值填充Age
X['Age'].fillna(X['Age'].mean(),inplace=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)
#将Sex转化为数值特征 用0/1代替
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.fit_transform(X_test.to_dict(orient='record'))

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
print 'DCT score:',model.score(X_test,y_test)
print classification_report(y_test,y_predict)

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
print 'GBC score:',model.score(X_test,y_test)
print classification_report(y_test,y_predict)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
print 'RFC score:',model.score(X_test,y_test)
print classification_report(y_test,y_predict)





