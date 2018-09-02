# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 14:42:18 2017

@author: sky
"""

from sklearn.datasets import load_digits
digits = load_digits()

#按3：1划分训练、测试集
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.25)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

model = LinearSVC()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)

print 'Score:',model.score(X_test,y_test)
print classification_report(y_test,y_predict,target_names = digits.target_names.astype(str))

