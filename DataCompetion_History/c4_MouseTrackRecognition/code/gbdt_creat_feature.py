# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 19:14:42 2017

@author: sky
"""

from sklearn.model_selection import train_test_split
from pandas import DataFrame
from sklearn import metrics
from sklearn.datasets  import  make_hastie_10_2
from xgboost.sklearn import XGBClassifier
import xgboost as xgb

X, y = make_hastie_10_2(random_state=0)
X = DataFrame(X)
y = DataFrame(y)
y.columns={"label"}
label={-1:0,1:1}
y.label=y.label.map(label)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)#划分数据集
y_train.head()

clf = XGBClassifier(
    n_estimators=30,#三十棵树
    learning_rate =0.3,
    max_depth=3,
    min_child_weight=1,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=12,
    scale_pos_weight=1,
    reg_lambda=1,
    seed=27)

model_sklearn=clf.fit(X_train, y_train)
y_sklearn= clf.predict_proba(X_test)[:,1]
print("XGBoost_sklearn接口 AUC Score : %f" % metrics.roc_auc_score(y_test, y_sklearn))
train_new_feature= clf.apply(X_train)
test_new_feature= clf.apply(X_test)
train_new_feature2 = DataFrame(train_new_feature)
test_new_feature2 = DataFrame(test_new_feature)
new_feature2=clf.fit(train_new_feature2, y_train)
y_new_feature2= clf.predict_proba(test_new_feature2)[:,1]
#用两组新的特征分别训练，预测
print("XGBoost自带接口生成的新特征预测结果 AUC Score : %f" % metrics.roc_auc_score(y_test, y_new_feature2))