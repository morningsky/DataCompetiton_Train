# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 17:03:41 2017

@author: sky
"""
import warnings 
warnings.filterwarnings('ignore') #由于sklearn内部一些库提醒更新新版本的问题 warning提示较多 先关闭warning输出

from sklearn.datasets import load_boston
boston = load_boston() #加载sklearn内部的boston房价数据

import numpy as np
from sklearn.cross_validation import train_test_split

X = boston.data
y = boston.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

#粗略观察预测值差异
print 'Max:', np.max(boston.target)
print 'Min:', np.min(boston.target)
print 'Mean:', np.mean(boston.target)

#对特征值和预测目标均作标准化
from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
X_train = ss_x.fit_transform(X_train)
X_test = ss_x.fit_transform(X_test)

ss_y =StandardScaler() #预测连续变量 由于预测目标值差异较大 可先标准化后还原
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.fit_transform(y_test)

#使用常规计算方法的线性回归模型预测房价
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)

#使用随机梯度下降迭代参数的线性回归模型预测房价
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor()
sgd.fit(X_train,y_train)
sgd_y_predict = sgd.predict(X_test)

#评价R-squared,MSE,MAE
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print 'lr模型自带评价方法：',lr.score(X_test,y_test)
print 'lr_r2: ',r2_score(y_test,lr_y_predict) #r2类似于分类算法中的正确率 值为0-1
print 'lr_MAE: ',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict))#先逆标准化
print 'lr_MSE: ',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict))#先逆标准化

print 'sgd模型自带评价方法：',sgd.score(X_test,y_test)
print 'sgd_r2: ',r2_score(y_test,sgd_y_predict) 
print 'sgd_MAE: ',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgd_y_predict))
print 'sgd_MSE: ',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgd_y_predict))

#使用svm进行回归预测
from sklearn.svm import SVR

#使用线性核函数
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train,y_train)
linear_svr_y_predict = linear_svr.predict(X_test)

#使用多项式核函数
poly_svr = SVR(kernel= 'poly')
poly_svr.fit(X_train,y_train)
poly_svr_y_predict = poly_svr.predict(X_test)
print 'linear_svr_r2: ',linear_svr.score(X_test,y_test)
print 'poly_svr_r2: ',poly_svr.score(X_test,y_test)
print 'linear_svr_MSE:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict))
print 'poly_svr_MSE:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict))

#使用KNN进行回归预测
from sklearn.neighbors import KNeighborsRegressor
uni_knr = KNeighborsRegressor(weights='uniform') #使用周围K个数的平均值作为回归值
uni_knr.fit(X_train,y_train)
uni_knr_y_predict = uni_knr.predict(X_test)

dis_knr = KNeighborsRegressor(weights='distance') #根据距离加权回归
dis_knr.fit(X_train,y_train)
dis_knr_y_predict = dis_knr.predict(X_test)

print 'uni_knr_r2: ',uni_knr.score(X_test,y_test)
print 'dis_knr_r2: ',dis_knr.score(X_test,y_test)
print 'uni_knr_mse: ',mean_squared_error(ss_y.inverse_transform(y_test),uni_knr_y_predict)
print 'dis_knr_mse: ',mean_squared_error(ss_y.inverse_transform(y_test),dis_knr_y_predict)



