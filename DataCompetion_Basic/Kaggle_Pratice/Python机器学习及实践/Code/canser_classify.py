# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 14:30:56 2017

@author: sky
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression


#读取数据 1标识列 2特征列 1标签列
train = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-train.csv')
test = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-test.csv')

#构建测试集训练特征
test_positive = test.loc[test['Type'] == 1][['Clump Thickness','Cell Size']] #负样本训练特征
test_negative = test.loc[test['Type'] == 0][['Clump Thickness','Cell Size']] #正样本训练特征

def draw1():
    #绘制正负样本分布散点图
    plt.scatter(test_negative['Clump Thickness'],test_negative['Cell Size'],marker='o',s=200,c='red')
    plt.scatter(test_positive['Clump Thickness'],test_positive['Cell Size'],marker='x',s=150,c='black')
    plt.xlabel('Clump Thickness')
    plt.ylabel('Cell Size')
    plt.show()

def draw2():
    #随机生成直线的参数
    intercept = np.random.random([1])
    coef = np.random.random([2])
    lx = np.arange(0,12) #x轴
    ly = (-intercept -lx * coef[0])/coef[1] #y轴 直线方程ax+by+c=0
    #绘制一条随机直线进行区分
    plt.scatter(test_negative['Clump Thickness'],test_negative['Cell Size'],marker='o',s=200,c='red')
    plt.scatter(test_positive['Clump Thickness'],test_positive['Cell Size'],marker='x',s=150,c='black')
    plt.xlabel('Clump Thickness')
    plt.ylabel('Cell Size')                        
    plt.plot(lx,ly,c='yellow')
    plt.show()

def draw3():
    #使用逻辑回归算法训练前10条训练样本
    model = LogisticRegression()
    model.fit(train[['Clump Thickness','Cell Size']][:10],train['Type'][:10])
    print model.score(test[['Clump Thickness','Cell Size']],test['Type'])
    #绘制使用逻辑回归训练前10条样本得到的分类直线
    #直线参数由算法生成
    intercept = model.intercept_
    coef = model.coef_[0,:]
    lx = np.arange(0,12) #x轴
    ly = (-intercept -lx * coef[0])/coef[1] 
    
    plt.scatter(test_negative['Clump Thickness'],test_negative['Cell Size'],marker='o',s=200,c='red')
    plt.scatter(test_positive['Clump Thickness'],test_positive['Cell Size'],marker='x',s=150,c='black')
    plt.xlabel('Clump Thickness')
    plt.ylabel('Cell Size') 
    plt.plot(lx,ly,c='green')
    plt.show()

def draw4():
    #绘制使用逻辑回归训练全部样本得到的分类直线
    model = LogisticRegression()
    model.fit(train[['Clump Thickness','Cell Size']],train['Type'])
    print model.score(test[['Clump Thickness','Cell Size']],test['Type'])
    intercept = model.intercept_
    coef = model.coef_[0,:]
    lx = np.arange(0,12) #x轴
    ly = (-intercept -lx * coef[0])/coef[1]
    
    plt.scatter(test_negative['Clump Thickness'],test_negative['Cell Size'],marker='o',s=200,c='red')
    plt.scatter(test_positive['Clump Thickness'],test_positive['Cell Size'],marker='x',s=150,c='black')
    plt.xlabel('Clump Thickness')
    plt.ylabel('Cell Size') 
    plt.plot(lx,ly,c='blue')   
    plt.show()                             

