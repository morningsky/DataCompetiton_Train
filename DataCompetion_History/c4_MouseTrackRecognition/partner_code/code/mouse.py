#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: Gsscsd   
@email: Gsscsd@qq.com 
@site: http://gsscsd.loan  
@file: mouse.py 
@time: 2017/4/25
"""
from tools import *
from matplotlib import pyplot as plt
import numpy as np





def readFile(fileName,flag):
    """
    :param fileName: 文件名称
    :param flag:标记
    :return: 处理数据并返回数据矩阵
    """
    allMatrix = []
    with open(fileName,'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split()
        allMatrix.append(line)
        # print line
    solveA1(allMatrix)
    solveA2(allMatrix)
    solveA3(allMatrix)
    if flag == 1 :
        solveA4(allMatrix)
    # print allMatrix
    return allMatrix



def getTranData(datas,flag):
    """
    :param datas: 输入数据集
    :return: 返回规则数据
    """
    train_y = []
    a1 = []
    a2 = []
    for data in datas:
        train_y.append(data[-1])
    # print tran_y
    for data in datas:
        a1.append(data[1])
        a2.append(data[2])
    # print a1
    # print np.array(a1)
    # print np.array(a2)
    # print np.concatenate((a1,a2))
    if flag == 1:
        return a1,a2,train_y
    else:
        return a1,a2,None

def plotVisual(x,y,t,i):
    """
    :可视化数据
    """
    #x,y
    plt.plot(x,y,'b.')
    plt.plot(x,y,'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('X and Y  '+ str(i))
    plt.grid(True)
    plt.savefig('XandY'+ str(i))
    plt.show()
    plt.clf()
    #x,t
    plt.plot(x,t,'b.')
    plt.plot(x,t,'r')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('X and T ' + str(i))
    plt.grid(True)
    plt.savefig('XandT' + str(i))
    plt.show()
    plt.clf()


def transTable(tables):
    xlines = []
    ylines = []
    tlines = []
    x = []
    y = []
    t = []
    i = 1
    for line in tables:
        for row in line:
            x.append(row[0])
            y.append(row[1])
            t.append(row[-1])
        # plotVisual(x,y,t,i)
        xlines.append(x)
        ylines.append(y)
        tlines.append(t)
        x = []
        y = []
        t = []
        i += 1
    return xlines,ylines,tlines



if __name__ == "__main__":

    fileName1 = '../data/dsjtzs_txfz_training_sample.txt'
    fileName2 = '../data/dsjtzs_txfz_test_sample.txt'
    train_X = []
    train_y = []
    data = readFile(fileName2,0)
    # print data
    Ids = get_Id(data)
    print Ids
    #以下是数据清洗和特征提取
    # tables,goals,label = getTranData(data,1)
    # xlines,ylines,tlines = transTable(tables)
    # xgoals,ygoals = getGoalFea(goals)
    # print xgoals
    # print ygoals
    # print label
    # xinits,xmeans,xstds,xmaxs,xmins,xlasts = optFeature(xlines)
    # tinits,tmeans,tstds,tmaxs,tmins,tlasts = optFeature(tlines)
    # yinits,ymeans,ystds,ymaxs,ymins,ylasts = optFeature(ylines)
    # a_s,b_s,c_s = getFits(xlines,tlines)
    # # train_X.append(xinits,xmeans,xstds,xmaxs,xmins,xlasts)
    # # train_X.append(tinits,tmeans,tstds,tmaxs,tmins,tlasts)
    # # train_X.append(yinits,ymeans,ystds,ymaxs,ymins,ylasts)
    # # train_X.append(a_s,b_s,c_s)
    # #x的相关特征
    # train_X.append(xinits)
    # train_X.append(xmeans)
    # train_X.append(xstds)
    # train_X.append(xmaxs)
    # train_X.append(xmins)
    # train_X.append(xlasts)
    # #y的相关特征
    # train_X.append(yinits)
    # train_X.append(ymeans)
    # train_X.append(ystds)
    # train_X.append(ymaxs)
    # train_X.append(ymins)
    # train_X.append(ylasts)
    # #t的相关特征
    # train_X.append(tinits)
    # train_X.append(tmeans)
    # train_X.append(tstds)
    # train_X.append(tmaxs)
    # train_X.append(tmins)
    # train_X.append(tlasts)
    # train_X.append(xgoals)
    # train_X.append(ygoals)
    # #拟合的相关特征
    # train_X.append(a_s)
    # train_X.append(b_s)
    # train_X.append(c_s)
    # #终极特征矩阵
    # train_X = np.array(train_X).T
    # train_y = np.array(label).T
    #
    # print train_X
    # print train_y
