#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: Gsscsd   
@email: Gsscsd@qq.com 
@site: http://gsscsd.loan  
@file: tools.py 
@time: 2017/5/2 
"""

import numpy as np
import pandas as pd

#处理属性1
def solveA1(matrixs):
    for matrix in matrixs:
        matrix[0] = int(matrix[0])

#处理属性2
def solveA2(matrixs):
    vectors = []
    for matrix in matrixs:
        nums = matrix[1].strip('\n').split(';')
        nums = [num for num in nums if(len(num) > 2) ]
        # print nums
        for num in nums:
            vector = num.split(',')
            vector = [float(i) for i in vector]
            # print vector
            vectors.append(vector)
        matrix[1] = vectors
        vectors = []

#处理属性3
def solveA3(matrixs):
    for matrix in matrixs:
        nums = matrix[2].strip('\n').split(',')
        nums = [float(i) for i in nums]
        matrix[2] = nums
        pass

#处理属性4
def solveA4(matrixs):
    for matrix in matrixs:
        matrix[-1] = int(matrix[-1])

def get_Id(matrixs):
    """
    获取编号
    :param matrixs:
    :return:
    """
    Ids = []
    for lines in matrixs:
        Ids.append(lines[0])
    return Ids

#特征值：最小值，平均值，标准差，最大值，初始值，最后值
def optFeature(lines):
    """
    :param lines: 数组
    :return: 返回最小值，平均值，标准差，最大值，初始值，最后值
    """
    lines = np.array(lines)
    # print lines
    inits = []
    means = []
    stds = []
    maxs = []
    mins = []
    lasts = []
    for line in lines:
        inits.append(line[0])
        lasts.append(line[-1])
        stds.append(np.std(line))
        means.append(np.mean(line))
        maxs.append(np.max(line))
        mins.append(np.min(line))
    # print inits
    # print mins
    # print stds
    # print means
    return inits,means,stds,maxs,mins,lasts

#多次拟合的特征值 求a,b,c
def manyfit(x,t):
    """
    拟合多次次函数
    :param x: 位移
    :param t: 时间
    :return:返回参数a,b,c
    """
    #二次多项式拟合
    cof = np.polyfit(x,t,6)
    return cof[0],cof[1],cof[2],cof[3],cof[4],cof[5],cof[6],cof[-1]

#取得系数列表a_s,b_s,c_s,
def getFits(xs,ts):
    a_s = []
    b_s = []
    c_s = []
    d_s = []
    e_s = []
    f_s = []
    g_s = []
    h_s = []
    for i in xrange(len(xs)):
        a,b,c,d,e,f,g,h = manyfit(xs[i],ts[i])
        a_s.append(a)
        b_s.append(b)
        c_s.append(c)
        d_s.append(d)
        e_s.append(e)
        f_s.append(f)
        g_s.append(g)
        h_s.append(h)
    # print a_s
    # print b_s
    # print c_s
    return a_s,b_s,c_s,d_s,e_s,f_s,g_s,h_s

#目标特征提取
def getGoalFea(goals):
    xgoals = []
    ygoals = []
    for goal in goals:
        xgoals.append(goal[0])
        ygoals.append(goal[-1])
    return xgoals,ygoals

#正负标记
def getTag(Ids,labels):
    TIds = []
    FIds = []
    i = 0
    for label in labels:
        if label == 1:
            TIds.append(Ids[i])
        else:
            FIds.append(Ids[i])
        i +=1
    return TIds,FIds

#写入文件
def toText(TIds,fileName):
    NpTIds = np.array(TIds)
    datas = pd.DataFrame(NpTIds)
    path = "../data/"
    datas.to_csv(path+fileName,index=None)

if __name__ == "__main__":
    pass