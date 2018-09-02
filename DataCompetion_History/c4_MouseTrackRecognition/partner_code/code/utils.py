#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: Gsscsd   
@email: Gsscsd@qq.com 
@site: http://gsscsd.loan  
@file: utils.py 
@time: 2017/5/19 
"""
import numpy as np

def standards(tables):
    """
    标准归一化
    :param tables:
    :return:
    """
    diffNum = 0
    meanNum = 0
    otherTables =[]
    for lines in tables:
        diffNum = np.std(lines)
        meanNum = np.mean(lines)
        lines = [(num - meanNum)/diffNum for num in lines]
        otherTables.append(lines)
    return otherTables

def standardsIMP(tables):
    """
    改进的标准归一化
    :param tables:
    :return:
    """
    midNum = 0
    absStd = 0
    meanNum = 0
    sums = 0
    otherTables = []
    for lines in tables:
        #获取中位数
        if len(lines)%2 == 0:
            midNum = lines[len(lines)/2]
        else:
            midNum = (lines[len(lines)/2]+lines[len(lines)/2+1])/2.0
        meanNum = sum(lines)/len(lines)
        for num in lines:
            sums += abs(num-meanNum)
        absStd = sums/len(lines)
        lines = [(num - midNum)/absStd for num in lines]
        otherTables.append(lines)
    return otherTables


if __name__ == "__main__":
    tables = [[1,2,3],[2,4,6],[3,6,9]]
    s1 = standards(tables)
    # s2 = standardsIMP(tables)
    print "s1:"
    print s1
    # print "s2:"
    # print s2