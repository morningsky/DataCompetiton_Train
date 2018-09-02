# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 12:18:10 2018

@author: sky
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

import seaborn as sns
import os
import lightgbm as lgb
import time

'''
    orderHistory_test中有用户2686，均出现在了action_test中
    orderFuture_test中有用户10076，均出现在了train_order,train_action中
    action_test中有用户10076

    orderHistory_train中有用户10637
    action_train中有用户40307
    orderFuture_train中有用户40307

    未加分差统计特征之前 52条 94分 加了差分统计特征8*4之后 76条
'''

##read_data###
action_train=pd.read_csv('../data/trainingset/action_train.csv')#用户行为数据
#行为类型一共有9个，其中1是唤醒app；2~4是浏览产品，无先后关系；5~9则是有先后关系的，从填写表单到提交订单再到最后支付。
orderFuture_train=pd.read_csv('../data/trainingset/orderFuture_train.csv')#待预测数据
orderHistory_train=pd.read_csv('../data/trainingset/orderHistory_train.csv')#用户历史订单数据
userComment_train=pd.read_csv('../data/trainingset/userComment_train.csv')#用户评论数据
userProfile_train=pd.read_csv('../data/trainingset/userProfile_train.csv')#用户个人信息
action_test=pd.read_csv('../data/test/action_test.csv')
orderFuture_test=pd.read_csv('../data/test/orderFuture_test.csv')
orderHistory_test=pd.read_csv('../data/test/orderHistory_test.csv')
userComment_test=pd.read_csv('../data/test/userComment_test.csv')
userProfile_test=pd.read_csv('../data/test/userProfile_test.csv')

def time_conv(x):
    timeArray=time.localtime(x)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime

orderHistory_train.orderTime=pd.to_datetime(orderHistory_train.orderTime.map(lambda x: time_conv(x)),format="%Y-%m-%d %H:%M:%S")
orderHistory_test.orderTime=pd.to_datetime(orderHistory_test.orderTime.map(lambda x: time_conv(x)),format="%Y-%m-%d %H:%M:%S")
action_train.actionTime=pd.to_datetime(action_train.actionTime.map(lambda x: time_conv(x)),format="%Y-%m-%d %H:%M:%S")
action_test.actionTime=pd.to_datetime(action_test.actionTime.map(lambda x: time_conv(x)),format="%Y-%m-%d %H:%M:%S")
orderFuture_train.rename(columns={'orderType':'label'},inplace=True)

print u'时间转换完毕！'
####feature#####
##订单表中提取特征
def orderHistory_feat(df):
    grouped=df[['userid','orderType']].groupby('userid',as_index=False)
    df_count=grouped.count()
    df_count.rename(columns={'orderType':'df_count'},inplace=True) #用户历史订单总数
    df_sum=grouped.sum()
    df_sum.rename(columns={'orderType':'df_sum'},inplace=True) #没有下精品单为0 下了为1 求和就是下精品单数
    df_merge=pd.merge(df_count,df_sum,on='userid',how='left')
    df_merge['rate']=df_merge['df_sum']/df_merge['df_count'] #下单率
    #del df_merge['df_count']
    df_merge.rename(columns={'df_count':'orderHistory_feat_count','df_sum':'orderHistory_feat_sum','rate':'orderHistory_feat_rate'},inplace=True)
    # 上述分布表示表示总订单数目 精品单数目 下精品单率
    df_merge['isOrder'] = df['orderType']
    return df_merge

'''
#将时间差转为秒
def func_time_delta(t):
    action_time_delta = []
    for i in t.actionTime:
        action_time_delta.append(i.seconds)
    action_time_delta = pd.DataFrame(action_time_delta,columns=['action_time_delta'])
    return action_time_delta
'''
#将时间差格式转为秒
def func_timedelta_to_second(t):
    if pd.isnull(t): #用t=='NaT'不行 用t==None不行 用t==pd.NaT不行 服了
        return 0
    else:
        return t.seconds

#行为表中提取特征
def actions_orderType(df):
    flag = 0
    df['count']=1 #辅助
    df_count=df[['userid','count']].groupby('userid',as_index=False).count() #统计用户行为总数 实际也是每个用户的action特征集长度
    action_time_fea = []
    time_list = []
    for userid,group in df[['userid','actionType','actionTime']].groupby('userid',as_index=False):
        start_time = time.time()
        if (flag%1000 == 0):
            print 'Now is : ',flag
        group.reset_index(drop=True,inplace=True) #在求距离的时候要用到索引 所以重置index
        time_diff = group['actionTime'].diff().map(func_timedelta_to_second)
        group['time_diff'] = time_diff
        action_time_diff_mean = time_diff.mean()
        action_time_diff_var = time_diff.var() #方差
        action_time_diff_min = time_diff.min()
        if time_diff.size <= 4:
            action_time_diff_last1 = action_time_diff_last2 = action_time_diff_last3 = action_time_diff_last4 = time_diff.iloc[-1]
        else:
            action_time_diff_last1 = time_diff.iloc[-1] #时间间隔末尾值
            action_time_diff_last2 = time_diff.iloc[-2] #时间间隔倒数第二个值
            action_time_diff_last3 = time_diff.iloc[-3] #时间间隔倒数第三个值
            action_time_diff_last4 = time_diff.iloc[-4] #时间间隔倒数第四个
        if time_diff.size == 1:
            action_time_diff_first1 = time_diff.iloc[0]
        else:
            action_time_diff_first1 = time_diff.iloc[1]  #第一个时间间隔 真正的第一个为0
        action_time_diff_last1_3 = np.array([action_time_diff_last1,action_time_diff_last2,action_time_diff_last3])
        action_time_diff_last1_3_mean = action_time_diff_last1_3.mean()  #最后三个时间间隔均值
        action_time_diff_last1_3_var = action_time_diff_last1_3.var()    #最后三个时间间隔方差
        action_type = group['actionType']
        action_size = action_type.size
        action_type_last = action_type.iloc[-1] #type最后一个
        action_type_first = action_type.iloc[0] #第一个type
        if action_size == 2:
            action_type_last2 = action_type.iloc[-1] #倒数第二个type
            action_type_last3 = action_type.iloc[-2] #倒数第三个type
        elif action_size == 1:
            action_type_last2 = action_type.iloc[-1]
            action_type_last3 = action_type.iloc[-1]
        else:
            action_type_last2 = action_type.iloc[-2]
            action_type_last3 = action_type.iloc[-3]

        max_time = group['actionTime'].max() #最后一个action的时间
        action_time_delta = (max_time - group['actionTime'].min()).seconds #action时间的最大时间间隔
        action_list = set(action_type)

        if 1 in action_list:
            group_1 = group[group['actionType'] == 1]
            dis1_time = (max_time - group_1['actionTime'].max()).seconds #离最近的action1的时间
            action1_time_delta = (group_1['actionTime'].max() - group_1['actionTime'].min()).seconds #第一个action1到最后一个action1的最大时间间隔
        else:
            dis1_time = 0
            action1_time_delta = action_time_delta

        if 2 in action_list:
             group_2 = group[group['actionType'] == 2]
             nearest2_loc = action_type[action_type == 2].index[-1] #离最近的2的索引
             dis2 = action_size -  nearest2_loc#离最近的2的距离
             dis2_time = ( max_time- group_2['actionTime'].max()).seconds #离最近的action2的时间
             dis2_time_diff = group['time_diff'][nearest2_loc:]
             dis2_time_diff_min = dis2_time_diff.min() #离最近的2的时间间隔最小值
             dis2_time_diff_max = dis2_time_diff.max() #离最近的2的时间间隔最大值
             dis2_time_diff_mean = dis2_time_diff.mean() #离最近的2的时间间隔均值
             dis2_time_diff_var = dis2_time_diff.var() #离最近的2的时间间隔方差
        else:
             dis2 = 0 #不存在action2，设置为0
             dis2_time = 0
             dis2_time_diff_min = 0
             dis2_time_diff_max = 0
             dis2_time_diff_mean = 0
             dis2_time_diff_var = 0

        if 3 in action_list:
             group_3 = group[group['actionType'] == 3]
             nearest3_loc = action_type[action_type == 3].index[-1] #离最近的3的索引
             dis3 = action_size - nearest3_loc #离最近的3的距离
             dis3_time = ( max_time- group_3['actionTime'].max()).seconds #离最近的action3的时间
             dis3_time_diff = group['time_diff'][nearest3_loc:]
             dis3_time_diff_min = dis3_time_diff.min() #离最近的3的时间间隔最小值
             dis3_time_diff_max = dis3_time_diff.max() #离最近的3的时间间隔最大值
             dis3_time_diff_mean = dis3_time_diff.mean() #离最近的3的时间间隔均值
             dis3_time_diff_var = dis3_time_diff.var() #离最近的3的时间间隔方差
        else:
             dis3 = 0
             dis3_time = 0
             dis3_time_diff_min = 0
             dis3_time_diff_max = 0
             dis3_time_diff_mean = 0
             dis3_time_diff_var = 0

        if 4 in action_list:
             group_4 = group[group['actionType'] == 4]
             nearest4_loc = action_type[action_type == 4].index[-1] #离最近的4的索引
             dis4 = action_size - nearest4_loc #离最近的4的距离
             dis4_time = ( max_time- group_4['actionTime'].max()).seconds #离最近的action4的时间
             dis4_time_diff = group['time_diff'][nearest4_loc:]
             dis4_time_diff_min = dis4_time_diff.min() #离最近的4的时间间隔最小值
             dis4_time_diff_max = dis4_time_diff.max() #离最近的4的时间间隔最大值
             dis4_time_diff_mean = dis4_time_diff.mean() #离最近的4的时间间隔均值
             dis4_time_diff_var = dis4_time_diff.var() #离最近的4的时间间隔方差
        else:
             dis4 = 0
             dis4_time = 0
             dis4_time_diff_min = 0
             dis4_time_diff_max = 0
             dis4_time_diff_mean = 0
             dis4_time_diff_var = 0

        if 5 in action_list:
             nearest5_loc = action_type[action_type == 5].index[-1] #离最近的5的索引
             dis5 = action_size - nearest5_loc #离最近的5的距离
             group_5 = group[group['actionType'] == 5]
             dis5_time = ( max_time- group_5['actionTime'].max()).seconds #离最近的action5的时间
             action5_time_delta = (group_5['actionTime'].max() - group_5['actionTime'].min()).seconds #actionType5是最重要的特征 所以单独计算所有actionType5之间的最大时间间隔
             action1_5time_delta = (group_5['actionTime'].min() - group['actionTime'].min()).seconds #第一个action5到第一个action1的距离
             dis5_time_diff = group['time_diff'][nearest5_loc:]
             dis5_time_diff_min = dis5_time_diff.min() #离最近的5的时间间隔最小值
             dis5_time_diff_max = dis5_time_diff.max() #离最近的5的时间间隔最大值
             dis5_time_diff_mean = dis5_time_diff.mean() #离最近的5的时间间隔均值
             dis5_time_diff_var = dis5_time_diff.var() #离最近的5的时间间隔方差
        else:
             dis5 = 0
             dis5_time = 0
             action5_time_delta = action_time_delta #不存在 设置为最大距离
             action1_5time_delta = action_time_delta
             dis5_time_diff_min = 0
             dis5_time_diff_max = 0
             dis5_time_diff_mean = 0
             dis5_time_diff_var = 0

        if 6 in action_list:
             nearest6_loc = action_type[action_type == 6].index[-1] #离最近的6的索引
             dis6 = action_size - nearest6_loc #离最近的6的距离
             group_6 = group[group['actionType'] == 6]
             dis6_time = ( max_time- group_6['actionTime'].max()).seconds #离最近的action6的时间
             dis6_time_diff = group['time_diff'][nearest6_loc:]
             dis6_time_diff_min = dis6_time_diff.min() #离最近的6的时间间隔最小值
             dis6_time_diff_max = dis6_time_diff.max() #离最近的6的时间间隔最大值
             dis6_time_diff_mean = dis6_time_diff.mean() #离最近的6的时间间隔均值
             dis6_time_diff_var = dis6_time_diff.var() #离最近的6的时间间隔方差
        else:
             dis6 = 0
             dis6_time = 0
             dis6_time_diff_min = 0 #离最近的6的时间间隔最小值
             dis6_time_diff_max = 0 #离最近的6的时间间隔最大值
             dis6_time_diff_mean = 0 #离最近的6的时间间隔均值
             dis6_time_diff_var = 0 #离最近的6的时间间隔方差

        if 7 in action_list:
             nearest7_loc = action_type[action_type == 7].index[-1] #离最近的7的索引
             group_7 = group[group['actionType'] == 7]
             dis7 = action_size - nearest7_loc
             dis7_time = ( max_time- group_7['actionTime'].max()).seconds #离最近的action7的时间
             dis7_time_diff = group['time_diff'][nearest7_loc:]
             dis7_time_diff_min = dis7_time_diff.min() #离最近的7的时间间隔最小值
             dis7_time_diff_max = dis7_time_diff.max() #离最近的7的时间间隔最大值
             dis7_time_diff_mean = dis7_time_diff.mean() #离最近的7的时间间隔均值
             dis7_time_diff_var = dis7_time_diff.var() #离最近的7的时间间隔方差
        else:
             dis7 = 0
             dis7_time = 0
             dis7_time_diff_min = 0  #离最近的7的时间间隔最小值
             dis7_time_diff_max = 0 #离最近的7的时间间隔最大值
             dis7_time_diff_mean = 0 #离最近的7的时间间隔均值
             dis7_time_diff_var = 0 #离最近的7的时间间隔方差

        if 8 in action_list:
             nearest8_loc = action_type[action_type == 8].index[-1] #离最近的8的索引
             group_8 = group[group['actionType'] == 8]
             dis8 = action_size - nearest8_loc
             dis8_time = ( max_time- group_8['actionTime'].max()).seconds #离最近的action8的时间
             dis8_time_diff = group['time_diff'][nearest8_loc:]
             dis8_time_diff_min = dis8_time_diff.min() #离最近的8的时间间隔最小值
             dis8_time_diff_max = dis8_time_diff.max() #离最近的8的时间间隔最大值
             dis8_time_diff_mean = dis8_time_diff.mean() #离最近的8的时间间隔均值
             dis8_time_diff_var = dis8_time_diff.var() #离最近的8的时间间隔方差
        else:
             dis8 = 0
             dis8_time = 0
             dis8_time_diff_min = 0  #离最近的8的时间间隔最小值
             dis8_time_diff_max = 0 #离最近的8的时间间隔最大值
             dis8_time_diff_mean = 0 #离最近的8的时间间隔均值
             dis8_time_diff_var = 0 #离最近的8的时间间隔方差

        if 9 in action_list:
             nearest9_loc = action_type[action_type == 9].index[-1] #离最近的9的索引
             group_9 = group[group['actionType'] == 9]
             dis9 = action_size - nearest9_loc #离最近的9的距离
             dis9_time = ( max_time- group_9['actionTime'].max()).seconds #离最近的action9的时间
             dis9_time_diff = group['time_diff'][nearest8_loc:]
             dis9_time_diff_min = dis9_time_diff.min() #离最近的9的时间间隔最小值
             dis9_time_diff_max = dis9_time_diff.max() #离最近的9的时间间隔最大值
             dis9_time_diff_mean = dis9_time_diff.mean() #离最近的9的时间间隔均值
             dis9_time_diff_var = dis9_time_diff.var() #离最近的9的时间间隔方差
             dis9_time_diff_var_mean =     dis9_time_diff_mean * dis9_time_diff_var
        else:
             dis9 = 0
             dis9_time = 0
             dis9_time_diff_min = 0  #离最近的9的时间间隔最小值
             dis9_time_diff_max = 0 #离最近的9的时间间隔最大值
             dis9_time_diff_mean = 0 #离最近的9的时间间隔均值
             dis9_time_diff_var = 0 #离最近的9的时间间隔方差
             dis9_time_diff_var_mean = 0

        action_time_fea.append([userid,action_time_delta,action5_time_delta,action1_5time_delta,action1_time_delta,
                                action_time_diff_mean,action_time_diff_var,action_time_diff_min,
                                action_time_diff_last1,action_time_diff_last2,action_time_diff_last3,action_time_diff_last4,
                                action_time_diff_first1,action_time_diff_last1_3_mean,action_time_diff_last1_3_var,
                                action_type_last,action_type_first,action_type_last2,action_type_last3,
                                dis2,dis3,dis4,dis5,dis6,dis7,dis8,dis9,
                                dis1_time,dis2_time,dis3_time,dis4_time,dis5_time,dis6_time,dis7_time,dis8_time,dis9_time,
                                dis2_time_diff_min,dis3_time_diff_min,dis4_time_diff_min,dis5_time_diff_min,dis6_time_diff_min,dis7_time_diff_min,dis8_time_diff_min,dis9_time_diff_min,
                                dis2_time_diff_max,dis3_time_diff_max,dis4_time_diff_max,dis5_time_diff_max,dis6_time_diff_max,dis7_time_diff_max,dis8_time_diff_max,dis9_time_diff_max,
                                dis2_time_diff_mean,dis3_time_diff_mean,dis4_time_diff_mean,dis5_time_diff_mean,dis6_time_diff_mean,dis7_time_diff_mean,dis8_time_diff_mean,dis9_time_diff_mean,
                                dis2_time_diff_var,dis3_time_diff_var,dis4_time_diff_var,dis5_time_diff_var,dis6_time_diff_var,dis7_time_diff_var,dis8_time_diff_var,dis9_time_diff_var, dis9_time_diff_var_mean,

                                ])
        #stop_time = time.time()
        #print 'This epoch use: ', stop_time - start_time
        flag += 1


    action_time_fea = pd.DataFrame(action_time_fea,columns=
                                ['userid','action_time_delta','action5_time_delta','action1_5time_delta','action1_time_delta',
                                'action_time_diff_mean','action_time_diff_var','action_time_diff_min',
                                'action_time_diff_last1','action_time_diff_last2','action_time_diff_last3','action_time_diff_last4',
                                'action_time_diff_first1','action_time_diff_last1_3_mean','action_time_diff_last1_3_var',
                                'action_type_last','action_type_first','action_type_last2','action_type_last3',
                                'dis2','dis3','dis4','dis5','dis6','dis7','dis8','dis9',
                                'dis1_time','dis2_time','dis3_time','dis4_time','dis5_time','dis6_time','dis7_time','dis8_time','dis9_time',
                                'dis2_time_diff_min','dis3_time_diff_min','dis4_time_diff_min','dis5_time_diff_min','dis6_time_diff_min','dis7_time_diff_min','dis8_time_diff_min','dis9_time_diff_min',
                                'dis2_time_diff_max','dis3_time_diff_max','dis4_time_diff_max','dis5_time_diff_max','dis6_time_diff_max','dis7_time_diff_max','dis8_time_diff_max','dis9_time_diff_max',
                                'dis2_time_diff_mean','dis3_time_diff_mean','dis4_time_diff_mean','dis5_time_diff_mean','dis6_time_diff_mean','dis7_time_diff_mean','dis8_time_diff_mean','dis9_time_diff_mean',
                                'dis2_time_diff_var','dis3_time_diff_var','dis4_time_diff_var','dis5_time_diff_var','dis6_time_diff_var','dis7_time_diff_var','dis8_time_diff_var','dis9_time_diff_var','dis9_time_diff_var_mean',
                                ])

    #df['df_action_type_median'] = df['actionType'].median() #中间状态 无效特征
    actionType=pd.get_dummies(df['actionType'],prefix='actionType') # 生成行为类型的OneHot特征 prefix表示OneHote编码特征的命名前缀 默认为“prefix_特征值”
    df=pd.concat([df['userid'],actionType],axis=1)
    df=df.groupby('userid',as_index=False).sum() #统计各个actionType的个数
    for column in range(1,df.shape[1]):
        df['actionType_{}'.format(column)]=df['actionType_{}'.format(column)]/df_count['count'] #求每种行为占行为总数的比例
    df = pd.merge(df,action_time_fea,how='left',on='userid')
    print u'action表特征提取完毕！'
    return df

#用户信息表中提取特征
def profile_feat(df):
    df['gender'] = df['gender'].map(list(df['gender'].unique()).index)
    df['province'] = df['province'].map(list(df['province'].unique()).index)
    df['age'] = df['age'].map(list(df['age'].unique()).index)
    print u'用户表特征提取完毕！'
    return df

#将评为5星与非5星各分成1类
def func_change_rating(x):
    if x==5:
        return 1
    else:
        return 0

#获取标签的个数
def func_getlen_tag(x):
    if type(x) == float:
        return 0
    else:
        return len(x.split('|'))

#评论表中提取特征  1个
def userComment_feat(df):
    df['rating'] = df['rating'].astype(int) #评分值中含有小数
    df['rating'] = df['rating'].map(func_change_rating)  #注意应用map模块不需要使用func()，只要方法名

    df['tags_len'] = df['tags'].map(func_getlen_tag)
    print u'评论表特征提取完毕！'
    return df[['userid','rating','tags_len']]

#绘制特征值分布柱状图
def draw_distributing(df,col_num):
    comment.rating.value_counts().plot(kind='bar')
    plt.ylabel('num')
    plt.title('rating counts')
    plt.show()

def draw_fea_importance(model,features_list):
    #features = train_data.columns
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    fi_threshold = 1
    important_idx = np.where(feature_importance > fi_threshold)[0]
    important_features = features_list[important_idx]
    sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
    #get the figure about important features
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.title('Feature Importance')
    plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]],color='r',align='center')
    plt.yticks(pos, important_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    plt.draw()
    plt.show()

def gen_train_feat():
    actions=orderFuture_train
    actions=pd.merge(actions,orderHistory_feat(orderHistory_train),on='userid',how='left')
    actions=pd.merge(actions,userComment_feat(userComment_train),on='userid',how='left')
    actions=pd.merge(actions,profile_feat(userProfile_train),on='userid',how='left')
    actions=pd.merge(actions,actions_orderType(action_train),on='userid',how='left')
    ###add feature###
    return actions

def gen_test_feat():
    actions=orderFuture_test
    actions=pd.merge(actions,orderHistory_feat(orderHistory_test),on='userid',how='left')
    actions=pd.merge(actions,userComment_feat(userComment_test),on='userid',how='left')
    actions=pd.merge(actions,profile_feat(userProfile_test),on='userid',how='left')
    actions=pd.merge(actions,actions_orderType(action_test),on='userid',how='left')
    return actions


train_data=gen_train_feat()
print u'训练集特征提取完毕！ '
test_data=gen_test_feat()
print u'测试集特征提取完毕！ '

del train_data['userid']
del test_data['userid']
#train_data.fillna(0,inplace=True)
#test_data.fillna(0,inplace=True)
from sklearn.model_selection import train_test_split
train_label=train_data['label']
del train_data['label']

x_train,x_val,y_train,y_val=train_test_split(train_data,train_label,test_size=0.2,random_state=100)

'''
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_val)

from sklearn.metrics import roc_auc_score
print("AUC is %f" % roc_auc_score(y_val,y_pred))

fea_list = train_data.columns.values
draw_fea_importance(model,fea_list)
'''

import xgboost as xgb
print ('start running ....')
dtrain = xgb.DMatrix(x_train,label=y_train)
dval = xgb.DMatrix(x_val,label=y_val)
param = {
        'learning_rate' : 0.1,
        'n_estimators': 1000,
        'max_depth': 3,
        'min_child_weight': 5,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eta': 0.05,
        'silent': 1,
        'objective':
        'binary:logistic',
        'scale_pos_weight':1
        }

num_round =1000
plst = list(param.items())
plst += [('eval_metric', 'auc')]
evallist = [(dval, 'eval'), (dtrain, 'train')]
bst=xgb.train(plst,dtrain,num_round,evallist,early_stopping_rounds=200)
dtest = xgb.DMatrix(test_data)
y = bst.predict(dtest)
'''
print 'Start Running!'
dtrain = lgb.Dataset(x_train,label=y_train)
dval = lgb.Dataset(x_val,label=y_val)
#dtest = lgb.Dataset(test_data) lgb做预测不需要转换格式
params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'num_leaves': 3,
        'learning_rate': 0.01,
        'feature_fraction': 0.83,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 1
}
gbm = lgb.train(params,
                dtrain,
                num_boost_round=1000,
                valid_sets=[dtrain, dval],
                verbose_eval = True)
y = gbm.predict(dtest)
'''
#orderFuture_test['orderType']=y
#orderFuture_test.to_csv('../output/0125_xgb2.csv',index=False)
