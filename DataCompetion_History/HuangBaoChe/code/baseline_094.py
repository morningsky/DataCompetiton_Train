# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 13:20:51 2018

@author: sky
"""

import pandas as pd
import numpy as np
import time

ac = pd.read_csv('../data/trainingset/action_train.csv')
order = pd.read_csv('../data/trainingset/orderFuture_train.csv')
order_his = pd.read_csv('../data/trainingset/orderHistory_train.csv')
comment = pd.read_csv('../data/trainingset/userComment_train.csv')
profile = pd.read_csv('../data/trainingset/userProfile_train.csv')

ac_test = pd.read_csv('../data/test/action_test.csv')
order_test = pd.read_csv('../data/test/orderFuture_test.csv')
order_his_test = pd.read_csv('../data/test/orderHistory_test.csv')
comment_test = pd.read_csv('../data/test/userComment_test.csv')
profile_test = pd.read_csv('../data/test/userProfile_test.csv')

# 用户id，行为类型，发生时间
# 行为类型一共有9个，其中1是唤醒app；2~4是浏览产品，无先后关系；5~9则是有先后关系的，从填写表单到提交订单再到最后支付
ac.head()

order.head()
#两列，分别是用户id和订单类型。供参赛者训练模型使用。其中1表示购买了精品旅游服务，0表示未购买精品旅游服务（包括普通旅游服务和未下订单）

# 预测订单类型

comment.head(5)

order_his.head()
# 数据共有7列，分别是用户id，订单id，订单时间，订单类型，旅游城市，国家，大陆。其中1表示购买了精品旅游服务，0表示普通旅游服务。

profile.head()
# 数据共有四列，分别是用户id、性别、省份、年龄段

timeArray = time.localtime(1474300753)
time.strftime('%Y-%m-%d %H:%M:%S', timeArray)

order[order.userid == 100000000231]

comment.commentsKeyWords.head(20)

def time_help(x):
    timeArray = time.localtime(x)
    return time.strftime('%Y-%m-%d %H:%M:%S', timeArray)
    
ac.actionTime = pd.to_datetime(ac.actionTime.apply(lambda x:time_help(x)), format='%Y-%m-%d %H:%M:%S')
order_his.orderTime = pd.to_datetime(order_his.orderTime.apply(lambda x:time_help(x)), format='%Y-%m-%d %H:%M:%S')
ac_test.actionTime = pd.to_datetime(ac_test.actionTime.apply(lambda x:time_help(x)), format='%Y-%m-%d %H:%M:%S')
order_his_test.orderTime = pd.to_datetime(order_his_test.orderTime.apply(lambda x:time_help(x)), format='%Y-%m-%d %H:%M:%S')

ac.head()

# t = ac.actionTime.iloc[1]
ac['day'] = ac.actionTime.apply(lambda x:x.day)
ac_test['day'] = ac_test.actionTime.apply(lambda x:x.day)

# print(t.year, t.month, t.day, t.second, t.minute,t.hour)


# ac['count'] = 1
# ac[['userid', 'actionType']].groupby('userid', as_index=False).agg('last')

# ac[(ac.userid == 100000000459) && (ac.actionT)]

# order[order.userid == 100000000459]
# order['count'] = 1
# order[['userid', 'count']].groupby('userid', as_index=False).agg('count')
# del order['count']

# ac[['userid', 'actionTime']].groupby('userid', as_index=False).agg('last')
# ac[['userid', 'actionTime']].groupby('userid', as_index=False).agg('first')

# ac.actionTime.
# ac = pd.read_csv('data/trainset/action_train.csv')
# ac[['userid', 'actionTime']].groupby('userid', as_index=False).agg('last')
# ac[['userid', 'actionTime']].groupby('userid', as_index=False).agg('mean')

# order_his[['userid', 'orderType']].groupby('userid', as_index=False).agg('max')

# order  对应 userid,  所在  order_his的 最后一个 的 城市

# order_his.head()
# order_his.drop_duplicates(subset='userid', keep='last')
# order_his[[order_his[['userid']].drop_duplicates(keep='last').index]]
# pd.merge(order_his[['userid']].drop_duplicates(keep='last'), order_his, on='userid', how='left')


# len(order_his_test.city.unique())

##user过去是否订购过orderType    以及比率
def orderHistory_feat(df):
    grouped=df[['userid','orderType']].groupby('userid',as_index=False)
    df_count=grouped.count()
    df_count.rename(columns={'orderType':'df_count'},inplace=True)
    df_sum=grouped.sum()
    
    df_sum.rename(columns={'orderType':'df_sum'},inplace=True)
    df_merge=pd.merge(df_count,df_sum,on='userid',how='left')
    df_merge['rate']=df_merge['df_sum']/df_merge['df_count']
    del df_merge['df_count']
    df_merge.rename(columns={'df_sum':'orderHistory_feat_sum','rate':'orderHistory_feat_rate'},inplace=True)
    
    # rate  的 score 很低
    del df_merge['orderHistory_feat_rate']
    return df_merge

# 点击比率
def actions_orderType(df):
    df['count']=1
    df_count=df[['userid','count']].groupby('userid',as_index=False).count()
    actionType=pd.get_dummies(df['actionType'],prefix='actionType')
    df=pd.concat([df['userid'],actionType],axis=1)
    df=df.groupby('userid',as_index=False).sum()
    for column in range(1,df.shape[1]):
        df['actionType_{}'.format(column)]=df['actionType_{}'.format(column)]/df_count['count']
    return df


def get_two(arr):
    try:
        tem = int(arr.iloc[-2])
        return tem
    except:
        return np.nan

def get_three(arr):
    try:
        tem = int(arr.iloc[-3])
        return tem
    except:
        return np.nan

# order 的 actionType 的最后一个 行为， 倒数第二个， 倒数第三个
def last_action (df):
    last_type = df[['userid', 'actionType']].groupby('userid', as_index=False).agg('last')
    last_type.rename(columns={'actionType':'last_type'}, inplace=True)
    
    two_type = df[['userid', 'actionType']].groupby('userid', as_index=False).agg(get_two)
    two_type.rename(columns={'actionType':'two_type'}, inplace=True)
    
    three_type = df[['userid', 'actionType']].groupby('userid', as_index=False).agg(get_three)
    three_type.rename(columns={'actionType':'three_type'}, inplace=True)
    two_type
    
    last = pd.merge(last_type, two_type, on='userid', how='left')
    last = pd.merge(last, three_type, on='userid', how='left')
    
    return last
    

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# order所在城市
def last_city(train, test):
    city = list(order_his['city'].unique())
    city_test = list(order_his_test['city'].unique())
    for i in city_test:
        if i not in city:
            city.append(i)
    le.fit(city)
    
    new = order_his[['userid', 'city', 'continent']].drop_duplicates(subset='userid', keep='last')
    new['city'] = le.transform(new['city'])
   
    new_test = order_his_test[['userid', 'city', 'continent']].drop_duplicates(subset='userid', keep='last')
    new_test['city'] = le.transform(new_test['city']) 
    
    new['continent'] = le.fit_transform(new['continent'])
    new_test['continent'] = le.transform(new_test['continent'])
    
    train = pd.merge(train, new, on='userid', how='left')
    test = pd.merge(test, new_test, on='userid', how='left')
    return train, test


def profile_me(train, test):
    
    for i in ['province', 'gender', 'age']:
        train[i] = train[i].factorize()[0]
        test[i] = test[i].factorize()[0]
        
        train[i] = le.fit_transform(train[i])
        test[i] = le.transform(test[i])
    
    return train, test


def fun1(arr):
#     print(arr.iloc[-1], arr.iloc[-2])
    try:
        return ((arr.iloc[-1]) - (arr.iloc[-2])).seconds
    except:
        return np.nan

def fun2(arr):
    try:
        return ((arr.iloc[-1]) - (arr.iloc[-2])).days
    except:
        return np.nan

def fun3(arr):
    try:
        return ((arr.iloc[-2]) - (arr.iloc[-3])).days
    except:
        return np.nan

def fun4(arr):
    try:
        return ((arr.iloc[-2]) - (arr.iloc[-3])).seconds
    except:
        return np.nan
        
# ac 倒数第一次和 倒数第二次， 倒数第三次的间隔时间
def ac_time(train, test):

    las_two_day = ac[['userid', 'actionTime']].groupby('userid', as_index=False).agg(fun2)
    las_two_day.rename(columns={'actionTime':'last_two_day'}, inplace=True)
    las_two_second = ac[['userid', 'actionTime']].groupby('userid', as_index=False).agg(fun1)
    las_two_second.rename(columns={'actionTime':'last_two_second'}, inplace=True)
    
    two_three_day = ac[['userid', 'actionTime']].groupby('userid', as_index=False).agg(fun3)
    two_three_day.rename(columns={'actionTime':'two_three_day'}, inplace=True)
    two_three_second = ac[['userid', 'actionTime']].groupby('userid', as_index=False).agg(fun4)
    two_three_second.rename(columns={'actionTime':'two_three_second'}, inplace=True)
    
#         time_mean = ac[['userid', 'actionTime']].groupby('userid', as_index=False).agg(fun3)
#         time_mean.rename(columns={'actionTime':'time_mean'}, inplace=True)
#         time_std = ac[['userid', 'actionTime']].groupby('userid', as_index=False).agg(fun4)
#         time_std.rename(columns={'actionTime':'time_std'}, inplace=True)

    train = pd.merge(train, las_two_day, on='userid', how='left')
    train = pd.merge(train, las_two_second, on='userid', how='left')
    train = pd.merge(train, two_three_day, on='userid', how='left')
    train = pd.merge(train, two_three_second, on='userid', how='left')

    las_two_day_ = ac_test[['userid', 'actionTime']].groupby('userid', as_index=False).agg(fun2)
    las_two_day_.rename(columns={'actionTime':'last_two_day'}, inplace=True)
    las_two_second_ = ac_test[['userid', 'actionTime']].groupby('userid', as_index=False).agg(fun1)
    las_two_second_.rename(columns={'actionTime':'last_two_second'}, inplace=True)
    
    two_three_day = ac_test[['userid', 'actionTime']].groupby('userid', as_index=False).agg(fun3)
    two_three_day.rename(columns={'actionTime':'two_three_day'}, inplace=True)
    two_three_second = ac_test[['userid', 'actionTime']].groupby('userid', as_index=False).agg(fun4)
    two_three_second.rename(columns={'actionTime':'two_three_second'}, inplace=True)
    
#         time_std = ac_test[['userid', 'actionTime']].groupby('userid', as_index=False).agg(fun4)
#         time_std.rename(columns={'actionTime':'time_std'}, inplace=True)


    test = pd.merge(test, las_two_day_, on='userid', how='left')
    test = pd.merge(test, las_two_second_, on='userid', how='left')
    test = pd.merge(test, two_three_day, on='userid', how='left')
    test = pd.merge(test, two_three_second, on='userid', how='left')

    return train, test

# comment rating
# def ratting_score()

def fun5(timeseries):
# 时间间隔的均值
    sum = []
    for i in range(len(timeseries) - 1):
        sum.append((timeseries.iloc[i+ 1] - timeseries.iloc[i]).seconds)
    return np.mean(sum)

def fun6(timeseries):
# 时间间隔的均值
    sum = []
    for i in range(len(timeseries) - 1):
        sum.append((timeseries.iloc[i+1] - timeseries.iloc[i]).seconds)
    return np.std(sum)

def time_delta(train, test):
    time_mean = ac[['userid', 'actionTime']].groupby('userid', as_index=False).agg(fun5)
    time_mean.rename(columns={'actionTime':'timedelta_mean'}, inplace=True)
    time_std = ac[['userid', 'actionTime']].groupby('userid', as_index=False).agg(fun5)
    time_std.rename(columns={'actionTime':'timedelta_std'}, inplace=True)
    
    train = pd.merge(train, time_mean, on='userid', how='left')
    train = pd.merge(train, time_std, on='userid', how='left')
    
    time_mean = ac_test[['userid', 'actionTime']].groupby('userid', as_index=False).agg(fun5)
    time_mean.rename(columns={'actionTime':'timedelta_mean'}, inplace=True)
    time_std = ac_test[['userid', 'actionTime']].groupby('userid', as_index=False).agg(fun6)
    time_std.rename(columns={'actionTime':'timedelta_std'}, inplace=True)
    
    test = pd.merge(test, time_mean, on='userid', how='left')
    test = pd.merge(test, time_std, on='userid', how='left')
    return train, test

# 历史最后一次时间对应action中点击的什么, 组合
def his_s(train, test):
    df = order_his.drop_duplicates(subset='userid', keep='last').reset_index(drop=True)
    last_type = []
    for i in range(len(df)):
        temp_time = df.loc[i, 'orderTime']
        temp_id = df.loc[i, 'userid']
        new = ac[ac.userid == temp_id]
        new = new[new.actionTime < temp_time]
        if(len(new) != 0):
            new = new.iloc[-1]
            last_type.append(new.actionType)
        else:
            last_type.append(np.nan)
    last_type = pd.Series(np.array(last_type))
    last_type.index = df.index
    last_type = pd.DataFrame(last_type, columns=['hist_type'])
    last_type['userid'] = df['userid']
    train = pd.merge(train, last_type, on='userid', how='left')
    
    df = order_his_test.drop_duplicates(subset='userid', keep='last').reset_index(drop=True)
    last_type = []
    for i in range(len(df)):
        temp_time = df.loc[i, 'orderTime']
        temp_id = df.loc[i, 'userid']
        new = ac_test[ac_test.userid == temp_id]
        new = new[new.actionTime < temp_time]
        if(len(new) != 0):
            new = new.iloc[-1]
            last_type.append(new.actionType)
        else:
            last_type.append(np.nan)
    last_type = pd.Series(np.array(last_type))
    last_type.index = df.index
    last_type = pd.DataFrame(last_type, columns=['hist_type'])
    last_type['userid'] = df['userid']
    
    test = pd.merge(test, last_type, on='userid', how='left')
    return train, test


# def fun7(type)

# 距离 最近的 5, 6 , 7 , 8的 seconds
def distance_last(odf, a=5):
    name = 'flag_'+ str(a)
    odf[name] = (odf.actionType == a) * 1
    
    id = []
    dis = []
    for i in list(odf.userid.unique()):
        id.append(i)
        
        df = odf[odf.userid == i]
        time1 = df.actionTime.iloc[-1]
        
        df = df[df[name] == 1]
        if (len(df) != 0):
            if df.iloc[-1].actionTime == time1:
                df = df[:-1]
            if len(df) >= 1:
                dis.append((time1 - df.iloc[-1].actionTime).seconds)
            else:
                dis.append(np.nan)
        else:
            dis.append(np.nan)
    column_name = 'last_' + str(a) + 'dis'
    new_df = pd.DataFrame({'userid':id,
                           column_name:dis})
    return new_df
    
        
    

# 最后操作那一天 action 的count
def action_count(train, test):
    count = []
    id = []
    for i in (ac.userid.unique()):
        id.append(i)
        df = ac[ac.userid == i]
        df = df.loc[df.day == df.iloc[-1].day]
        count.append(len(df))
    new = pd.DataFrame({'userid':id, 
                       'count':count})
    train = pd.merge(train, new, on='userid', how='left')
    
    count = []
    id = []
    for i in (ac_test.userid.unique()):
        id.append(i)
        df = ac_test[ac_test.userid == i]
        df = df.loc[df.day == df.iloc[-1].day]
        count.append(len(df))
    new = pd.DataFrame({'userid':id, 
                       'count':count})
    test = pd.merge(test, new, on='userid', how='left')
    return train, test

# orderType
def flags_buy(df):
    flags = []
    id = []
    for i in list(df.userid.unique()):
        id.append(i)
        df_ = df[df.userid == i]
        if df_.orderType.sum() != 0:
            flags.append(1)
        else:
            flags.append(0)
    new = pd.DataFrame({
        'userid':id,
        'buy_flags':flags
    })
    return new

def gen_train_feat():
    actions = order
    actions_test = order_test
    
   
    actions = pd.merge(actions, distance_last(ac, 5), on='userid', how='left')
    actions_test = pd.merge(actions_test, distance_last(ac_test, 5), on='userid', how='left')
    
    actions = pd.merge(actions, distance_last(ac, 6), on='userid', how='left')
    actions_test = pd.merge(actions_test, distance_last(ac_test, 6), on='userid', how='left')
    
    actions = pd.merge(actions, distance_last(ac, 7), on='userid', how='left')
    actions_test = pd.merge(actions_test, distance_last(ac_test, 7), on='userid', how='left')
    
    actions = pd.merge(actions, distance_last(ac, 8), on='userid', how='left')
    actions_test = pd.merge(actions_test, distance_last(ac_test, 8), on='userid', how='left')
    
    actions = pd.merge(actions, distance_last(ac, 9), on='userid', how='left')
    actions_test = pd.merge(actions_test, distance_last(ac_test, 9), on='userid', how='left')
    actions, actions_test = action_count(actions, actions_test)
    actions = pd.merge(actions,orderHistory_feat(order_his),on='userid',how='left')
    actions = pd.merge(actions,actions_orderType(ac),on='userid',how='left')
    actions = pd.merge(actions,last_action(ac),on='userid',how='left')  
    
    actions_test = pd.merge(actions_test,orderHistory_feat(order_his_test),on='userid',how='left')
    actions_test = pd.merge(actions_test,actions_orderType(ac_test),on='userid',how='left')
    actions_test = pd.merge(actions_test,last_action(ac_test),on='userid',how='left')
    
    actions, actions_test = last_city(actions, actions_test)
    
    profile_, profile_test_ = profile_me(profile, profile_test)
    actions = pd.merge(actions, profile_, on='userid', how='left')
    actions_test = pd.merge(actions_test, profile_test_, on='userid', how='left')
    actions, actions_test = ac_time(actions, actions_test)
    
    actions = pd.merge(actions, comment[['userid', 'rating']], on='userid', how='left')
    actions_test = pd.merge(actions_test, comment_test[['userid', 'rating']], on='userid', how='left')
    
    actions['flag_five'] = (actions['last_type'] == 5) * 1
    actions['flag_six'] = (actions['last_type'] == 6) * 1
    actions['flag_seven'] = (actions['last_type'] == 1) * 1
    
    actions_test['flag_five'] = (actions_test['last_type'] == 5) * 1
    actions_test['flag_six'] = (actions_test['last_type'] == 6) * 1
    actions_test['flag_seven'] = (actions_test['last_type'] == 7) * 1    
    return actions, actions_test


order.rename(columns={'orderType':'label'},inplace=True)
train_data2, test_data2=gen_train_feat()
train_data2.head()
