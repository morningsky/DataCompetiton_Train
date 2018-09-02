#coding:utf-8

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import roc_auc_score

# 获取训练数据
def get_train_info():
    # userid actionType actionTime
    action_train = pd.read_csv('../data/trainingset/action_train.csv')
    # userid gender province age
    userProfile_train = pd.read_csv('../data/trainingset/userProfile_train.csv')
    # userid orderid rating tags commentsKeyWords
    userComment_train = pd.read_csv('../data/trainingset/userComment_train.csv')
    # userid  orderType
    orderFuture_train = pd.read_csv('../data/trainingset/orderFuture_train.csv')
    # userid orderid orderTime orderType city country continent
    orderHistory_train = pd.read_csv('../data/trainingset/orderHistory_train.csv')
    return action_train,userProfile_train,userComment_train,orderFuture_train,orderHistory_train

# 获取测试数据
def get_test_info():
    # userid actionType actionTime
    action_test = pd.read_csv('../data/test/action_test.csv')
    # userid gender province age
    userProfile_test = pd.read_csv('../data/test/userProfile_test.csv')
    # userid orderid rating tags commentsKeyWords
    userComment_test = pd.read_csv('../data/test/userComment_test.csv')
    # userid  orderType == 需要预测的值
    orderFuture_test = pd.read_csv('../data/test/orderFuture_test.csv')
    # userid orderid orderTime orderType city country continent
    orderHistory_test = pd.read_csv('../data/test/orderHistory_test.csv')
    return action_test, userProfile_test, userComment_test, orderFuture_test, orderHistory_test


# 编码cate信息
def code_category(train,test,cate_list = None):
    print('category2code')
    train_test = pd.concat([train,test],axis=0)
    if cate_list is None:
        pass
    else:
        for cate in cate_list:
            train_test[cate] = train_test[cate].astype('category')
            train_test[cate] = train_test[cate].cat.codes
    del train
    del test
    return train_test

# 获取行为时间差
def get_time_diff(actionTime):
    actionTime = list(actionTime)
    if actionTime.__len__() == 1:
        return -1
    else:
        return max(actionTime) - min(actionTime)

# 行为转为向量
def action2len(actionType):
    actionType = list(actionType)
    actionType = ''.join(actionType)
    actionType = len(actionType)
    return actionType

# 时间差特征
def time_shift_1(data):
    data = data.reset_index().reset_index()
    data = data[['level_0','userid','actionType','shift_1']]
    data = data[~data['level_0'].isin([0])]
    return data[['userid', 'actionType', 'shift_1']]

import time
def time_conv(x):
    timeArray=time.localtime(x)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime

def day_trp(day):
    day = int(day)
    if day <= 10:
        return 1
    elif day >= 20:
        return 2
    else:
        return 3

###########################################################################################################
# main
action_train, userProfile_train, userComment_train, orderFuture_train, orderHistory_train = get_train_info()
action_test, userProfile_test, userComment_test, orderFuture_test, orderHistory_test = get_test_info()
# 用户个人信息 1
userProfile = code_category(userProfile_train,userProfile_test,['gender','province','age'])

# 需要预测的信息 userid  orderType 50383
orderFuture = pd.concat([orderFuture_train,orderFuture_test],axis=0)
orderFuture = orderFuture[['userid','orderType']]

# 用户行为 'userid', 'actionType', 'actionTime'
action = pd.concat([action_train,action_test],axis=0)
action = action.sort_values(by=['userid', 'actionTime'], ascending=True)

##############################################################################################################
# 第一次5 6发生时经历的时间
action = action.reset_index()

action_first_happend_9 = action[action['actionType'] == 6].groupby(['userid'],as_index=False).first()
action_first_use_happend = action.groupby(['userid'],as_index=False).first()
action_last_use_happend = action.groupby(['userid'],as_index=False).last()

action_first_happend_9 = pd.merge(action_first_use_happend,action_first_happend_9,on=['userid'],how='outer')
action_first_happend_9 = pd.merge(action_first_happend_9,action_last_use_happend,on=['userid'],how='outer')
action_first_happend_9['actionTime_y'] = action_first_happend_9['actionTime_y'].fillna(action_first_happend_9['actionTime'])
action_first_happend_9['actionType_y'] = action_first_happend_9['actionType_y'].fillna(action_first_happend_9['actionType'])
action_first_happend_9['index_y'] = action_first_happend_9['index_y'].fillna(action_first_happend_9['index'])
action_first_happend_9['first_finish_order'] = action_first_happend_9['actionTime_y'] - action_first_happend_9['actionTime_x']
action_first_happend_9['first_last_action_type'] = action_first_happend_9['actionType_y'] * 10 + action_first_happend_9['actionType_x']
action_first_happend_9['first_last_action_span'] = action_first_happend_9['index_y'] - action_first_happend_9['index_x']
action_first_happend_9['first_finish_order/first_last_action_span'] = action_first_happend_9['first_finish_order'] / action_first_happend_9['first_last_action_span']
action_first_happend_9 = action_first_happend_9[['userid','first_finish_order','first_last_action_type','first_last_action_span','first_finish_order/first_last_action_span']]
# print(action_first_happend_9)

action_first_happend_5 = action[action['actionType'] == 5].groupby(['userid'],as_index=False).first()
action_first_use_happend = action.groupby(['userid'],as_index=False).first()
action_last_use_happend = action.groupby(['userid'],as_index=False).last()

action_first_happend_5 = pd.merge(action_first_use_happend,action_first_happend_5,on=['userid'],how='outer')
action_first_happend_5 = pd.merge(action_first_happend_5,action_last_use_happend,on=['userid'],how='outer')
action_first_happend_5['actionTime_y'] = action_first_happend_5['actionTime_y'].fillna(action_first_happend_5['actionTime'])
action_first_happend_5['actionType_y'] = action_first_happend_5['actionType_y'].fillna(action_first_happend_5['actionType'])
action_first_happend_5['index_y'] = action_first_happend_5['index_y'].fillna(action_first_happend_5['index'])
action_first_happend_5['first_finish_order_5'] = action_first_happend_5['actionTime_y'] - action_first_happend_5['actionTime_x']
action_first_happend_5['first_last_action_type_5'] = action_first_happend_5['actionType_y'] * 10 + action_first_happend_5['actionType_x']
action_first_happend_5['first_last_action_span_5'] = action_first_happend_5['index_y'] - action_first_happend_5['index_x']
action_first_happend_5['first_finish_order_5/first_last_action_span_5'] = action_first_happend_5['first_finish_order_5'] / action_first_happend_5['first_last_action_span_5']
action_first_happend_5 = action_first_happend_5[['userid','first_finish_order_5','first_last_action_type_5','first_last_action_span_5','first_finish_order_5/first_last_action_span_5']]
# print(action_first_happend_9)

# exit()
#
# TODO 2018 01 29
# action analy

print('用户行为和日期的组合')
action_analy = action.copy()
action_analy['actionTime_ymd'] = pd.to_datetime(action_analy.actionTime.map(lambda x: time_conv(x)),format="%Y-%m-%d %H:%M:%S")
action_analy['o_day'] = action_analy['actionTime_ymd'].dt.day
action_analy['t_day'] = action_analy['o_day'].map(day_trp)

action_last_analy = action_analy.groupby(['userid'],as_index=False).last()
action_last_analy.rename(columns={'actionTime_ymd':'e_actionTime_ymd'},inplace=True)

###############################################窗口特征##############################################
# 做滑动窗口特征
# 1 时间衰减特征
print('时间衰减特征')
action_analy = pd.merge(action_analy,action_last_analy[['userid','e_actionTime_ymd']],on='userid')
action_analy['actionTime_diff'] = (action_analy['e_actionTime_ymd'] - action_analy['actionTime_ymd']).apply(lambda x:x.days + 1)
action_analy['weight_1'] = 1.0 / action_analy['actionTime_diff']
action_analy['weight_2'] = 1.0 / np.expm1(action_analy['actionTime_diff'])
action_analy['w_action_Type_1'] = action_analy['actionType'] * action_analy['weight_1']
action_analy['w_action_Type_2'] = action_analy['actionType'] * action_analy['weight_2']
# 特征拼接
action_analy_w_sum = action_analy.groupby(['userid'],as_index=False)[['w_action_Type_1','w_action_Type_2']].sum()

action_analy_one_hot = pd.get_dummies(action_analy['actionType'],prefix='actionType_windows')
del action_analy_one_hot['actionType_windows_9']
# del action_analy_one_hot['actionType_windows_8']
# del action_analy_one_hot['actionType_windows_7']
# del action_analy_one_hot['actionType_windows_4']
# del action_analy_one_hot['actionType_windows_3']
# del action_analy_one_hot['actionType_windows_2']
t_action_analy = pd.concat([action_analy[['userid','actionTime_diff']],action_analy_one_hot],axis=1)

print('时间滑动特征')
# 不同action发生的频次
for index,d in enumerate([15,7,5,3,2,1]):
    print('windows',d)
    tmp = t_action_analy[t_action_analy['actionTime_diff'] <= d]
    tmp = tmp.groupby(['userid'],as_index=False).sum()
    del tmp['actionTime_diff']

    tmp.rename( columns= {
                # 'actionType_windows_9': 'actionType_windows_9_%d' % (d),
                'actionType_windows_8': 'actionType_windows_8_%d' % (d),
                'actionType_windows_7': 'actionType_windows_7_%d' % (d),
                'actionType_windows_6': 'actionType_windows_6_%d' % (d),
                'actionType_windows_5': 'actionType_windows_5_%d' % (d),
                'actionType_windows_4': 'actionType_windows_4_%d' % (d),
                'actionType_windows_3': 'actionType_windows_3_%d' % (d),
                'actionType_windows_2': 'actionType_windows_2_%d' % (d),
                'actionType_windows_1': 'actionType_windows_1_%d' % (d)
                },inplace=True)

    if index == 0:
        action_windows_feat = tmp
    else:
        action_windows_feat = pd.merge(action_windows_feat,tmp,on=['userid'],how='outer')

# 用户历史订单
orderHistory = pd.concat([orderHistory_train, orderHistory_test],axis=0)
# #
# new_orderHistory = orderHistory.copy()
# new_orderHistory['orderType'] = new_orderHistory['orderType'].astype('str')
# new_orderHistory_ = new_orderHistory.groupby(['userid'],as_index=False).orderType.apply(lambda x:''.join(list(x))).reset_index()
# new_orderHistory_['userid'] = list(new_orderHistory['userid'].unique())
# new_orderHistory_.rename(columns={0:'ll'},inplace=True)
# del new_orderHistory_['index']
#
# new_orderHistory_['ll'] = '1000000000000' + new_orderHistory_['ll']
# new_orderHistory_['ll'] = new_orderHistory_['ll'].astype(np.float32)
# new_orderHistory_ = new_orderHistory_[['userid','ll']]
# new_orderHistory_ = pd.DataFrame(new_orderHistory_).drop_duplicates(['userid','ll'])

#####################################################################################
# 用户信息二次划分
userProfile_2 = pd.concat([userProfile_train,userProfile_test],axis=0)

def age_scale(age):
    age = str(age).split('后')[0]

    if age == 'nan':
        return 0
    else:
        age = int(age)
        if (age >60) & (age <=70):
            return 1
        elif (age >70) & (age <=80):
            return 2
        elif (age >80) & (age <=90):
            return 3
        elif (age >90) | (age == 0):
            return 4
        else:
            return 5

def province_scale(province):
    if (province == '上海') | (province == '北京') | (province == '广东') | (province == '天津') | (province == '海南'):
        return 1
    elif (province == '内蒙古') | (province == '青海') | (province == '西藏') | (province == '宁夏') | (province == '新疆') | (province == '甘肃'):
        return 2
    else:
        return 3

userProfile_2['age_scale'] = userProfile_2['age'].apply(age_scale)
userProfile_2['province_scale'] = userProfile_2['province'].apply(province_scale)
userProfile_2 = userProfile_2[['userid','age_scale','province_scale']]
province = pd.get_dummies(userProfile['province'],prefix='province')
userProfile_2 = pd.concat([userProfile_2,province],axis=1)

# 评论数据 需要处理tages 暂时使用userid  orderid  rating
userComment = pd.concat([userComment_train, userComment_test],axis=0)
userComment_info = userComment.copy()

print('用户评论的长度信息')
orderHistory_userComment_info = pd.merge(orderHistory[['userid','orderid','orderType']],userComment_info,on=['userid','orderid'],how='left')
orderHistory_userComment_info['tags'] = orderHistory_userComment_info['tags'].apply(lambda x:str(x).split('|'))
orderHistory_userComment_info['len_tags'] = orderHistory_userComment_info['tags'].apply(lambda x:len(x)) - 1
orderHistory_userComment_info['len_tags_rating'] = orderHistory_userComment_info['len_tags'] * 10 + orderHistory_userComment_info['rating']
info_comment_add = orderHistory_userComment_info[['userid','len_tags_rating','len_tags']].groupby(['userid'],as_index=False).sum()

print('用户评论的关键字信息')
def key_count(x):
    x = str(x).split(',')
    return len(x) - 1

userComment_info = userComment_info[['userid','commentsKeyWords']]
userComment_info['commentsKeyWords_len'] = userComment_info['commentsKeyWords'].apply(key_count)
info_comment_add = pd.merge(info_comment_add,userComment_info[['commentsKeyWords_len','userid']],on=['userid'],how='outer')

userComment = userComment[['userid','orderid','rating']]

print('历史订单的热度')
# 历史订单的热度
continent_static_feature = orderHistory.groupby(['continent'],as_index=False).orderid.agg({'countinent_sum':np.size})
country_static_feature = orderHistory.groupby(['country'],as_index=False).orderid.agg({'country_sum':np.size})
city_static_feature = orderHistory.groupby(['city'],as_index=False).orderid.agg({'city_sum':np.size})
orderHistory = pd.merge(orderHistory,continent_static_feature,on=['continent'],how='outer')
orderHistory = pd.merge(orderHistory,country_static_feature,on=['country'],how='outer')
orderHistory = pd.merge(orderHistory,city_static_feature,on=['city'],how='outer')
orderHistory_count_ = orderHistory[['userid','countinent_sum','country_sum','city_sum']]
orderHistory_count_ = pd.DataFrame(orderHistory_count_).drop_duplicates(['userid'])

city_ = pd.get_dummies(orderHistory['country'],prefix='country')
city_ = pd.concat([orderHistory['userid'],city_],axis=1)
city_ = city_.groupby(['userid'],as_index=False).sum()
orderHistory_count_ = pd.merge(orderHistory_count_,city_,on=['userid'],how='left')
orderHistory_count_ = pd.DataFrame(orderHistory_count_).drop_duplicates(['userid'])

print('复制数据')
data = orderFuture.copy()

# 历史订单的特征统计特征
orderHistory_one_hot = pd.get_dummies(orderHistory.orderType,prefix='orderType_h')
orderHistory_one_hot = pd.concat([orderHistory[['userid']],orderHistory_one_hot],axis=1)
orderHistory_one_hot = orderHistory_one_hot.groupby(['userid'],as_index=False).sum()
orderHistory_one_hot['orderType_h_sum'] = orderHistory_one_hot[['orderType_h_0','orderType_h_1']].sum(axis=1)
orderHistory_one_hot['orderType_h_1_ratdio'] = orderHistory_one_hot['orderType_h_0'] / orderHistory_one_hot['orderType_h_sum']
orderHistory_one_hot['orderType_h_0/1'] = (orderHistory_one_hot['orderType_h_0'] + 1) / (orderHistory_one_hot['orderType_h_1'] + 2)

orderHistory = code_category(orderHistory_train, orderHistory_test, ['city','country','continent'])
orderHistory_category2code = orderHistory[['userid','city','country','continent']]
orderHistory_category2code = orderHistory_category2code.groupby(['userid'],as_index=False).last()

# 用户对旅行的评论rating
print('用户旅行评论数据')
userComment = userComment.groupby(['userid'],as_index=False).rating.agg({'rating_mean':np.mean})

orderFuture = pd.merge(orderFuture,action,on=['userid'],how='outer')
user_id = list(orderFuture['userid'].unique())

# 行为类型一共有9个，其中1是唤醒app；2~4是浏览产品，无先后关系；5~9则是有先后关系的，从填写表单到提交订单再到最后支付。
# 序列长度  需要进一步处理
orderFuture['actionType'] = orderFuture['actionType'].astype(str)
action_len = orderFuture.groupby(['userid'],as_index=False).actionType.apply(action2len).reset_index()
action_len.rename(columns={'index':'userid',0:'actionlen'},inplace=True)
action_len['userid'] = user_id

def action_ser(action):
    return ''.join(list(action))

action_ser = orderFuture.groupby(['userid'],as_index=False).actionType.apply(action_ser).reset_index()
action_ser.rename(columns={'index':'userid',0:'action_ser'},inplace=True)
action_ser['userid'] = user_id
time_orderFuture = orderFuture.copy()


# 时间处理
orderHistory_last = orderHistory.groupby(['userid'],as_index=False).last()
orderHistory_last.rename(columns={'orderTime':'h_orderTime'},inplace=True)
#
time_orderFuture['actionTime_ymd'] = pd.to_datetime(time_orderFuture.actionTime.map(lambda x: time_conv(x)),format="%Y-%m-%d %H:%M:%S")
time_orderFuture['year'] = time_orderFuture['actionTime_ymd'].dt.year
time_orderFuture['month'] = time_orderFuture['actionTime_ymd'].dt.month
time_orderFuture['week'] = time_orderFuture['actionTime_ymd'].dt.week
time_orderFuture['weekday'] = time_orderFuture['actionTime_ymd'].dt.weekday + 1
time_orderFuture['date'] = time_orderFuture['actionTime_ymd'].dt.dayofyear
time_orderFuture['hour'] = time_orderFuture['actionTime_ymd'].dt.hour

def hour_map(hour):
    hour = int(hour)
    if (hour <=22) & (hour >= 18):
        return 1
    elif (hour <=17) & (hour >= 9):
        return 2
    else:
        return 3

time_orderFuture['hour'] = time_orderFuture['hour'].apply(hour_map)
# 最开始和最后一次发生行为的时间
time_last_orderFuture = time_orderFuture.groupby(['userid'],as_index=False).last()
time_first_orderFuture = time_orderFuture.groupby(['userid'],as_index=False).first()
time_first_orderFuture.rename(columns={'year':'f_year','month':'f_month','week':'f_week','weekday':'f_weekday','date':'f_date','hour':'f_hour'},inplace=True)

# history_now_order
history_now_order = pd.merge(time_last_orderFuture[['userid','actionTime']],orderHistory_last,on='userid',how='left')
history_now_order['last_order_diff_time'] = history_now_order['actionTime'] - history_now_order['h_orderTime']
history_now_order = history_now_order[['userid','last_order_diff_time']]


# time_orderFuture_weekday_one_hot = pd.get_dummies(time_orderFuture['weekday'],prefix='weekday_h')
# time_orderFuture_weekday_one_hot['weekday_h_sum'] = time_orderFuture_weekday_one_hot.sum(axis=1)
# time_orderFuture_weekday_one_hot = pd.concat([time_orderFuture['userid'],time_orderFuture_weekday_one_hot],axis=1)
# time_orderFuture_weekday_one_hot = time_orderFuture_weekday_one_hot.groupby(['userid'],as_index=False).sum()
#
# for i in time_orderFuture_weekday_one_hot.columns:
#     if (i == 'userid') | (i=='weekday_h_sum'):
#         pass
#     else:
#         time_orderFuture_weekday_one_hot['%s_ratdio'%(i)] = time_orderFuture_weekday_one_hot[i] / time_orderFuture_weekday_one_hot['weekday_h_sum']

################################################################################最后行为的差值################################################################################
# 最后两个行为的间隔差值
time_orderFuture['actionType'] = time_orderFuture['actionType'].astype(int)
action_Type_1 = time_orderFuture[time_orderFuture['actionType']==1].groupby('userid', as_index=False)['actionTime'].agg({'last_1_time': 'last'})
action_Type_5 = time_orderFuture[time_orderFuture['actionType']==5].groupby('userid', as_index=False)['actionTime'].agg({'last_5_time': 'last'})
action_Type_6 = time_orderFuture[time_orderFuture['actionType']==6].groupby('userid', as_index=False)['actionTime'].agg({'last_6_time': 'last'})
action_Type_7 = time_orderFuture[time_orderFuture['actionType']==7].groupby('userid', as_index=False)['actionTime'].agg({'last_7_time': 'last'})
action_Type_8 = time_orderFuture[time_orderFuture['actionType']==8].groupby('userid', as_index=False)['actionTime'].agg({'last_8_time': 'last'})
action_Type_4 = time_orderFuture[time_orderFuture['actionType']==4].groupby('userid', as_index=False)['actionTime'].agg({'last_4_time': 'last'})

action_Type_1_diff_5 = pd.merge(action_Type_1,action_Type_5,on=['userid'],how='left')
action_Type_5_diff_6 = pd.merge(action_Type_5,action_Type_6,on=['userid'],how='left')
action_Type_5_diff_7 = pd.merge(action_Type_5,action_Type_7,on=['userid'],how='left')
action_Type_6_diff_7 = pd.merge(action_Type_6,action_Type_7,on=['userid'],how='left')
action_Type_7_diff_8 = pd.merge(action_Type_7,action_Type_8,on=['userid'],how='left')

action_Type_4_diff_6 = pd.merge(action_Type_4,action_Type_6,on=['userid'],how='left')

action_Type_1_diff_5['last_1_5_diff'] = action_Type_1_diff_5['last_1_time'] - action_Type_1_diff_5['last_5_time']
action_Type_5_diff_6['last_5_6_diff'] = action_Type_5_diff_6['last_5_time'] - action_Type_5_diff_6['last_6_time']

action_Type_5_diff_7['last_5_7_diff'] = action_Type_5_diff_7['last_5_time'] - action_Type_5_diff_7['last_7_time']

action_Type_6_diff_7['last_6_7_diff'] = action_Type_6_diff_7['last_6_time'] - action_Type_6_diff_7['last_7_time']
action_Type_7_diff_8['last_7_8_diff'] = action_Type_7_diff_8['last_7_time'] - action_Type_7_diff_8['last_8_time']

action_Type_4_diff_6['last_4_6_diff'] = action_Type_4_diff_6['last_4_time'] - action_Type_4_diff_6['last_6_time']

del action_Type_4_diff_6['last_4_time']
del action_Type_4_diff_6['last_6_time']

del action_Type_5_diff_7['last_5_time']
del action_Type_5_diff_7['last_7_time']

del action_Type_5_diff_6['last_5_time']
# del action_Type_5_diff_6['last_6_time']

del action_Type_6_diff_7['last_6_time']
# del action_Type_6_diff_7['last_7_time']

del action_Type_7_diff_8['last_7_time']
# del action_Type_7_diff_8['last_8_time']


# 第一次行为的记录特征
# 第一次行为1和最后一次行为5
time_orderFuture['actionType'] = time_orderFuture['actionType'].astype(int)
action_Type_f_1 = time_orderFuture[time_orderFuture['actionType']==1].groupby('userid', as_index=False)['actionTime'].first()
action_Type_f_1.rename(columns={'actionTime':'first_action_1_time_'},inplace=True)
action_Type_f_4 = time_orderFuture[time_orderFuture['actionType']==4].groupby('userid', as_index=False)['actionTime'].first()
action_Type_f_4.rename(columns={'actionTime':'first_action_4_time_'},inplace=True)

action_Type_f_5 = time_orderFuture[time_orderFuture['actionType']==5].groupby('userid', as_index=False)['actionTime'].first()
action_Type_f_5.rename(columns={'actionTime':'first_action_5_time_'},inplace=True)
action_Type_f_6 = time_orderFuture[time_orderFuture['actionType']==6].groupby('userid', as_index=False)['actionTime'].first()
action_Type_f_6.rename(columns={'actionTime':'first_action_6_time_'},inplace=True)
action_Type_f_7 = time_orderFuture[time_orderFuture['actionType']==7].groupby('userid', as_index=False)['actionTime'].first()
action_Type_f_7.rename(columns={'actionTime':'first_action_7_time_'},inplace=True)
action_Type_f_8 = time_orderFuture[time_orderFuture['actionType']==8].groupby('userid', as_index=False)['actionTime'].first()
action_Type_f_8.rename(columns={'actionTime':'first_action_8_time_'},inplace=True)

action_Type_f_1_5 = pd.merge(action_Type_f_1,action_Type_f_5,on=['userid'],how='left')
action_Type_f_5_6 = pd.merge(action_Type_f_5,action_Type_f_6,on=['userid'],how='left')
action_Type_f_4_6 = pd.merge(action_Type_f_4,action_Type_f_6,on=['userid'],how='left')
action_Type_f_6_7 = pd.merge(action_Type_f_6,action_Type_f_7,on=['userid'],how='left')
action_Type_f_7_8 = pd.merge(action_Type_f_7,action_Type_f_8,on=['userid'],how='left')

action_Type_f_1_5['first_1_5_diff'] = - action_Type_f_1_5['first_action_1_time_'] + action_Type_f_1_5['first_action_5_time_']
action_Type_f_5_6['first_5_6_diff'] = - action_Type_f_5_6['first_action_5_time_'] + action_Type_f_5_6['first_action_6_time_']
action_Type_f_6_7['first_6_7_diff'] = - action_Type_f_6_7['first_action_6_time_'] + action_Type_f_6_7['first_action_7_time_']
action_Type_f_7_8['first_7_8_diff'] = - action_Type_f_7_8['first_action_7_time_'] + action_Type_f_7_8['first_action_8_time_']
action_Type_f_4_6['first_4_6_diff'] = - action_Type_f_4_6['first_action_4_time_'] + action_Type_f_4_6['first_action_6_time_']


del action_Type_f_4_6['first_action_4_time_']
del action_Type_f_4_6['first_action_6_time_']

del action_Type_f_5_6['first_action_5_time_']
# del action_Type_f_5_6['actionTime_y']

del action_Type_f_6_7['first_action_6_time_']
# del action_Type_f_6_7['actionTime_y']

del action_Type_f_7_8['first_action_7_time_']
# del action_Type_f_7_8['actionTime_y']


# 第一次5 最后一次6行为的差值
action_Type_f_1_1= pd.merge(action_Type_f_1,action_Type_1,on=['userid'],how='left')
action_Type_f_5_5= pd.merge(action_Type_f_5,action_Type_5,on=['userid'],how='left')
action_Type_f_6_6= pd.merge(action_Type_f_6,action_Type_6,on=['userid'],how='left')
action_Type_f_7_7= pd.merge(action_Type_f_7,action_Type_7,on=['userid'],how='left')
action_Type_f_8_8= pd.merge(action_Type_f_8,action_Type_8,on=['userid'],how='left')

action_Type_f_1_1['1_1_diff'] = action_Type_f_1_1['last_1_time'] - action_Type_f_1_1['first_action_1_time_']
action_Type_f_5_5['5_5_diff'] = action_Type_f_5_5['last_5_time'] - action_Type_f_5_5['first_action_5_time_']
action_Type_f_6_6['6_6_diff'] = action_Type_f_6_6['last_6_time'] - action_Type_f_6_6['first_action_6_time_']
action_Type_f_7_7['7_7_diff'] = action_Type_f_7_7['last_7_time'] - action_Type_f_7_7['first_action_7_time_']
action_Type_f_8_8['8_8_diff'] = action_Type_f_8_8['last_8_time'] - action_Type_f_8_8['first_action_8_time_']

del action_Type_f_1_1['last_1_time']
del action_Type_f_1_1['first_action_1_time_']

del action_Type_f_5_5['last_5_time']
del action_Type_f_5_5['first_action_5_time_']

del action_Type_f_6_6['last_6_time']
del action_Type_f_6_6['first_action_6_time_']

del action_Type_f_7_7['last_7_time']
del action_Type_f_7_7['first_action_7_time_']

del action_Type_f_8_8['last_8_time']
del action_Type_f_8_8['first_action_8_time_']

###########################################################################################################################################################
# 序列间隔特征 行为之间的间隔空隙
action_Type_span_1 = time_orderFuture[time_orderFuture['actionType'] == 1].reset_index().groupby(['userid'],as_index=False).index.agg({'sp_1':'last'})
action_Type_span_5 = time_orderFuture[time_orderFuture['actionType'] == 5].reset_index().groupby(['userid'],as_index=False).index.agg({'sp_5':'last'})
action_Type_span_6 = time_orderFuture[time_orderFuture['actionType'] == 6].reset_index().groupby(['userid'],as_index=False).index.agg({'sp_6':'last'})
action_Type_span_7 = time_orderFuture[time_orderFuture['actionType'] == 7].reset_index().groupby(['userid'],as_index=False).index.agg({'sp_7':'last'})
action_Type_span_8 = time_orderFuture[time_orderFuture['actionType'] == 8].reset_index().groupby(['userid'],as_index=False).index.agg({'sp_8':'last'})
action_Type_span_index = time_orderFuture.reset_index().groupby(['userid'],as_index=False).index.agg({'sp_index':'last'})

action_Type_span_index = pd.merge(action_Type_span_index,action_Type_span_1,on=['userid'],how='outer')
action_Type_span_index = pd.merge(action_Type_span_index,action_Type_span_5,on=['userid'],how='outer')
action_Type_span_index = pd.merge(action_Type_span_index,action_Type_span_6,on=['userid'],how='outer')
action_Type_span_index = pd.merge(action_Type_span_index,action_Type_span_7,on=['userid'],how='outer')
action_Type_span_index = pd.merge(action_Type_span_index,action_Type_span_8,on=['userid'],how='outer')
# action_Type_span_index = pd.merge(action_Type_span_index,action_Type_span_9,on=['userid'],how='outer')

action_Type_span_index['sp_index_1'] = action_Type_span_index['sp_index'] - action_Type_span_index['sp_1']
action_Type_span_index['sp_index_5'] = action_Type_span_index['sp_index'] - action_Type_span_index['sp_5']
action_Type_span_index['sp_index_6'] = action_Type_span_index['sp_index'] - action_Type_span_index['sp_6']
action_Type_span_index['sp_index_7'] = action_Type_span_index['sp_index'] - action_Type_span_index['sp_7']
action_Type_span_index['sp_index_8'] = action_Type_span_index['sp_index'] - action_Type_span_index['sp_8']
action_Type_span_index['sp_5_6'] = action_Type_span_index['sp_6'] - action_Type_span_index['sp_5']
#
action_Type_span_index.drop(['sp_1','sp_6','sp_5','sp_7','sp_8','sp_index'],axis=1,inplace=True)

# action_Type_span_index = action_Type_span_index.fillna(-999)

# 同行为特征的统计特征
action_Type_same_1 = time_orderFuture[time_orderFuture['actionType'] == 1].reset_index()
action_Type_same_1.rename(columns={'index':'action_index'},inplace=True)
add_nan = action_Type_same_1[['userid']].drop_duplicates()
add_nan['action_index'] = np.nan
action_Type_same_1 = pd.concat([action_Type_same_1, add_nan], ignore_index=True).sort_values(['userid', 'action_index'])
# 差分特征
action_Type_same_1['action_diff'] = action_Type_same_1['action_index'].diff()
action_Type_same_1 = action_Type_same_1.groupby('userid', as_index=False)['action_diff'].agg({
                                                                          'actionType_dis_between_1_mean':'mean',
                                                                          'actionType_dis_between_1_min':'min','actionType_dis_between_1_max':'max',
                                                                          'actionType_dis_between_1_var':'var'})

action_Type_same_5 = time_orderFuture[time_orderFuture['actionType'] == 5].reset_index()
action_Type_same_5.rename(columns={'index':'action_index'},inplace=True)
add_nan = action_Type_same_5[['userid']].drop_duplicates()
add_nan['action_index'] = np.nan
action_Type_same_5 = pd.concat([action_Type_same_5, add_nan], ignore_index=True).sort_values(['userid', 'action_index'])
# 差分特征
action_Type_same_5['action_diff'] = action_Type_same_5['action_index'].diff()
action_Type_same_5 = action_Type_same_5.groupby('userid', as_index=False)['action_diff'].agg({
                                                                          'actionType_dis_between_5_mean':'mean',
                                                                          'actionType_dis_between_5_min':'min','actionType_dis_between_5_max':'max',
                                                                          'actionType_dis_between_5_var':'var'})

action_Type_same_8 = time_orderFuture[time_orderFuture['actionType'] == 8].reset_index()
action_Type_same_8.rename(columns={'index':'action_index'},inplace=True)
add_nan = action_Type_same_8[['userid']].drop_duplicates()
add_nan['action_index'] = np.nan
action_Type_same_8 = pd.concat([action_Type_same_8, add_nan], ignore_index=True).sort_values(['userid', 'action_index'])
# 差分特征
action_Type_same_8['action_diff'] = action_Type_same_8['action_index'].diff()
action_Type_same_8 = action_Type_same_8.groupby('userid', as_index=False)['action_diff'].agg({
                                                                          'actionType_dis_between_8_mean':'mean',
                                                                          'actionType_dis_between_8_min':'min',
                                                                          'actionType_dis_between_8_max':'max',
                                                                          'actionType_dis_between_8_var':'var'
                                                                            })

action_Type_same_7 = time_orderFuture[time_orderFuture['actionType'] == 7].reset_index()
action_Type_same_7.rename(columns={'index':'action_index'},inplace=True)
add_nan = action_Type_same_7[['userid']].drop_duplicates()
add_nan['action_index'] = np.nan
action_Type_same_7 = pd.concat([action_Type_same_7, add_nan], ignore_index=True).sort_values(['userid', 'action_index'])
# 差分特征
action_Type_same_7['action_diff'] = action_Type_same_7['action_index'].diff()
action_Type_same_7 = action_Type_same_7.groupby('userid', as_index=False)['action_diff'].agg({
                                                                          'actionType_dis_between_7_mean':'mean',
                                                                          'actionType_dis_between_7_min':'min',
                                                                          'actionType_dis_between_7_max':'max',
                                                                          'actionType_dis_between_7_var':'var'})

action_Type_same_6 = time_orderFuture[time_orderFuture['actionType'] == 6].reset_index()
action_Type_same_6.rename(columns={'index':'action_index'},inplace=True)
add_nan = action_Type_same_6[['userid']].drop_duplicates()
add_nan['action_index'] = np.nan
action_Type_same_6 = pd.concat([action_Type_same_6, add_nan], ignore_index=True).sort_values(['userid', 'action_index'])
# 差分特征
action_Type_same_6['action_diff'] = action_Type_same_6['action_index'].diff()
action_Type_same_6 = action_Type_same_6.groupby('userid', as_index=False)['action_diff'].agg({
                                                                          'actionType_dis_between_6_mean':'mean',
                                                                          'actionType_dis_between_6_min':'min',
                                                                          'actionType_dis_between_6_max':'max',
                                                                          'actionType_dis_between_6_var':'var'})


# 时间偏移特征
del orderFuture['index']
orderFuture_shift = orderFuture.actionTime.shift(1).reset_index()
orderFuture_shift.rename(columns={'actionTime':'shift_actionTime'},inplace=True)
orderFuture_shift.fillna(0,inplace=True)
orderFuture_shift = pd.concat([orderFuture,orderFuture_shift['shift_actionTime']],axis=1)
orderFuture_shift['shift_1'] = - orderFuture_shift['shift_actionTime'] + orderFuture_shift['actionTime']
orderFuture_shift = orderFuture_shift.groupby(['userid'],as_index=False).apply(time_shift_1)
orderFuture_shift = orderFuture_shift[['userid','actionType','shift_1']]
orderFuture_shift = orderFuture_shift.groupby(['userid','actionType'],as_index=False).shift_1.mean()
orderFuture_shift = orderFuture_shift.set_index(['userid','actionType']).shift_1.unstack(level=-1).fillna(0)
orderFuture_shift = orderFuture_shift.reindex(orderFuture_shift.index.get_level_values(0))
orderFuture_shift = orderFuture_shift.reset_index()

####################################################################################
# 行为时间差特征
# del orderFuture['index']
#
# print(orderFuture)
# exit()


####################################################################################

orderFuture['actionType'] = orderFuture['actionType'].astype(int)
action_time_diff = orderFuture.groupby(['userid'],as_index=False).actionTime.apply(get_time_diff).reset_index()
action_time_diff['index'] = user_id
action_time_diff.rename(columns={'index':'userid',0:'action_time_diff'},inplace=True)

####################################################################################
# 用户发生各个行为的次数
action_Type_one_hot = pd.get_dummies(orderFuture['actionType'],prefix='actionType_c')
action_Type_one_hot['actionType_sum'] = action_Type_one_hot.sum(axis=1)
action_Type_one_hot['actionType_big_5'] = action_Type_one_hot[['actionType_c_5','actionType_c_6','actionType_c_7','actionType_c_8','actionType_c_9']].sum(axis=1)
action_Type_one_hot = pd.concat([orderFuture['userid'],action_Type_one_hot],axis=1)
action_Type_one_hot = action_Type_one_hot.groupby(['userid'],as_index=False).sum()

# 用户发生各个行为占比
for i in action_Type_one_hot.columns:
    if (i == 'userid') | (i == 'actionType_sum'):
        pass
    else:
        action_Type_one_hot['%s_ratio'%(i)] = action_Type_one_hot[i] / action_Type_one_hot['actionType_sum']

# 比例特征
action_Type_one_hot['actionType_c_1/actionType_c_5'] = (action_Type_one_hot['actionType_c_1'] + 1)/ (action_Type_one_hot['actionType_c_5'] + 2)
action_Type_one_hot['actionType_c_1/actionType_big_5'] = (action_Type_one_hot['actionType_c_1'] + 1 ) / (action_Type_one_hot['actionType_big_5'] + 2)

# 用户倒数x的行为
last_one_action = orderFuture.groupby(['userid'],as_index=False).nth(-1)
last_one_action.rename(columns={'actionTime':'actionTime_-1','actionType':'actionType_-1'},inplace=True)
last_two_action = orderFuture.groupby(['userid'],as_index=False).nth(-2)
last_two_action.rename(columns={'actionTime':'actionTime_-2','actionType':'actionType_-2'},inplace=True)
last_three_action = orderFuture.groupby(['userid'],as_index=False).nth(-3)
last_three_action.rename(columns={'actionTime':'actionTime_-3','actionType':'actionType_-3'},inplace=True)
last_four_action = orderFuture.groupby(['userid'],as_index=False).nth(-4)
last_four_action.rename(columns={'actionTime':'actionTime_-4','actionType':'actionType_-4'},inplace=True)
last_five_action = orderFuture.groupby(['userid'],as_index=False).nth(-5)
last_five_action.rename(columns={'actionTime':'actionTime_-5','actionType':'actionType_-5'},inplace=True)
last_six_action = orderFuture.groupby(['userid'],as_index=False).nth(-6)
last_six_action.rename(columns={'actionTime':'actionTime_-6','actionType':'actionType_-6'},inplace=True)
last_seven_action = orderFuture.groupby(['userid'],as_index=False).nth(-7)
last_seven_action.rename(columns={'actionTime':'actionTime_-7','actionType':'actionType_-7'},inplace=True)
# last_eight_action = orderFuture.groupby(['userid'],as_index=False).nth(-8)
# last_eight_action.rename(columns={'actionTime':'actionTime_-8','actionType':'actionType_-8'},inplace=True)
# 用户前数x的行为
first_zero_action = orderFuture.groupby(['userid'],as_index=False).nth(0)
first_zero_action.rename(columns={'actionTime':'actionTime_x_0','actionType':'actionType_x_0'},inplace=True)
first_one_action = orderFuture.groupby(['userid'],as_index=False).nth(1)
first_one_action.rename(columns={'actionTime':'actionTime_x_1','actionType':'actionType_x_1'},inplace=True)
first_two_action = orderFuture.groupby(['userid'],as_index=False).nth(2)
first_two_action.rename(columns={'actionTime':'actionTime_x_2','actionType':'actionType_x_2'},inplace=True)
first_three_action = orderFuture.groupby(['userid'],as_index=False).nth(3)
first_three_action.rename(columns={'actionTime':'actionTime_x_3','actionType':'actionType_x_3'},inplace=True)
# first_four_action = orderFuture.groupby(['userid'],as_index=False).nth(4)
# first_four_action.rename(columns={'actionTime':'actionTime_x_4','actionType':'actionType_x_4'},inplace=True)
# merge data

user_first_action = pd.merge(
    first_one_action[['userid','actionTime_x_1','actionType_x_1']],
    first_two_action[['userid','actionTime_x_2','actionType_x_2']],
    on=['userid'],
    how='outer'
)

user_first_action = pd.merge(
    user_first_action,
    first_three_action[['userid','actionTime_x_3','actionType_x_3']],
    on=['userid'],
    how='outer'
)

user_first_action = pd.merge(
    user_first_action,
    first_zero_action[['userid','actionTime_x_0','actionType_x_0']],
    on=['userid'],
    how='outer'
)

# user_first_action = pd.merge(
#     user_first_action,
#     first_four_action[['userid','actionTime_x_4','actionType_x_4']],
#     on=['userid'],
#     how='outer'
# )

user_last_action = pd.merge(
    last_one_action[['userid','actionTime_-1','actionType_-1']],
    last_two_action[['userid','actionTime_-2','actionType_-2']],
    on=['userid'],
    how='outer'
)

user_last_action = pd.merge(
    user_last_action,
    last_three_action[['userid','actionTime_-3','actionType_-3']],
    on=['userid'],
    how='outer'
)

user_last_action = pd.merge(
    user_last_action,
    last_four_action[['userid','actionTime_-4','actionType_-4']],
    on=['userid'],
    how='outer'
)

user_last_action = pd.merge(
    user_last_action,
    last_five_action[['userid','actionTime_-5','actionType_-5']],
    on=['userid'],
    how='outer'
)
user_last_action = pd.merge(
    user_last_action,
    last_six_action[['userid','actionTime_-6','actionType_-6']],
    on=['userid'],
    how='outer'
)

user_last_action = pd.merge(
    user_last_action,
    last_seven_action[['userid','actionTime_-7','actionType_-7']],
    on=['userid'],
    how='outer'
)

# user_last_action = pd.merge(
#     user_last_action,
#     last_eight_action[['userid','actionTime_-8','actionType_-8']],
#     on=['userid'],
#     how='outer'
# )

# 最后3行为的时间占比
user_last_action['-1_-2_time_diff'] = user_last_action['actionTime_-1'] - user_last_action['actionTime_-2'] + 0.0001
user_last_action['-2_-3_time_diff'] = user_last_action['actionTime_-2'] - user_last_action['actionTime_-3'] + 0.0001
user_last_action['-3_-4_time_diff'] = user_last_action['actionTime_-3'] - user_last_action['actionTime_-4'] + 0.0001
user_last_action['-4_-5_time_diff'] = user_last_action['actionTime_-4'] - user_last_action['actionTime_-5'] + 0.0001
user_last_action['-5_-6_time_diff'] = user_last_action['actionTime_-5'] - user_last_action['actionTime_-6'] + 0.0001
user_last_action['-6_-7_time_diff'] = user_last_action['actionTime_-6'] - user_last_action['actionTime_-7'] + 0.0001

user_last_action['-1_-7_time_diff'] = user_last_action['actionTime_-1'] - user_last_action['actionTime_-7'] + 0.0001

# user_last_action.drop(['actionTime_-3','actionTime_-2','actionTime_-1','actionTime_-4','actionTime_-5','actionTime_-6','actionTime_-7'],axis=1,inplace=True)
# user_last_action.drop(['actionTime_-1'],axis=1,inplace=True)


user_first_action['0_1_time_diff'] = user_first_action['actionTime_x_0'] - user_first_action['actionTime_x_1'] + 0.0001
user_first_action['1_2_time_diff'] = user_first_action['actionTime_x_1'] - user_first_action['actionTime_x_2'] + 0.0001
user_first_action['2_3_time_diff'] = user_first_action['actionTime_x_2'] - user_first_action['actionTime_x_3'] + 0.0001

# user_first_action['3_4_time_diff'] = user_first_action['actionTime_x_3'] - user_first_action['actionTime_x_4'] + 0.0001
# user_first_action['4_0_time_diff'] = user_first_action['actionTime_x_4'] - user_first_action['actionTime_x_0'] + 0.0001

# user_first_action.drop(['actionTime_0'],axis=1,inplace=True)
# user_first_action.drop(['actionTime_3','actionTime_2','actionTime_1','actionTime_0'],axis=1,inplace=True)

######################################## 合并数据 ########################################
data = pd.merge(data,action_time_diff,on=['userid'],how='outer')
data = pd.merge(data,action_Type_one_hot,on=['userid'],how='outer')
data = pd.merge(data,user_last_action,on=['userid'],how='outer')
data = pd.merge(data,user_first_action,on=['userid'],how='outer')
data = pd.merge(data,userProfile,on=['userid'],how='outer')
#
data = pd.merge(data,action_len,on=['userid'],how='outer')

# orderHistory
data = pd.merge(data,orderHistory_one_hot,on=['userid'],how='outer')

data = pd.merge(data,orderHistory_category2code,on=['userid'],how='outer')
# userComment
data = pd.merge(data,userComment,on=['userid'],how='outer')

######################################################################
# 最后几个行为的差值
data = pd.merge(data,action_Type_1_diff_5,on=['userid'],how='outer')
data = pd.merge(data,action_Type_5_diff_6,on=['userid'],how='outer')
data = pd.merge(data,action_Type_6_diff_7,on=['userid'],how='outer')
data = pd.merge(data,action_Type_5_diff_7,on=['userid'],how='outer')
data = pd.merge(data,action_Type_7_diff_8,on=['userid'],how='outer')
data = pd.merge(data,action_Type_4_diff_6,on=['userid'],how='outer')

# 第一次行为与最后一次行为的
data = pd.merge(data,action_Type_f_1_5,on=['userid'],how='outer')
data = pd.merge(data,action_Type_f_5_6,on=['userid'],how='outer')
data = pd.merge(data,action_Type_f_4_6,on=['userid'],how='outer')
data = pd.merge(data,action_Type_f_6_7,on=['userid'],how='outer')
data = pd.merge(data,action_Type_f_7_8,on=['userid'],how='outer')

data = pd.merge(data,action_Type_f_1_1,on=['userid'],how='outer')
data = pd.merge(data,action_Type_f_5_5,on=['userid'],how='outer')
data = pd.merge(data,action_Type_f_6_6,on=['userid'],how='outer')
data = pd.merge(data,action_Type_f_7_7,on=['userid'],how='outer')
data = pd.merge(data,action_Type_f_8_8,on=['userid'],how='outer')

######################################################################

# new feature
data = pd.merge(data,time_last_orderFuture[['userid','month','week','weekday','hour']],on=['userid'],how='outer')
data = pd.merge(data,time_first_orderFuture[['userid','f_month','f_week','f_weekday','f_hour']],on=['userid'],how='outer')
data['month_diff'] = data['f_month'] - data['month']
data['is_weekday'] = data['weekday'] == data['f_weekday']

data = pd.merge(data,history_now_order,on=['userid'],how='outer')
#
# data = pd.merge(data,time_orderFuture_weekday_one_hot,on=['userid'],how='outer')

# span
data = pd.merge(data,action_Type_span_index,on=['userid'],how='outer')
# 差分特征
data = pd.merge(data,action_Type_same_5,on=['userid'],how='outer')
data = pd.merge(data,action_Type_same_6,on=['userid'],how='outer')
data = pd.merge(data,action_Type_same_7,on=['userid'],how='outer')
data = pd.merge(data,action_Type_same_8,on=['userid'],how='outer')
data = pd.merge(data,action_Type_same_1,on=['userid'],how='outer')

# 取消注释最后
data = pd.merge(data,orderFuture_shift,on=['userid'],how='outer')

# TODO new feature
data = pd.merge(data,action_first_happend_9,on=['userid'],how='outer')
data = pd.merge(data,action_first_happend_5,on=['userid'],how='outer')


data = pd.merge(data,info_comment_add,on=['userid'],how='outer')
data = pd.merge(data,orderHistory_count_,on=['userid'],how='outer')

# 三级特征 属于原始数据跨区组合特征
data['-1_-2_time_diff/action_time_diff'] = data['-1_-2_time_diff'] / data['action_time_diff']
data['-2_-3_time_diff/action_time_diff'] = data['-2_-3_time_diff'] / data['action_time_diff']
data['-3_-4_time_diff/action_time_diff'] = data['-3_-4_time_diff'] / data['action_time_diff']
data['-4_-5_time_diff/action_time_diff'] = data['-4_-5_time_diff'] / data['action_time_diff']
data['-5_-6_time_diff/action_time_diff'] = data['-5_-6_time_diff'] / data['action_time_diff']
data['-6_-7_time_diff/action_time_diff'] = data['-6_-7_time_diff'] / data['action_time_diff']
# data['-7_-8_time_diff/action_time_diff'] = data['-7_-8_time_diff'] / data['action_time_diff']
data['-1_-7_time_diff/action_time_diff'] = data['-1_-7_time_diff'] / data['action_time_diff']
# first
# data['3_4_time_diff/action_time_diff'] = data['3_4_time_diff'] / data['action_time_diff']
data['2_3_time_diff/action_time_diff'] = data['2_3_time_diff'] / data['action_time_diff']
data['1_2_time_diff/action_time_diff'] = data['1_2_time_diff'] / data['action_time_diff']
data['0_1_time_diff/action_time_diff'] = data['0_1_time_diff'] / data['action_time_diff']
# 第一次行为和最后一次行为的比例
data['f_l_action_ratdio'] = data['0_1_time_diff/action_time_diff'] / data['-1_-2_time_diff/action_time_diff']

# 窗口特征
# 1 操作权重
data = pd.merge(data,action_analy_w_sum,on=['userid'],how='outer')
# 2 历史统计
data = pd.merge(data,action_windows_feat,on=['userid'],how='outer')

# 2次地区划分
data = pd.merge(data,userProfile_2,on=['userid'],how='outer')


data = pd.merge(data,action_ser,on=['userid'],how='outer')


# addddddd
# data = pd.merge(data,new_orderHistory_,on=['userid'],how='left')

data = pd.DataFrame(data).drop_duplicates(['userid'])
# 二次特征
data['last_5_6_diff_first_5_6_diff'] = data['last_5_6_diff'] - data['first_5_6_diff']
data['last_6_7_diff_first_6_7_diff'] = data['last_6_7_diff'] - data['first_6_7_diff']

#
# 获取train和test
test = data[data['userid'].isin(list(orderFuture_test['userid'].unique()))]
train = data[data['userid'].isin(list(orderFuture_train['userid'].unique()))]


# 用户独立
y_train = train['orderType']
y_test = test['orderType']

sub_id = test['userid']
del train['orderType']
del test['orderType']
del train['userid']
# del train['weekday_h_sum']
del test['userid']
# del test['weekday_h_sum']

cv = KFold(n_splits=4,shuffle=True,random_state=42)
results = []
feature_import = pd.DataFrame()
sub_array = []
feature_import['col'] = list(train.columns)

train = train.values
test = test.values
y_train = y_train.values

import xgboost as xgb

for traincv, testcv in cv.split(train,y_train):
    # dtrain = xgb.DMatrix(train[traincv], label=y_train[traincv])
    # dval = xgb.DMatrix(train[testcv], label=y_train[testcv])
    #
    # params = {
    #     'learning_rate': 0.01,
    #     'n_estimators': 1000,
    #     'max_depth': 10,
    #     'min_child_weight': 5,
    #     'gamma': 0,
    #     'colsample_bytree': 0.8,
    #     'eta': 0.05,
    #     'silent': 1,
    #     'objective': 'binary:logistic',
    #     'eval_metric': 'auc',
    #     'scale_pos_weight': 1,
    #     # 'nthread': 16,
    # }
    #
    # watchlist = [(dtrain, 'train'), (dval, 'eval')]
    # model = xgb.train(params, dtrain, 4000, watchlist,verbose_eval=200, early_stopping_rounds=100)
    # y_t = model.predict(xgb.DMatrix(train[testcv]))
    # results.append(roc_auc_score(y_train[testcv],y_t))
    # sub_array.append(model.predict(xgb.DMatrix(test)))

    lgb_train = lgb.Dataset(train[traincv], y_train[traincv])
    lgb_eval = lgb.Dataset(train[testcv], y_train[testcv], reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 10,
        'verbose': 0
    }

    # params = {
    #     'learning_rate': 0.05,
    #     'metric': 'auc',
    #     'num_leaves': 60,
    #     'num_trees': 490,
    #     'min_sum_hessian_in_leaf': 0.2,
    #     'min_data_in_leaf': 70,
    #     'bagging_fraction': 0.5,
    #     'feature_fraction': 0.3,
    #     'lambda_l1': 0,
    #     'lambda_l2': 11.88,
    #     'num_threads': 4,
    #     'scale_pos_weight': 1,
    #     'application': 'binary',
    # }

    gbm = lgb.train(params, lgb_train, num_boost_round=4000, valid_sets=[lgb_train, lgb_eval],verbose_eval = 200,early_stopping_rounds= 200)

    y_t = gbm.predict(train[testcv], num_iteration=gbm.best_iteration)
    results.append(roc_auc_score(y_train[testcv],y_t))

    feature_import[roc_auc_score(y_train[testcv],y_t)] = list(gbm.feature_importance())

    sub_array.append(gbm.predict(test,num_iteration=gbm.best_iteration))

print("Results: " + str( np.array(results).mean()))

feature_import.to_csv('./feature_imp.csv')

s = 0
for i in sub_array:
    s = s + i

s = s / 4

r = pd.DataFrame()
r['userid'] = list(sub_id.values)
r['orderType'] = s

r.to_csv('../result/result_20180207_2.csv' ,index=False)


# lgb
# Results: 0.804918010368 0.82459
# Results: 0.927354681314 0.93000
# Results: 0.937764080283 0.9400
# Results: 0.940130585824 0.94330
# Results: 0.950834664322 0.95270
# Results: 0.9570165787 0.95660
# Results: 0.958215101779 0.95929
# Results: 0.959025933974 0.96069
# Results: 0.960412952322 0.96149
# Results: 0.962307082184
# 


# xgb
# Results: 0.961907656795
# Results: 0.96229576385 0.96280
# Results: 0.963444211261 0.96509
# Results: 0.963189523051 0.96630

# 融合数据
# Results: 0.96352843761 + # Results: 0.963444211261 0.96619


#Results: 0.963949662592
# Results: 0.964077602354
# Results: 0.963703341494
# Results: 0.970178196036