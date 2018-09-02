print('包加载...')
import gc
import re
import sys
import time
#import jieba
import string
import codecs
import pickle
import hashlib
import os.path
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost
import matplotlib
import matplotlib.pyplot as plt
import seaborn 
#from datetime import datetime
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder  
from itertools import groupby 

######################################## data read #####################################
print('数据清洗...')
data_path = 'C:/Users/wuzhengxiang/Desktop/皇包车比赛/'
import os
os.chdir(data_path)#设置当前工作空间
print (os.getcwd())#获得当前工作目录

orderFuture_train = pd.read_csv('orderFuture_train.csv')#0-33682  1-6625  all-40307
orderFuture_test  = pd.read_csv('orderFuture_test.csv')#10076 #6625/40307*10076=1656.127
orderFuture_test['orderType'] = -1
data = pd.concat([orderFuture_train,orderFuture_test])
data.columns = ['userid', 'label']
 
userProfile_train  = pd.read_csv('userProfile_train.csv')
userProfile_test   = pd.read_csv('userProfile_test.csv')
userProfile = pd.concat([userProfile_train,userProfile_test]).fillna('unknow')

action_train  = pd.read_csv('action_train.csv')
action_test   = pd.read_csv('action_test.csv')
action_train['time_rank'] = action_train['actionTime'].groupby(action_train['userid']).rank(ascending=False,method='first')
action_test['time_rank']  = action_test['actionTime'].groupby(action_test['userid']).rank(ascending=False,  method='first')
action =  pd.concat([action_train,action_test])

orderHistory_train  = pd.read_csv('orderHistory_train.csv')
orderHistory_test   = pd.read_csv('orderHistory_test.csv')
orderHistory =  pd.concat([orderHistory_train,orderHistory_test])

userComment_train  = pd.read_csv("userComment_train.csv")
userComment_test   = pd.read_csv("userComment_test.csv")
userComment =  pd.concat([userComment_train,userComment_test])


#######################################feature#######################################
################ 1、userProfile
class_le = LabelEncoder()  
userProfile['gender']   = class_le.fit_transform(userProfile['gender'].values)
userProfile['province'] = class_le.fit_transform(userProfile['province'].values)
userProfile['age']      = class_le.fit_transform(userProfile['age'].values)
data = data.merge(userProfile, on='userid', how='left')

#one-hot
feat = userProfile[['userid','gender','age']]
feat['gender'] = ['_{0}'.format(i) for i in feat['gender']]
feat['age']    = ['_{0}'.format(i)    for i in feat['age']]
feat = pd.get_dummies(feat)
data = data.merge(feat, on='userid', how='left')

#count
feat = pd.DataFrame(userProfile.groupby(['gender'])[['userid']].count()).reset_index()
feat.columns=["gender"]+["gender_cnt"]
data = data.merge(feat, on='gender', how='left')

feat = pd.DataFrame(userProfile.groupby(['age'])[['userid']].count()).reset_index()
feat.columns=["age"]+["age_cnt"]
data = data.merge(feat, on='age', how='left')

feat = pd.DataFrame(userProfile.groupby(['province'])[['userid']].count()).reset_index()
feat.columns=["province"]+["province_cnt"]
data = data.merge(feat, on='province', how='left')

#concat
data['age_province'] = data['age']*data['province']
data['age_gender'] = data['age']*data['gender']
data['province_gender'] = data['province']*data['gender']

#concat-count
feat = pd.DataFrame(data.groupby(['age_province'])[['userid']].count()).reset_index()
feat.columns=["age_province"]+["age_province_cnt"]
data = data.merge(feat, on='age_province', how='left')

feat = pd.DataFrame(data.groupby(['age_gender'])[['userid']].count()).reset_index()
feat.columns=["age_gender"]+["age_gender_cnt"]
data = data.merge(feat, on='age_gender', how='left')

feat = pd.DataFrame(data.groupby(['province_gender'])[['userid']].count()).reset_index()
feat.columns=["province_gender"]+["province_gender_cnt"]
data = data.merge(feat, on='province_gender', how='left')


################ 2、action
action['actionType'] = pow(2,action['actionType'])


#all actionType
feat = pd.DataFrame(action.groupby(['userid'])[['actionType']].std()).reset_index()
feat.columns=["userid"]+["actionType_std"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(action.groupby(['userid'])[['actionType']].mean()).reset_index()
feat.columns=["userid"]+["actionType_mean"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(action.groupby(['userid'])[['actionType']].skew()).reset_index()
feat.columns=["userid"]+["actionType_skew"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(action.groupby(['userid'])[['actionType']].max()).reset_index()
feat.columns=["userid"]+["actionType_max"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(action.groupby(['userid'])[['actionType']].min()).reset_index()
feat.columns=["userid"]+["actionType_min"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(action.groupby(['userid'])[['actionType']].sum()).reset_index()
feat.columns=["userid"]+["actionType_sum"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(action.groupby(['userid'])[['actionType']].median()).reset_index()
feat.columns=["userid"]+["actionType_median"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(action.groupby(['userid'])[['actionType']].cumprod()).reset_index()
feat.columns=["userid"]+["actionType_cumprod"]
data = data.merge(feat, on='userid', how='left')

data['actionType_jc'] = data['actionType_max']-data['actionType_min']
data['actionType_cv'] = data['actionType_std']/data['actionType_mean']

#time_rank actionType
feat=pd.DataFrame(pd.pivot_table(action[['userid','time_rank','actionType']], index='userid', columns='time_rank')).reset_index()
feat = feat.iloc[:,:26]
feat.columns=["userid"]+["actionType_rank_{0}".format(i) for i in range(1,26)]
data = data.merge(feat, on='userid', how='left')

data['actionType_rank_diff_1'] = data['actionType_rank_1']-data['actionType_rank_2']
data['actionType_rank_diff_2'] = data['actionType_rank_2']-data['actionType_rank_3']
data['actionType_rank_diff_3'] = data['actionType_rank_3']-data['actionType_rank_4']
data['actionType_rank_diff_4'] = data['actionType_rank_4']-data['actionType_rank_5']
data['actionType_rank_diff_5'] = data['actionType_rank_5']-data['actionType_rank_6']
data['actionType_rank_diff_6'] = data['actionType_rank_6']-data['actionType_rank_7']
data['actionType_rank_diff_7'] = data['actionType_rank_7']-data['actionType_rank_8']
data['actionType_rank_diff_8'] = data['actionType_rank_8']-data['actionType_rank_9']
data['actionType_rank_diff_9'] = data['actionType_rank_9']-data['actionType_rank_10']
data['actionType_rank_diff_10'] = data['actionType_rank_10']-data['actionType_rank_11']


name = ["actionType_rank_diff_{0}".format(i) for i in range(1,11)]
feat['actionType_rank_diff_sum']  = data[name].apply(lambda x: x.sum(),axis=1)
feat['actionType_rank_diff_std']  = data[name].apply(lambda x: x.std(),axis=1)
feat['actionType_rank_diff_mean'] = data[name].apply(lambda x: x.mean(),axis=1)
data = data.merge(feat, on='userid', how='left')

action['actionTime_hour']  = action['actionTime'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_hour)




#all actionTime
feat = pd.DataFrame(action.groupby(['userid'])[['actionTime']].std()).reset_index()
feat.columns=["userid"]+["actionTime_std"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(action.groupby(['userid'])[['actionTime']].mean()).reset_index()
feat.columns=["userid"]+["actionTime_mean"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(action.groupby(['userid'])[['actionTime']].skew()).reset_index()
feat.columns=["userid"]+["actionTime_skew"]
data = data.merge(feat, on='userid', how='left')


feat = pd.DataFrame(action.groupby(['userid'])[['actionTime_hour']].std()).reset_index()
feat.columns=["userid"]+["actionTime_hour_std"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(action.groupby(['userid'])[['actionTime_hour']].mean()).reset_index()
feat.columns=["userid"]+["actionTime_hour_mean"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(action.groupby(['userid'])[['actionTime_hour']].max()).reset_index()
feat.columns=["userid"]+["actionTime_hour_max"]
data = data.merge(feat, on='userid', how='left')


#all count
feat = pd.DataFrame(action.groupby(['userid'])[['actionTime']].count()).reset_index()
feat.columns=["userid"]+["action_cnt_all"]
data = data.merge(feat, on='userid', how='left')

#actionType count
feat = pd.DataFrame(pd.pivot_table(action,index=["userid"],columns='actionType',values=["actionTime"],aggfunc=len)).reset_index() 
feat.columns=["userid"]+["action_cnt_00{0}".format(i) for i in range(1,10)]
data = data.merge(feat, on='userid', how='left')

#actionType rate
data['action_001_rate'] = data['action_cnt_001'] /data['action_cnt_all']
data['action_002_rate'] = data['action_cnt_002'] /data['action_cnt_all']
data['action_003_rate'] = data['action_cnt_003'] /data['action_cnt_all']
data['action_004_rate'] = data['action_cnt_004'] /data['action_cnt_all']
data['action_005_rate'] = data['action_cnt_005'] /data['action_cnt_all']
data['action_006_rate'] = data['action_cnt_006'] /data['action_cnt_all']
data['action_007_rate'] = data['action_cnt_007'] /data['action_cnt_all']
data['action_008_rate'] = data['action_cnt_008'] /data['action_cnt_all']
data['action_009_rate'] = data['action_cnt_009'] /data['action_cnt_all']
data['action_1_4_rate'] = (data['action_cnt_001']+data['action_cnt_002']+
                           data['action_cnt_003']+data['action_cnt_004'])/data['action_cnt_all']
data['action_5_9_rate'] = (data['action_cnt_005']+data['action_cnt_006']+data['action_cnt_007']+
                           data['action_cnt_008']+data['action_cnt_009'])/data['action_cnt_all']

data['action_5_6_rate'] = data['action_005_rate'] +data['action_006_rate']
data['action_5_6_rara'] = data['action_005_rate'] /data['action_006_rate']

data['action_1_2_diff'] = data['action_cnt_001']-data['action_cnt_002']
data['action_2_3_diff'] = data['action_cnt_002']-data['action_cnt_003']
data['action_3_4_diff'] = data['action_cnt_003']-data['action_cnt_004']
data['action_4_5_diff'] = data['action_cnt_004']-data['action_cnt_005']
data['action_5_6_diff'] = data['action_cnt_005']-data['action_cnt_006']
data['action_6_7_diff'] = data['action_cnt_006']-data['action_cnt_007']
data['action_7_8_diff'] = data['action_cnt_007']-data['action_cnt_008']
data['action_8_9_diff'] = data['action_cnt_008']-data['action_cnt_009']
data['action_1_3_diff'] = data['action_cnt_001']-data['action_cnt_003']
data['action_5_9_diff'] = data['action_cnt_005']-data['action_cnt_009']


#max time all
feat = pd.DataFrame(action.groupby(['userid'])[['actionTime']].max()).reset_index()
feat.columns=["userid"]+["userid_actionTime_max"]
data = data.merge(feat, on='userid', how='left')

data['userid_actionTime_max_year']  = data['userid_actionTime_max'].apply(lambda x: time.localtime(int(x)).tm_year)
data['userid_actionTime_max_month'] = data['userid_actionTime_max'].apply(lambda x: time.localtime(int(x)).tm_mon)
data['userid_actionTime_max_hour']  = data['userid_actionTime_max'].apply(lambda x: time.localtime(int(x)).tm_hour)

#for i in data['userid_actionTime_max']: 
#    data['userid_actionTime_max_month'] = time.localtime(int(i)).tm_mon

#min time all
feat = pd.DataFrame(action.groupby(['userid'])[['actionTime']].max()).reset_index()
feat.columns=["userid"]+["userid_actionTime_min"]
data = data.merge(feat, on='userid', how='left')

data['userid_actionTime_last'] = data['userid_actionTime_max']-data['userid_actionTime_min']

#actionType max time
feat = pd.DataFrame(pd.pivot_table(action,index=["userid"],columns='actionType',values=["actionTime"],aggfunc=max)).reset_index() 
feat.columns=["userid"]+["actionTime_max_00{0}".format(i) for i in range(1,10)]
data = data.merge(feat, on='userid', how='left')

data['actionTime_max_005_year']  = data['actionTime_max_005'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_year)
data['actionTime_max_005_month'] = data['actionTime_max_005'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_mon)
data['actionTime_max_005_hour']  = data['actionTime_max_005'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_hour)

data['actionTime_max_006_year']  = data['actionTime_max_006'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_year)
data['actionTime_max_006_month'] = data['actionTime_max_006'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_mon)
data['actionTime_max_006_hour']  = data['actionTime_max_006'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_hour)

data['actionTime_max_007_year']  = data['actionTime_max_007'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_year)
data['actionTime_max_007_month'] = data['actionTime_max_007'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_mon)
data['actionTime_max_007_hour']  = data['actionTime_max_007'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_hour)


data['actionTime_max_diff_9_8'] = data['actionTime_max_009']-data['actionTime_max_008']
data['actionTime_max_diff_8_7'] = data['actionTime_max_008']-data['actionTime_max_007']
data['actionTime_max_diff_7_6'] = data['actionTime_max_007']-data['actionTime_max_006']
data['actionTime_max_diff_6_5'] = data['actionTime_max_006']-data['actionTime_max_005']
data['actionTime_max_diff_5_4'] = data['actionTime_max_005']-data['actionTime_max_004']
data['actionTime_max_diff_9_7'] = data['actionTime_max_009']-data['actionTime_max_007']
data['actionTime_max_diff_9_6'] = data['actionTime_max_009']-data['actionTime_max_006']
data['actionTime_max_diff_9_5'] = data['actionTime_max_009']-data['actionTime_max_005']
data['actionTime_max_diff_9_4'] = data['actionTime_max_009']-data['actionTime_max_004']

data['actionTime_max_diff_9_m'] = data['actionTime_max_009']-data['userid_actionTime_max']
data['actionTime_max_diff_8_m'] = data['actionTime_max_008']-data['userid_actionTime_max']
data['actionTime_max_diff_7_m'] = data['actionTime_max_007']-data['userid_actionTime_max']
data['actionTime_max_diff_6_m'] = data['actionTime_max_006']-data['userid_actionTime_max']
data['actionTime_max_diff_5_m'] = data['actionTime_max_005']-data['userid_actionTime_max']
data['actionTime_max_diff_4_m'] = data['actionTime_max_004']-data['userid_actionTime_max']
data['actionTime_max_diff_3_m'] = data['actionTime_max_003']-data['userid_actionTime_max']
data['actionTime_max_diff_2_m'] = data['actionTime_max_002']-data['userid_actionTime_max']
data['actionTime_max_diff_1_m'] = data['actionTime_max_001']-data['userid_actionTime_max']



#actionType min time
feat = pd.DataFrame(pd.pivot_table(action,index=["userid"],columns='actionType',values=["actionTime"],aggfunc=min)).reset_index() 
feat.columns=["userid"]+["actionTime_min_00{0}".format(i) for i in range(1,10)]
data = data.merge(feat, on='userid', how='left')

data['actionTime_1_last'] = data['actionTime_max_001']-data['actionTime_min_001']
data['actionTime_2_last'] = data['actionTime_max_002']-data['actionTime_min_002']
data['actionTime_3_last'] = data['actionTime_max_003']-data['actionTime_min_003']
data['actionTime_4_last'] = data['actionTime_max_004']-data['actionTime_min_004']
data['actionTime_5_last'] = data['actionTime_max_005']-data['actionTime_min_005']
data['actionTime_6_last'] = data['actionTime_max_006']-data['actionTime_min_006']
data['actionTime_7_last'] = data['actionTime_max_007']-data['actionTime_min_007']
data['actionTime_8_last'] = data['actionTime_max_008']-data['actionTime_min_008']
data['actionTime_9_last'] = data['actionTime_max_009']-data['actionTime_min_009']

#actionType std time
feat = pd.DataFrame(pd.pivot_table(action,index=["userid"],columns='actionType',values=["actionTime"],aggfunc=np.std)).reset_index() 
feat.columns=["userid"]+["actionTime_std_00{0}".format(i) for i in range(1,10)]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(pd.pivot_table(action,index=["userid"],columns='actionType',values=["actionTime"],aggfunc=np.mean)).reset_index() 
feat.columns=["userid"]+["actionTime_mean_00{0}".format(i) for i in range(1,10)]
data = data.merge(feat, on='userid', how='left')



#time_rank actionTime
feat=pd.DataFrame(pd.pivot_table(action[['userid','time_rank','actionTime']], index='userid', columns='time_rank')).reset_index()
feat = feat.iloc[:,:26]
feat.columns=["userid"]+["actionTime_rank_{0}".format(i) for i in range(1,26)]

name = ["actionTime_rank_{0}".format(i) for i in range(1,26)]
feat['actionTime_rank_sum']  = feat[name].apply(lambda x: x.sum(),axis=1)
feat['actionTime_rank_std']  = feat[name].apply(lambda x: x.std(),axis=1)
feat['actionTime_rank_mean'] = feat[name].apply(lambda x: x.mean(),axis=1)
data = data.merge(feat, on='userid', how='left')


feat = pd.DataFrame(data['userid'])
feat['actionTime_rank_diff_1'] = data['actionTime_rank_1']-data['actionTime_rank_2']
feat['actionTime_rank_diff_2'] = data['actionTime_rank_2']-data['actionTime_rank_3']
feat['actionTime_rank_diff_3'] = data['actionTime_rank_3']-data['actionTime_rank_4']
feat['actionTime_rank_diff_4'] = data['actionTime_rank_4']-data['actionTime_rank_5']
feat['actionTime_rank_diff_5'] = data['actionTime_rank_5']-data['actionTime_rank_6']
feat['actionTime_rank_diff_6'] = data['actionTime_rank_6']-data['actionTime_rank_7']
feat['actionTime_rank_diff_7'] = data['actionTime_rank_7']-data['actionTime_rank_8']
feat['actionTime_rank_diff_8'] = data['actionTime_rank_8']-data['actionTime_rank_9']
feat['actionTime_rank_diff_9'] = data['actionTime_rank_9']-data['actionTime_rank_10']
feat['actionTime_rank_diff_10'] = data['actionTime_rank_10']-data['actionTime_rank_11']

name = ["actionTime_rank_diff_{0}".format(i) for i in range(1,10)]
feat['actionTime_rank_diff_sum']  = feat[name].apply(lambda x: x.sum(),axis=1)
feat['actionTime_rank_diff_std']  = feat[name].apply(lambda x: x.std(),axis=1)
feat['actionTime_rank_diff_mean'] = feat[name].apply(lambda x: x.mean(),axis=1)
data = data.merge(feat, on='userid', how='left')

#dropname = ["actionTime_rank_{0}".format(i) for i in range(1,26)]
#data = data.drop(dropname,axis=1)




#######################3、orderHistory
orderHistory['city']      = class_le.fit_transform(orderHistory['city'].values)
orderHistory['country']   = class_le.fit_transform(orderHistory['country'].values)
orderHistory['continent'] = class_le.fit_transform(orderHistory['continent'].values)

feat = pd.DataFrame(orderHistory.groupby(['userid'])[['orderType']].mean()).reset_index()
feat.columns=["userid"]+["orderType_mean"]
data = data.merge(feat, on='userid', how='left')


feat = pd.DataFrame(orderHistory.groupby(['userid'])[['orderType']].max()).reset_index()
feat.columns=["userid"]+["orderType_max"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(orderHistory.groupby(['userid'])[['orderType']].sum()).reset_index()
feat.columns=["userid"]+["orderType_sum"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(orderHistory.groupby(['userid'])[['orderType']].std()).reset_index()
feat.columns=["userid"]+["orderType_std"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(orderHistory.groupby(['userid'])[['orderTime']].count()).reset_index()
feat.columns=["userid"]+["orderHistory_cnt_all"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(orderHistory.groupby(['userid'])[['orderTime']].sum()).reset_index()
feat.columns=["userid"]+["orderTime_sum"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(orderHistory.groupby(['userid'])[['orderTime']].max()).reset_index()
feat.columns=["userid"]+["orderTime_max"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(orderHistory.groupby(['userid'])[['orderTime']].min()).reset_index()
feat.columns=["userid"]+["orderTime_min"]
data = data.merge(feat, on='userid', how='left')


feat = pd.DataFrame(orderHistory.groupby(['userid'])[['city']].mean()).reset_index()
feat.columns=["userid"]+["city_mean"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(orderHistory.groupby(['userid'])[['city']].std()).reset_index()
feat.columns=["userid"]+["city_std"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(orderHistory.groupby(['userid'])[['country']].mean()).reset_index()
feat.columns=["userid"]+["country_mean"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(orderHistory.groupby(['userid'])[['country']].std()).reset_index()
feat.columns=["userid"]+["country_std"]
data = data.merge(feat, on='userid', how='left')


feat = pd.DataFrame(orderHistory.groupby(['userid'])[['continent']].mean()).reset_index()
feat.columns=["userid"]+["ocontinent_mean"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(orderHistory.groupby(['userid'])[['continent']].std()).reset_index()
feat.columns=["userid"]+["ocontinent_std"]
data = data.merge(feat, on='userid', how='left')


data['orderTime_last'] = data['orderTime_max']-data['orderTime_min']

data['orderTime_max_year']  = data['orderTime_max'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_year)
data['orderTime_max_month'] = data['orderTime_max'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_mon)
data['orderTime_max_hour']  = data['orderTime_max'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_hour)

feat = pd.DataFrame(pd.pivot_table(orderHistory,index=["userid"],columns='orderType',values=["orderTime"],aggfunc=len)).reset_index() 
feat.columns=["userid"]+["orderType_max_{0}".format(i) for i in range(0,2)]
data = data.merge(feat, on='userid', how='left')



################ 4、userComment
userComment['tags_long']  = userComment['tags'].fillna('a').apply(lambda x: len(x))
userComment['wrds_long']  = userComment['commentsKeyWords'].fillna('a').apply(lambda x: len(x))

feat = pd.DataFrame(userComment.groupby(['userid'])[['rating']].mean()).reset_index()
feat.columns=["userid"]+["rating_mean"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(userComment.groupby(['userid'])[['rating']].count()).reset_index()
feat.columns=["userid"]+["userComment_cnt_all"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(userComment.groupby(['userid'])[['tags_long']].mean()).reset_index()
feat.columns=["userid"]+["tags_long_mean"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(userComment.groupby(['userid'])[['wrds_long']].mean()).reset_index()
feat.columns=["userid"]+["wrds_long_mean"]
data = data.merge(feat, on='userid', how='left')

data['long_mean_all'] = data['tags_long_mean']+data['wrds_long_mean']


################ 组合特征
data['orderHistory_ctr'] = data['orderHistory_cnt_all']/data['action_cnt_all']
data['actiontime_orderTime_diff'] = data['userid_actionTime_max']-data['orderTime_max']
data['userComment_rate'] = data['userComment_cnt_all']/data['orderHistory_cnt_all']

data['zh_01'] = data['actionTime_max_diff_6_5']*data['actionTime_rank_diff_2']
data['zh_02'] = data['actionTime_max_diff_6_5']*data['actionTime_rank_diff_3']
data['zh_03'] = data['actionTime_max_diff_6_5']*data['actionTime_rank_diff_1']
data['zh_04'] = data['actionTime_max_diff_6_5']*data['userid']
data['zh_05'] = data['actionTime_max_diff_6_5']*data['actionTime_rank_diff_4']
data['zh_05'] = data['actionTime_max_diff_6_5']*data['actionTime_max_diff_1_m']




feat = pd.DataFrame(pd.pivot_table(orderHistory,index=["userid"],columns='continent',values=["orderTime"],aggfunc=len)).reset_index() 
feat.columns=["userid"]+["continent_cnt_{0}".format(i) for i in range(1,7)]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(pd.pivot_table(orderHistory,index=["userid"],columns='country',values=["orderTime"],aggfunc=len)).reset_index() 
feat.columns=["userid"]+["country_cnt_{0}".format(i) for i in range(1,52)]
data = data.merge(feat, on='userid', how='left')


orderHistory['orderTime_hour']  = orderHistory['orderTime'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_hour)

feat = pd.DataFrame(orderHistory.groupby(['userid'])[['orderTime_hour']].mean()).reset_index()
feat.columns=["userid"]+["orderTime_hour_mean"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(orderHistory.groupby(['userid'])[['orderTime_hour']].std()).reset_index()
feat.columns=["userid"]+["orderTime_hour_std"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(orderHistory.groupby(['userid'])[['orderTime_hour']].median()).reset_index()
feat.columns=["userid"]+["orderTime_hour_median"]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(pd.pivot_table(orderHistory,index=["userid"],columns='orderTime_hour',values=["orderTime"],aggfunc=len)).reset_index() 
feat.columns=["userid"]+["orderTime_hour_cnt_{0}".format(i) for i in range(1,25)]
data = data.merge(feat, on='userid', how='left')



feat = pd.DataFrame(pd.pivot_table(orderHistory,index=["userid"],columns='city',values=["orderTime"],aggfunc=len)).reset_index() 
feat.columns=["userid"]+["city_cnt_{0}".format(i) for i in range(1,219)]
#feat=feat[['userid','city_cnt_101','city_cnt_58','city_cnt_3','city_cnt_165','city_cnt_112','city_cnt_68','city_cnt_122','city_cnt_90',
#           'city_cnt_212','city_cnt_42','city_cnt_40','city_cnt_140','city_cnt_13','city_cnt_117']]
data = data.merge(feat, on='userid', how='left')



#time_rank actionTime
feat=pd.DataFrame(pd.pivot_table(action[['userid','time_rank','actionTime']], index='userid', columns='time_rank')).reset_index()
feat = feat.iloc[:,:300]
feat.columns=["userid"]+["actionTime_rank_{0}".format(i) for i in range(1,300)]
feat_diff = pd.DataFrame(feat.iloc[:,1:301].diff(axis=1))
feat_diff.drop('actionTime_rank_1',axis=1)

feat1=pd.DataFrame()
feat1['actionTime_rank_all_sum']  = feat_diff.apply(lambda x: x.sum(),axis=1)
feat1['actionTime_rank_all_std']  = feat_diff.apply(lambda x: x.std(),axis=1)
feat1['actionTime_rank_all_mean'] = feat_diff.apply(lambda x: x.mean(),axis=1)
feat1['actionTime_rank_all_mad'] = feat_diff.apply(lambda x: x.mad(),axis=1)
feat1['actionTime_rank_all_skew'] = feat_diff.apply(lambda x: x.skew(),axis=1)
feat1['actionTime_rank_all_min']  = feat_diff.apply(lambda x: x.min(),axis=1)
feat1['actionTime_rank_all_max']  = feat_diff.apply(lambda x: x.max(),axis=1)
feat1['actionTime_rank_all_var'] = feat_diff.apply(lambda x: x.var(),axis=1)
feat1['actionTime_rank_all_kurt'] = feat_diff.apply(lambda x: x.kurt(),axis=1)
feat1['userid'] = feat['userid']
data = data.merge(feat1, on='userid', how='left')






#######################################feature#######################################
################ 1、userProfile

################ 2、action





feat1=pd.DataFrame()
feat1['actionTime_rank_all_20_sum']  = feat_diff.iloc[:,1:20].apply(lambda x: x.sum(),axis=1)
feat1['actionTime_rank_all_20_std']  = feat_diff.iloc[:,1:20].apply(lambda x: x.std(),axis=1)
feat1['actionTime_rank_all_20_mean'] = feat_diff.iloc[:,1:20].apply(lambda x: x.mean(),axis=1)
feat1['actionTime_rank_all_20_mad'] = feat_diff.iloc[:,1:20].apply(lambda x: x.mad(),axis=1)
feat1['actionTime_rank_all_20_skew'] = feat_diff.iloc[:,1:20].apply(lambda x: x.skew(),axis=1)
feat1['actionTime_rank_all_20_min']  = feat_diff.iloc[:,1:20].apply(lambda x: x.min(),axis=1)
feat1['actionTime_rank_all_20_max']  = feat_diff.iloc[:,1:20].apply(lambda x: x.max(),axis=1)
feat1['actionTime_rank_all_20_var'] = feat_diff.iloc[:,1:20].apply(lambda x: x.var(),axis=1)
feat1['actionTime_rank_all_20_kurt'] = feat_diff.iloc[:,1:20].apply(lambda x: x.kurt(),axis=1)
feat1['userid'] = feat['userid']
data = data.merge(feat1, on='userid', how='left')


feat1=pd.DataFrame()
feat1['actionTime_rank_all_50_sum']  = feat_diff.iloc[:,1:50].apply(lambda x: x.sum(),axis=1)
feat1['actionTime_rank_all_50_std']  = feat_diff.iloc[:,1:50].apply(lambda x: x.std(),axis=1)
feat1['actionTime_rank_all_50_mean'] = feat_diff.iloc[:,1:50].apply(lambda x: x.mean(),axis=1)
feat1['actionTime_rank_all_50_mad'] = feat_diff.iloc[:,1:50].apply(lambda x: x.mad(),axis=1)
feat1['actionTime_rank_all_50_skew'] = feat_diff.iloc[:,1:50].apply(lambda x: x.skew(),axis=1)
feat1['actionTime_rank_all_50_min']  = feat_diff.iloc[:,1:50].apply(lambda x: x.min(),axis=1)
feat1['actionTime_rank_all_50_max']  = feat_diff.iloc[:,1:50].apply(lambda x: x.max(),axis=1)
feat1['actionTime_rank_all_50_var'] = feat_diff.iloc[:,1:50].apply(lambda x: x.var(),axis=1)
feat1['actionTime_rank_all_50_kurt'] = feat_diff.iloc[:,1:50].apply(lambda x: x.kurt(),axis=1)
feat1['userid'] = feat['userid']
data = data.merge(feat1, on='userid', how='left')





name =["actionTime_rank_{0}".format(i) for i in range(1,500)]
feat['actionTime_rank_all_sum']  = feat[name].apply(lambda x: x.sum(),axis=1)
feat['actionTime_rank_all_std']  = feat[name].apply(lambda x: x.std(),axis=1)
feat['actionTime_rank_all_mean'] = feat[name].apply(lambda x: x.mean(),axis=1)
feat['actionTime_rank_all_mad'] = feat[name].apply(lambda x: x.mad(),axis=1)
feat['actionTime_rank_all_skew'] = feat[name].apply(lambda x: x.skew(),axis=1)

feat = feat[['userid','actionTime_rank_all_sum','actionTime_rank_all_std','actionTime_rank_all_mean','actionTime_rank_all_mad','actionTime_rank_all_skew']]
data = data.merge(feat, on='userid', how='left')



feat = pd.DataFrame(pd.pivot_table(orderHistory,index=["userid"],columns='continent',values=["orderTime"],aggfunc=len)).reset_index() 
feat.columns=["userid"]+["continent_cnt_{0}".format(i) for i in range(1,7)]
data = data.merge(feat, on='userid', how='left')

feat = pd.DataFrame(pd.pivot_table(orderHistory,index=["userid"],columns='country',values=["orderTime"],aggfunc=len)).reset_index() 
feat.columns=["userid"]+["country_cnt_{0}".format(i) for i in range(1,52)]
data = data.merge(feat, on='userid', how='left')




feat = pd.DataFrame(data['userid'])
feat['actionTime_rank_diff_1'] = data['actionTime_rank_1']-data['actionTime_rank_2']
feat['actionTime_rank_diff_2'] = data['actionTime_rank_2']-data['actionTime_rank_3']
feat['actionTime_rank_diff_3'] = data['actionTime_rank_3']-data['actionTime_rank_4']
feat['actionTime_rank_diff_4'] = data['actionTime_rank_4']-data['actionTime_rank_5']
feat['actionTime_rank_diff_5'] = data['actionTime_rank_5']-data['actionTime_rank_6']
feat['actionTime_rank_diff_6'] = data['actionTime_rank_6']-data['actionTime_rank_7']
feat['actionTime_rank_diff_7'] = data['actionTime_rank_7']-data['actionTime_rank_8']
feat['actionTime_rank_diff_8'] = data['actionTime_rank_8']-data['actionTime_rank_9']
feat['actionTime_rank_diff_9'] = data['actionTime_rank_9']-data['actionTime_rank_10']
feat['actionTime_rank_diff_10'] = data['actionTime_rank_10']-data['actionTime_rank_11']

name = ["actionTime_rank_diff_{0}".format(i) for i in range(1,10)]
feat['actionTime_rank_diff_sum']  = feat[name].apply(lambda x: x.sum(),axis=1)
feat['actionTime_rank_diff_std']  = feat[name].apply(lambda x: x.std(),axis=1)
feat['actionTime_rank_diff_mean'] = feat[name].apply(lambda x: x.mean(),axis=1)
data = data.merge(feat, on='userid', how='left')


#######################3、orderHistory
#actionType count






#actionType max time
action['actionTime_mon']  = action['actionTime'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_mon)
feat = pd.DataFrame(pd.pivot_table(action,index=["userid"],columns='actionTime_mon',values=["actionTime"],aggfunc=len)).reset_index() 
feat.columns=["userid"]+["actionTime_mon_cnt_{0}".format(i) for i in range(1,13)]
data = data.merge(feat, on='userid', how='left')

action['actionTime_hour']  = action['actionTime'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_hour)
feat = pd.DataFrame(pd.pivot_table(action,index=["userid"],columns='actionTime_hour',values=["actionTime"],aggfunc=len)).reset_index() 
feat.columns=["userid"]+["actionTime_hour_cnt_{0}".format(i) for i in range(1,25)]
data = data.merge(feat, on='userid', how='left')

action['actionTime_wday']  = action['actionTime'].fillna(0).apply(lambda x: time.localtime(int(x)).tm_wday)
feat = pd.DataFrame(pd.pivot_table(action,index=["userid"],columns='actionTime_wday',values=["actionTime"],aggfunc=len)).reset_index() 
feat.columns=["userid"]+["actionTime_wday_cnt_{0}".format(i) for i in range(1,8)]
data = data.merge(feat, on='userid', how='left')





feat = pd.DataFrame(pd.pivot_table(orderHistory,index=["userid"],columns='city',values=["orderTime"],aggfunc=len)).reset_index() 
feat.columns=["userid"]+["city_cnt_{0}".format(i) for i in range(1,219)]
data = data.merge(feat, on='userid', how='left')

for i in range(len(nstr[0])-11):
    featlist+=[xd.loc[:,nstr[0][i]]-xd.loc[:,nstr[0][i+11]]]
    namelist+=['P_'+nstr[0][i]+'_'+nstr[0][i+11]] 

################ 4、userComment


################ 组合特征

data.columns
#################################### lgb ############################
train_feat = data[data['label']>= 0].fillna(-1)
testt_feat = data[data['label']<=-1].fillna(-1)
label_x  = train_feat['label']
label_y  = testt_feat['label']

userdf = testt_feat['userid']

train_feat = train_feat.drop('label',axis=1)
testt_feat = testt_feat.drop('label',axis=1)

#lgb算法
train = lgb.Dataset(train_feat, label=label_x)
test  = lgb.Dataset(testt_feat, label=label_y,reference=train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    #'metric': 'binary_logloss',
    'metric': 'auc',
    'min_child_weight': 3,
    'num_leaves': 2 ** 5,
    'lambda_l2': 10,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'learning_rate': 0.02,
    'tree_method': 'exact',
    'seed': 2017,
    'nthread': 12,
    'silent': True
    }


num_round = 5000
early_stopping_rounds = 100

model_cv = lgb.cv(params, 
                  train, 
                  num_round,
                  nfold=5,
                  verbose_eval=50,
                  early_stopping_rounds=early_stopping_rounds
                  )
max(model_cv['auc-mean'])


#cv_agg's auc: 0.965006 + 0.001475--0.9668
#0.96511 --0.9662
#0.9652196702239344

num_round = 5000
gbm = lgb.train(params, 
                  train, 
                  num_round, 
                  verbose_eval=50,
                  valid_sets=[train,test]
                  )

preds_sub = gbm.predict(testt_feat)
         
#help(lgb)

#结果保存
submission = pd.DataFrame({'userid':userdf,'orderType':preds_sub})
submission = submission[['userid','orderType']]
submission.to_csv('2018012802_lgb_pred.csv', index=False)


#特征重要性  
features = pd.DataFrame() 
features['features']   = gbm.feature_name() 
features['importance'] = gbm.feature_importance() 
features.sort_values(by=['importance'],ascending=False,inplace=True)        





################################## xgb ################################

train_feat = data[data['label']>= 0].fillna(-1)
testt_feat = data[data['label']<=-1].fillna(-1)
label_x  = train_feat['label']
label_y  = testt_feat['label']

userdf = testt_feat['userid']

train_feat = train_feat.drop('label',axis=1)
testt_feat = testt_feat.drop('label',axis=1)


xgb_train = xgboost.DMatrix(train_feat, label=label_x)
xgb_eval  = xgboost.DMatrix(testt_feat)

xgb_params = {
        "objective": "reg:logistic",
        "eval_metric": "auc",
        "eta": 0.01,
        "max_depth": 5,
        "min_child_weight": 3,
        #"gamma": 0.70
        "subsample": 0.85,
        "colsample_bytree": 0.65
        #"alpha": 2e-05
        # "lambda": 10
        }

cv_out = xgboost.cv(
         xgb_params, 
         xgb_train, 
         num_boost_round=10000, 
         early_stopping_rounds=50, 
         verbose_eval=50, 
         show_stdv=False,
         nfold = 5
         ) 

#test-auc:0.958814 --0.9602
#test-auc:0.962092 --0.9630
#test-auc:0.962748 --0.9639
#test-auc:0.963451 --0.9642
#test-auc:0.963449 --0.9645
#test-auc:0.963298 --0.9643
#test-auc:0.963649 --0.9648
#test-auc:0.963585 --0.96549
bst = xgboost.train(
        params=xgb_params,
        dtrain=xgb_train,
        verbose_eval=50,
        
        num_boost_round=5000
        )

preds_sub = bst.predict(xgb_eval)

#结果保存
submission = pd.DataFrame({'userid':userdf,'orderType':preds_sub})
submission = submission[['userid','orderType']]
submission.to_csv('2018012501_xgb_pred.csv', index=False)

#特征重要性评估
featureImportance = bst.get_fscore()
features = pd.DataFrame() 
features['features']   = featureImportance.keys() 
features['importance'] = featureImportance.values() 
features.sort_values(by=['importance'],ascending=False,inplace=True) 
fig,ax= plt.subplots() 
fig.set_size_inches(20,10) 
plt.xticks(rotation=60) 
seaborn.barplot(data=features.head(30),x='features',y='importance',ax=ax,orient='v')




import lightgbm as lgb  
import pandas as pd  
import numpy as np  
import pickle  
from sklearn.metrics import roc_auc_score  
from sklearn.model_selection import train_test_split 

print("Loading Data ... ")
train_x, train_y, test_x = load_data()  

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold


test['label'] = np.nan

print('构造特征...')
train_feat_temp = make_set(train,train)
test_feat = make_set(train,test)
sumbmission = test_feat[['id']].copy()

predictors = [f for f in test_feat.columns if f not in ['id','label','enddate','pred11','pred12','pred',
                                                        'hy_16.0', 'hy_91.0', 'hy_94.0']]

train_feat = train_feat_temp.append(train_feat_temp[train_feat_temp['prov']==11])
train_feat = train_feat.append(train_feat_temp[train_feat_temp['prov']==11])
print('开始CV 5折训练...')
scores = []
t0 = time.time()
mean_score = []
train_preds = np.zeros(len(train_feat))
test_preds11 = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
    lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index],   train_feat['label'].iloc[test_index])

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 150,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.95,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 100,
    }
    gbm = lgb.train(params, lgb_train, 900)
    train_preds_sub = gbm.predict(train_feat[predictors].iloc[test_index])
    train_preds[test_index] += train_preds_sub
    test_preds_sub = gbm.predict(test_feat[predictors])
    test_preds11 += test_preds_sub

    score = roc_auc_score(train_feat['label'].iloc[test_index],train_preds_sub)
    scores.append(score)
    print('第{0}轮mae的得分: {1}'.format(i + 1, score))
test_preds11 = test_preds11/5
print('auc平均得分: {}'.format(np.mean(scores)))
print('CV训练用时{}秒'.format(time.time() - t0))

print('开始CV 5折训练...')
train_feat = train_feat_temp.append(train_feat_temp[(train_feat_temp['prov']==12)])
train_feat = train_feat.append(train_feat_temp[(train_feat_temp['prov']==12)])
scores = []
t0 = time.time()
mean_score = []
train_preds = np.zeros(len(train_feat))
test_preds12 = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
    lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['label'].iloc[test_index])

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 150,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.95,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 100,
    }
    gbm = lgb.train(params, lgb_train, 900)
    train_preds_sub = gbm.predict(train_feat[predictors].iloc[test_index])
    train_preds[test_index] += train_preds_sub
    test_preds_sub = gbm.predict(test_feat[predictors])
    test_preds12 += test_preds_sub

    score = roc_auc_score(train_feat['label'].iloc[test_index],train_preds_sub)
    scores.append(score)
    print('第{0}轮mae的得分: {1}'.format(i + 1, score))
test_preds12 = test_preds12/5
print('auc平均得分: {}'.format(np.mean(scores)))
print('CV训练用时{}秒'.format(time.time() - t0))

test_feat['pred11'] = test_preds11
test_feat['pred12'] = test_preds12
test_feat['pred'] = test_feat.apply(lambda x: x.pred11 if x.prov==11 else x.pred12, axis=1)
preds_scatter = get_threshold(test_feat['pred'].values)
submission = pd.DataFrame({'EID':sumbmission['id'],'FORTARGET':preds_scatter,'PROB':test_feat['pred'].values})
submission.to_csv('../output/piupiu_lgb_pred.csv', index=False)






################################## xgb重采样 ################################
import xgboost
train_feat = train_feat_temp.append(train_feat_temp[train_feat_temp['prov']==11])
train_feat = train_feat.append(train_feat_temp[train_feat_temp['prov']==11])

print('开始CV 5折训练...')
scores = []
t0 = time.time()
test_preds11 = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    xgb_train = xgboost.DMatrix(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
    xgb_eval = xgboost.DMatrix(test_feat[predictors])

    xgb_params = {
        "objective": "reg:logistic"
        , "eval_metric": "auc"
        , "eta": 0.01
        , "max_depth": 12
        , "min_child_weight": 10
        , "gamma": 0.70
        , "subsample": 0.76
        , "colsample_bytree": 0.95
        , "alpha": 2e-05
        , "lambda": 10
    }
    bst = xgboost.train(params=xgb_params,dtrain=xgb_train,num_boost_round=1200)
    test_preds_sub = bst.predict(xgb_eval)
    test_preds11 += test_preds_sub

test_preds11 = test_preds11/5
print('CV训练用时{}秒'.format(time.time() - t0))

print('开始CV 5折训练...')
train_feat = train_feat_temp.append(train_feat_temp[(train_feat_temp['prov']==12)])
train_feat = train_feat.append(train_feat_temp[(train_feat_temp['prov']==12)])
t0 = time.time()
test_preds12 = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    xgb_train = xgboost.DMatrix(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
    xgb_eval = xgboost.DMatrix(test_feat[predictors])

    xgb_params = {
        "objective": "reg:logistic"
        , "eval_metric": "auc"
        , "eta": 0.01
        , "max_depth": 12
        , "min_child_weight": 10
        , "gamma": 0.70
        , "subsample": 0.76
        , "colsample_bytree": 0.95
        , "alpha": 2e-05
        , "lambda": 10
    }
    bst = xgboost.train(params=xgb_params, dtrain=xgb_train, num_boost_round=1200)
    test_preds_sub = bst.predict(xgb_eval)
    test_preds12 += test_preds_sub

test_preds12 = test_preds12/5
print('CV训练用时{}秒'.format(time.time() - t0))

test_feat['pred11'] = test_preds11
test_feat['pred12'] = test_preds12
test_feat['pred'] = test_feat.apply(lambda x: x.pred11 if x.prov==11 else x.pred12, axis=1)
preds_scatter = get_threshold(test_feat['pred'].values)
submission = pd.DataFrame({'EID':sumbmission['id'],'FORTARGET':preds_scatter,'PROB':test_feat['pred'].values})
submission.to_csv('../output/piupiu_xgb_pred.csv', index=False)









