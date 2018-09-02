# -*- coding: utf-8 -*-
"""
Created on Fri Jan 05 20:51:07 2018

@author: sky
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

data_train = pd.read_excel('../data/train.xlsx')
data_testA = pd.read_excel('../data/testA.xlsx')
data_testA_y = pd.read_csv('../data/testA_y.csv',names=['ID','Y'])
data_testA = pd.merge(data_testA,data_testA_y)
data_test = pd.read_excel('../data/testB.xlsx')
data_train = pd.concat([data_train,data_testA],axis=0,ignore_index=True)
train_test = pd.concat([data_train,data_test],axis=0,ignore_index=True)
def func(x):
    try:
        return float(x)
    except:
        if x == 'NaN':
            return 0
        else:
            return x
    
y = train_test["Y"][:600]
train_test.applymap(func)
train_test.fillna(0,inplace=True)


fea_columns = list(train_test.columns.values)
fea_columns.remove("ID")
fea_columns.remove("Y")
data = train_test[fea_columns] #直接用列名索引比将需要的列concat效率高百倍！！！
cate_columns = data.select_dtypes(include=['object']).columns
num_columns = data.select_dtypes(exclude=['object']).columns
print("cate feat num: ",len(cate_columns),"num feat num: ", len(num_columns))
fea_cate = data[cate_columns]
fea_num = data[num_columns]
fea_cate_dummies = pd.get_dummies(fea_cate) #类别型特征热编码
ss = StandardScaler()
fea_num_scale = pd.DataFrame(ss.fit_transform(fea_num)) #数值型特征标准化
fea_all = pd.concat([fea_num_scale, fea_cate_dummies],axis=1)

'''                             
import keras
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from keras.layers import Dense,Activation,Input
from keras.models import Sequential,Model
# 数据切分
x_train, x_test, y_train, y_test = train_test_split(fea_all, fea_all, test_size=0.2, random_state=42)
# 自编码维度
encoding_dim = 100  
# 输入层
input_ = Input(shape=(8051,))
# 编码层
encoded = Dense(encoding_dim, activation='relu')(input_)
# 解码层
decoded = Dense(8051, activation='sigmoid')(encoded)
# 自编码器模型
autoencoder = Model(input=input_, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, 
                x_train,
                nb_epoch=500,
                batch_size=10,
                shuffle=True,
                validation_data=(x_test, x_test))
# 根据上面我们训练的自编码器，截取其中编码部分定义为编码模型
encoder = Model(input=input_, output=encoded)
# 对特征进行编码降维
feat_dim_100 = encoder.predict(fea_all)

'''
X_train,X_val,y_train,y_val = train_test_split(fea_all[:600],y,test_size=0.2, random_state=42)
from sklearn.svm import SVR
#model = SVR(verbose=True)
model = xgb.XGBRegressor()
model.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error
y_val_pred = model.predict(X_val)
print 'MSE:',mean_squared_error(y_val,y_val_pred)
'''
gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        learning_rate=0.01,
                        n_estimators=40)
gbm.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='l1',
        early_stopping_rounds=5,
        verbose= 10)

print('Start predicting...')
# predict
y_pred = gbm.predict(X_val)
# eval
print 'The rmse of prediction is:', mean_squared_error(y_val, y_pred)

estimator = lgb.LGBMRegressor(boosting_type='gbdt',seed=42)

from sklearn.model_selection import GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40,60]
}
model = GridSearchCV(estimator,param_grid,verbose=10)
model.fit(X_train,y_train)
print('Best parameters found by grid search are:', model.best_params_)

from sklearn.metrics import mean_squared_error
y_val_pred = model.predict(X_val)
print 'MSE:',mean_squared_error(y_val,y_val_pred)

y_result = model.predict(fea_all[500:])
submission = pd.DataFrame()
submission["ID"] = data_test["ID"]
submission["Y"] = y_result
#submission.to_csv('../output/submission_lgb.csv',index=False,header=False)

'''




