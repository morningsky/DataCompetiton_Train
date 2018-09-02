# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 17:00:38 2018

@author: CHENXINYE
w我将得到的特征进行清洗去重，去除相关性强的，防止过拟合，剩下约100多个特征，然后再一次进行训练，这只是初步的特征工程处理方式，没做模型的选择和cv，生成lgbm_submission的提交文件，还未提交
"""
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.externals import joblib
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import gc
#import xgboost as xgb
gc.collect()

feat = joblib.load("feature.selective.m")
train_df = pd.read_csv('output/train_df.csv')
test_df = pd.read_csv('output/test_df.csv')


feat = feat.drop_duplicates()#去重
feat.reset_index(inplace=True, drop=True)

feat_columns = []
for i in feat:
    feat_columns.append(i)
        
corr_metric  = train_df[feat_columns].corr()

print('feature_treatment begin!')
#corr_metric.to_csv('corr.csv',index=1)

def Multiple_collinearity_treat(debug):
    duplicate = []
    for i in feat_columns:
        duplicate.append(i)
        for j in feat_columns:
            if corr_metric[i][j] > debug:
                if j not in duplicate:
                    feat_columns.remove(j)
    return feat_columns

def get_count(train,col):
    train = train.fillna(0)
    train["mean"] = train[col].apply(lambda x: x.mean(), axis=1)
    train["median"] = train[col].apply(lambda x: x.median(), axis=1)
    train["max"] = train[col].apply(lambda x: x.max(), axis=1)
    train["min"] = train[col].apply(lambda x: x.min(), axis=1)
    train["std"] = train[col].apply(lambda x: x.std(), axis=1)
    train["skew"] = train[col].apply(lambda x: x.skew(), axis=1)
    train["sum"] = train[col].apply(lambda x: x.sum(), axis=1)
    train["kurtosis"] = train[col].apply(lambda x: x.kurtosis(), axis=1)
    
    for i in col:
        train[i] = train[i].astype(float)
    
    return train


'''feat_columns is my selective feature'''
col = ['mean', 'median', 'max', 'min', 'std', 'skew', 'kurtosis', 'sum']

feat_columns = Multiple_collinearity_treat(0.85)
        
train_x,train_y = get_count(train_df,feat_columns),train_df.TARGET

train_x = train_x[feat_columns]

test = get_count(test_df,feat_columns)

test = test[feat_columns]

for i in col:    
    feat_columns.append(i)

gc.collect()

print('training begin!')

def kfold_lightgbm(x, y,test,num_folds=10,stratified = True,debug= False):
    # Divide in training/validation and test data
    x.reset_index(inplace=True, drop=True)
    y.reset_index(inplace=True, drop=True)

    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True)
        
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(x, y)):
        print(n_fold, (train_idx, valid_idx))
        train_x, train_y = x.iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = x.iloc[valid_idx], y.iloc[valid_idx]
        
        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=5,
            #is_unbalance=True,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=40,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=20,
            reg_alpha=0.04,
            reg_lambda=0.073,
            min_split_gain=0.0222415,
            min_child_weight=40,
            silent=-1,
            verbose=-1,
            n_jobs=6,
            )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 1000, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    return sub_preds

P = kfold_lightgbm(train_x, train_y,test,num_folds=5)

sb = pd.read_csv('output/submission.csv')
sb.TARGET = P[:,1]
sb.to_csv('output/lgbm_submission.csv',index=False)



"""
def RandomForest_training(train_x,train_y):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=50,
                                 n_estimator=5000,
                                 creiterion='gini',
                                 max_features=200,
                                 verbose=1,
                                 n_job=5,
                                 max_leaf_nodes=50,
                                 min_weight_fraction_leaf=0.1675,
                                 min_samples_leaf=80,)
    clf.fit(train_x,train_y)
    return clf

def RandomTreesEmbedding_training(train_x,train_y):
    from sklearn.ensemble import RandomTreesEmbedding
    clf = RandomTreesEmbedding(n_estimators=5000,
                     max_depth=20,
                     min_sample_split=2,
                     min_samples_leaf=1,
                     min_weight_fraction_leaf=0.0,
                     max_leaf_nodes=None,
                     min_impurity_decrease=0.0,
                     min_impurity_split=None,
                     sparse_output=True,
                     n_jobs=1, random_state=None, verbose=0, warm_start=False)
    clf.fit(train_x,train_y)
    return clf
    

def XGBClassifier_training(train_x,train_y,test_df,feat):
    dtrain = xgb.DMatrix(train_x, train_y,missing = 0)
    dtest = xgb.DMatrix(test_df[feat],missing = 0)
    
    dtrain.save_binary('train.buffer')
    dtest.save_binary('test.buffer')
    
    param = {'max_depth': 50, 'eta': 0.01, 'silent': 0, 'objective': 'binary:logistic','max_leaf_nodes':80}
    param['nthread'] = 6
    param['eval_metric'] = 'auc'
    param['booster']='gbtree'
    param['scoring']='roc_auc'
    param['min_child_weight'] = 0.65
    
    num_round = 1000
    bst = xgb.train(param, dtrain, num_round)
    ypred = bst.predict(dtest)
#    P = bst.predict_proba(dtest)
    return bst,ypred

clf = LGBMClassifier(
    nthread=4,
    #is_unbalance=True,
    n_estimators=10000,
    learning_rate=0.02,
    num_leaves=32,
    colsample_bytree=0.9497036,
    subsample=0.8715623,
    max_depth=8,
    reg_alpha=0.04,
    reg_lambda=0.073,
    min_split_gain=0.0222415,
    min_child_weight=40,
    silent=-1,
    verbose=-1,
    #scale_pos_weight=11
    )

clf = LGBMClassifier(
    nthread=4,
    #is_unbalance=True,
    n_estimators=10000,
    learning_rate=0.02,
    num_leaves=32,
    colsample_bytree=0.9497036,
    subsample=0.8715623,
    max_depth=8,
    reg_alpha=0.04,
    reg_lambda=0.073,
    min_split_gain=0.0222415,
    min_child_weight=40,
    silent=-1,
    verbose=-1,
    #scale_pos_weight=11
    )

rf1 = RandomForest_training(train_x,train_y)

rf2 = RandomTreesEmbedding_training(train_x,train_y)

P1 = rf1.predict_proba(test_df[feat])

P2 = rf1.predict_proba(test_df[feat])
"""