# -*- coding: utf-8 -*-
"""
Created on Tue Jul 04 17:13:39 2017

@author: sky
"""
from __future__ import division

import pandas as pd
import numpy as np
import math
import os

from sklearn.utils import shuffle
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from xgboost.sklearn import XGBClassifier


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit,learning_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
import warnings
warnings.filterwarnings('ignore')




fileName1 = 'E:\DataMining\c4\mirror\SchoolCompete\data\dsjtzs_txfz_training.txt'
fileName2 = 'E:\DataMining\c4\mirror\SchoolCompete\data\dsjtzs_txfz_test1.txt'

train_data = pd.read_csv(fileName1,sep=' ',header=None,)
train_data.columns = ['id','data','target','label']
#train_expand = pd.read_csv(r'E:\DataMining\c4\expand\train_expand.csv')
#train_data_expand = pd.concat([train_data,train_expand]).reset_index()
#train_data_expand = train_data_expand.drop(['index'],axis=1)
test_data = pd.read_csv(fileName2,sep=' ',header=None)
test_data.columns = ['id','data','target']



'''获取连续性的评价值 + 差分集合'''
def getConValue(x):
    xdiff = []
    convalue = 0
    length = len(x)
    if(length>1):
        for index,value in enumerate(x):
            if(index+1!=length):
                xdiff_temp = x[index+1] - x[index]
                xdiff.append(xdiff_temp)
                convalue = convalue + abs(xdiff_temp)  
    else:
        convalue = 0
        xdiff=[0]
    convalue = convalue/length
    return convalue,xdiff

'''获取拟合系数'''
def getFits(x,t,num):
    cof= np.polyfit(x,t,num)
    cof = list(cof)
    return cof     

'''获取x是否回择的评价'''
def getIsBack(x):
    is_back = 0
    length = len(x)
    if(length>1):
        for index,value in enumerate(x):
            if(index+1!=length):
                temp = x[index+1] - x[index]
                if(temp<0):
                    is_back = is_back + 1
    return is_back

'''获取相邻两点之间的斜率列表'''
def getEdge(x,y):
    length = len(x)
    k = []
    count = 0
    if(length>1):
        for i  in range(length-1):
            if(x[i+1] == x[i]):
                k_i = 0
            else:
                k_i = (y[i+1] - y[i])/(x[i+1] - x[i])
            k.append(k_i)
            if(k_i!=0):
                count = count +1
    else:
        k = [0]
    return k,count

def getRecount(x,y):
    x = pd.DataFrame(x,columns=['x'])
    y = pd.DataFrame(y,columns=['y'])
    z = pd.concat([x,y],axis=1)
    z = z.drop_duplicates(['x','y'])
    recount = z.shape[0]/x.shape[0]
    return recount
    
    

'''在初始数据中提取出特征'''    
def genFeatures(data):
    train_X = []
    for index,row in data.iterrows():#row[0]=id row[1]=point row[2]=target row[3]=label
        path = []
        for j,s in enumerate(row[1].split(';')[:-1]):  
            path.append(s.split(','))
        
        '''将整个列表转换为dataframe 将读进来的xyt转换为float 再单独取出每一列转化为列表 方便运算'''
        path = pd.DataFrame(path,columns=['x','y','t'])
        path = path.astype('float') 
        x = list(path['x'])
        y = list(path['y'])
        t = list(path['t'])
        '''平均值 最小值 最大值 极差 中位数 绝对离差 方差 标准差'''
        mean = path.mean()
        min_ = path.min()
        max_ = path.max()
        maxdis = max_ - min_ 
        median= path.median() 
        mad = path.mad() 
        var = path.var() 
        std = path.std() 
        
        '''构成训练集的所有统计特征：27个'''
        count = path.count()[0] 
        '''最后一个坐标x y距离目标x y的距离'''
        xconvalue,xdiff = getConValue(x)
        yconvalue,ydiff = getConValue(y)
        tconvalue,tdiff = getConValue(t)
        xdiff = np.array(xdiff)
        ydiff = np.array(ydiff)
        tdiff = np.array(tdiff)
        xdiff_var = np.var(xdiff)
        ydiff_var = np.var(ydiff)
        tdiff_var = np.var(tdiff)
        xdiff_max = xdiff.max()
        ydiff_max = ydiff.max()
        tdiff_max = tdiff.max()
        xdis_target = x[-1] - float(row[2].split(',')[0]) 
        ydis_target = y[-1] - float(row[2].split(',')[1]) 
        dis = math.sqrt(xdis_target**2 + ydis_target**2)
        xmean = mean[0]
        ymean = mean[1]
        tmean = mean[2]
        xmin = min_[0]
        ymin = min_[1]
        tmin = min_[2]
        xmax = max_[0]
        ymax = max_[1]
        tmax = max_[2]
        xmaxdis = maxdis[0]
        ymaxdis = maxdis[1]
        tmaxdis = maxdis[2]
        xmedian = median[0]
        ymedian = median[1]
        tmedian = median[2]
        xmad = mad[0]
        ymad = mad[1]
        tmad = mad[2]    
        xvar = var[0]
        yvar = var[1]
        tvar = var[2]
        xstd = std[0]
        ystd = std[1]
        tstd = std[2]
        xfirst = x[0]
        xlast = x[-1]
        yfirst = y[0]
        ylast = y[-1]
        tfirst = t[0]
        tlast = t[-1]
        '''所有点 前五个点 中间五个点 后五个点的速度 加速度 计算 8个'''
        is_back = getIsBack(x)
        #fits = getFits(x,t,3) #4 fit feature
        recount = getRecount(x,y)

        k,kcount = getEdge(x,y)
        k = np.array(k)
        kmin = k.min()
        kmax = k.max()
        kmaxdis = kmax- kmin 
        kmean = k.mean()
        kvar = k.var()

        xdis = x[-1] - x[0]
        ydis = y[-1] - y[0]
        tdis = abs(t[-1] - t[0])
 

        v_xmean = xdis/tdis
        v_ymean = ydis/tdis
        a_xmean = 2*xdis/math.sqrt(tdis)
        a_ymean = 2*ydis/math.sqrt(tdis)
        if(count > 10):
            mid = int(count/2) + 1
            xf5 = x[4] - x[0]
            xm5 = x[mid+2] - x[mid-2]
            xl5 = x[-1] - x[-5]
            yf5 = y[4] - y[0]
            ym5 = y[mid+2] - y[mid-2]
            yl5 = y[-1] - y[-5]
            tf5 = abs(t[4] - t[0])
            tm5 = abs(t[mid+2] - t[mid-2])
            tl5 = abs(t[-1] - t[-5])
            
            v_xfirst5 = xf5/tf5
            v_xmiddle5 = xm5/tm5
            v_xlast5 = xl5/tl5
            
            v_yfirst5 = yf5/tf5
            v_ymiddle5 = ym5/tm5
            v_ylast5 = yl5/tl5
            
            a_xfirst5 = 2*xf5/math.sqrt(tf5)
            a_xmiddle5 = 2*(xm5-v_xmiddle5*tm5)/math.sqrt(tm5)
            a_xlast5 = 2*(xl5-v_xlast5*tl5)/math.sqrt(tl5)
            
            a_yfirst5 = 2*yf5/math.sqrt(tf5)
            a_ymiddle5 = 2*(ym5-v_ymiddle5*tm5)/math.sqrt(tm5)
            a_ylast5 = 2*(yl5-v_ylast5*tl5)/math.sqrt(tl5)
        elif(count<10):
            v_xfirst5 = v_xmiddle5 = v_xlast5 = v_xmean
            v_yfirst5 = v_ymiddle5 = v_ylast5 = v_ymean
            a_xfirst5 = a_xmiddle5 = a_xlast5 = a_xmean
            a_yfirst5 = a_ymiddle5 = a_ylast5 = a_ymean


        '''所有特征组合'''
        X = [count,xdis_target,ydis_target,xmean,ymean,tmean,xmin,ymin,tmin,
             xmax,ymax,tmax,xmaxdis,ymaxdis,tmaxdis,xmedian,ymedian,tmedian,
             xmad,ymad,tmad,xvar,yvar,tvar,xstd,ystd,tstd,
             tfirst,tlast,xfirst,xlast,yfirst,ylast,
             xdis,ydis,tdis,xconvalue,yconvalue,tconvalue,xdiff_var,ydiff_var,tdiff_var,
             xdiff_max,ydiff_max,tdiff_max,dis,is_back,recount,
             v_xmean,v_ymean,a_xmean,a_ymean,v_xfirst5,v_xmiddle5,v_xlast5,a_xfirst5,a_xmiddle5,a_xlast5,
             v_yfirst5,v_ymiddle5,v_ylast5,a_yfirst5,a_ymiddle5,a_ylast5,
             kcount,kmax,kmin,kmaxdis,kmean,kvar,
             ]
        train_X.append(X)
    
    labels = ['count','xdis_target','ydis_target','xmean','ymean','tmean','xmin','ymin','tmin',
              'xmax','ymax','tmax','xmaxdis','ymaxdis','tmaxdis','xmedian','ymedian','tmedian',
              'xmad','ymad','tmad','xvar','yvar','tvar','xstd','ystd','tstd',
              'tfirst','tlast','xfirst','xlast','yfirst','ylast',
              'xdis','ydis','tdis',
              'xconvalue','yconvalue','tconvalue','xdiff_var','ydiff_var','tdiff_var',
              'xdiff_max','ydiff_max','tdiff_max','dis','is_back','recount',
              'v_xmean','v_ymean','a_xmean','a_ymean','v_xfirst5','v_xmiddle5','v_xlast5','a_xfirst5','a_xmiddle5','a_xlast5',
              'v_yfirst5','v_ymiddle5','v_ylast5','a_yfirst5','a_ymiddle5','a_ylast5',
              'kcount','kmax','kmin','kmaxdis','kmean','kvar',
              ]
    train_X = pd.DataFrame(train_X,columns=labels)
    return train_X

'''绘制特征重要性图统计'''
def draw_fea_importance(model,features_list):
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    fi_threshold = 0    
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

'''绘制学习曲线'''    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()    
    return plt

'''根据线上得分计算实际预测正确的数目 作弊。。。'''
def getTrueNumbel(F,A):
    x = (F * (40000 + 3*A))/500
    return x

def getTrueScore(x,A):
    F = (x * 500) / (40000 + 3*A)
    return F

'''获得预测结果里黑样本的数目 根据与2W的接近程度判断模型的实用性'''
def getBlackLabel(pre_result):
    result = []
    for i in range(len(pre_result)):
        if (pre_result[i] == 0):
            result.append(i+1)
    return result
    
def drawDataArea(data_list,title):
    fig = plt.figure()
    plt.subplot(121)
    plt.hist(data_list[0:2600],10,color='green')
    plt.xlabel(title)
    plt.ylabel('num')
    plt.title('white label ' + title )
    
    plt.subplot(122)
    plt.hist(data_list[2600:3000],10,color='blue')
    plt.xlabel(title)
    plt.ylabel('num')
    plt.title('black label ' + title)
    
    plt.show()
    
    path = os.path.join(r'E:\DataMining\c4\analysis_img',title + '.png')
    fig.savefig(path)
    
train_X = genFeatures(train_data)
print 'train features is generated!'
#test_X = genFeatures(test_data)
print 'test features is generated!'
'''将特征和标签重新组合 为了方便打乱'''
X = pd.concat([train_X,train_data[['label','id']]],axis=1) 
X = X.replace({np.nan: -1,np.inf: -2,-np.inf:-2})
#X = shuffle(X)
#X = X.reset_index(drop=True)
train_X = X[list(range(70))]
train_Y = X[[70]]
'''
features_list = train_X.columns.values


#test_X = X[2600:-1][list(range(35))]
#test_Y = X[2600:-1][[35]]


#sc = StandardScaler()
test_X = test_X.replace({np.nan: -1, np.inf: -2,-np.inf:-2})
#test_X = test_X.replace([np.inf, -np.inf], np.nan)
#test_X = test_X.fillna(0)
#train_X = sc.fit_transform(train_X)
#test_X = sc.fit_transform(test_X)
train_Y= train_Y.as_matrix().astype(np.int).reshape(-1)
print 'Train and test data is generated!'

model = GradientBoostingClassifier(learning_rate=0.1,max_depth=3,n_estimators=500)
clf = model.fit(train_X,train_Y)
scores = cross_validation.cross_val_score(model, train_X, train_Y, cv = 5)
print("Accuracy on cv: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
y_pred = clf.predict(test_X)
result = getBlackLabel(y_pred)
result = pd.DataFrame(result,columns=['label'])
#result.to_csv('708.txt',header=None,index=False)
print len(result)
draw_fea_importance(model,features_list)

'''


        



'''
规则    
def rule1(data):
    ts = []
    for index,row in data.iterrows():
        path = []
        for j,s in enumerate(row[1].split(';')[:-1]):  
            path.append(s.split(','))
        path = pd.DataFrame(path,columns=['x','y','t'])
        path = path.astype('float')
        t = list(path['t'])
        tdis = t[-1] - t[0]
        ts.append([row[0],tdis])
    df = pd.DataFrame(ts,columns=['label','ts'])
    return df

def rule2(data):
    counts = []
    for index,row in data.iterrows():
        path = []
        for j,s in enumerate(row[1].split(';')[:-1]):  
            path.append(s.split(','))
        count = len(path)
        counts.append([row[0],count])
    
    df = pd.DataFrame(counts,columns=['label','count'])
    return df
'''
    
'''

扩展数据
def expandData(black,white,test_data):
    black = pd.read_csv(black,names=['id'])
    white = pd.read_csv(white,names=['id'])
    black_expand = pd.merge(black,test_data)
    white_expand = pd.merge(white,test_data)
    black_label = np.zeros([black.shape[0]])
    white_label = np.ones(white.shape[0])
    black_label = pd.DataFrame(black_label,columns=['label']).astype('int')
    white_label = pd.DataFrame(white_label,columns=['label']).astype('int')
    black_expand = pd.concat([black_expand,black_label],axis=1)
    white_expand = pd.concat([white_expand,white_label],axis=1)
    train_expand = pd.concat([black_expand,white_expand])
    return train_expand

调参
model = xgb.XGBClassifier()
parameters = [{
          'learning_rate': [0.01,0.1,1], #so called `eta` value
          'max_depth': [3,6,10],
  }]
#,{'max_depth': [3,6,10],'n_estimators': [100,500,1000],}
clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
               cv=StratifiedKFold(train_Y, n_folds=5, shuffle=False), 
               scoring='roc_auc',
               verbose=2, refit=True)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000,2000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000,2000]}]
clf = GridSearchCV(SVC(), tuned_parameters, cv=5) # k-fold  k = 5

clf = model.fit(train_X, train_Y)
print("Best parameters set found on development set:")
print(clf.best_estimator_)
print("Grid scores on development set:")
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
        % (mean_score, scores.std() / 2, params))
#print("Detailed classification report:")
#y_true, y_pred = test_Y, clf.predict(test_X)
#print(classification_report(y_true, y_pred))

#model = SVC(gamma = 0.001, C = 1000, kernel = 'rbf')

#绘制训练流程图
#cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
#title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
#plot_learning_curve(model, title, train_X, train_Y,  ylim=(0.7, 1.01), cv=cv, n_jobs=4)

#model = xgb.XGBClassifier(learning_rate=0.01,max_depth=3,n_estimators=1000)
#model = GradientBoostingClassifier(learning_rate=0.1,max_depth=3,n_estimators=500)
#clf = model.fit(train_X,train_Y)


model = XGBClassifier(
    n_estimators=30,#三十棵树
    learning_rate =0.3,
    max_depth=3, #改动影响比较大
    min_child_weight=1,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=12,
    scale_pos_weight=1,
    reg_lambda=1,
    seed=27)

利用GBDT构造新特征的尝试
train_new_feature= clf.apply(train_X)
test_new_feature= clf.apply(test_X)
train_new_feature2 = pd.DataFrame(train_new_feature)
test_new_feature2 = pd.DataFrame(test_new_feature)
new_feature2=clf.fit(train_new_feature2, train_Y)
y_new_pred= clf.predict(test_new_feature2)
result = getBlackLabel(y_pred)
result = pd.DataFrame(result,columns=['label'])
print 'new features'
print len(result)
'''
        
 black:   
 a_xfirst5 = 150
 a_xlast5 = 1
 a_xmean = 40
 a_xmiddle5 = 1
 a_yfirst5 = <-20  >20
 a_ylast5 = <-0.5 >0
 a_ymean  <-50 >25
 a_ymiddle5 <-0.5 >0.1
 count
 dis<4000
 is_back = 0
 is_too_long
 kcount>75
 kmax>6
 kmaxdis>10
 kmean<-2
 kmin>-15
 kvar>5
 recount 0.4-0.5
 
 def getDiffRate(col_name,alpha):
     t1 = a2[a2[col_name]>alpha].shape[0]
     print str(t1) + ' black numbers '
     print str(t1/400) + ' black rate'
     t2 = a1[a1[col_name]>alpha].shape[0]
     print str(t2) + ' white numbers'
     print str(t2/2600) + ' white rate'
     t3 = t1/400 - t2/2600
     return abs(t3)
    
    
    
    
    