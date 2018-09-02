#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: Gsscsd   
@email: Gsscsd@qq.com 
@site: http://gsscsd.loan  
@file: main.py 
@time: 2017/5/3 
"""

from mouse import *
# 为了追求机器学习和最优化算法的最佳性能，我们将特征缩放
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV


def GetFeaMatrix(filename,flag):

    train_X = []
    train_y = []
    data = readFile(filename,flag)
    Ids = get_Id(data)
    #以下是数据清洗和特征提取
    tables,goals,label = getTranData(data,flag)
    xlines,ylines,tlines = transTable(tables)
    xgoals,ygoals = getGoalFea(goals)
    # print xgoals
    # print ygoals
    # print label
    xinits,xmeans,xstds,xmaxs,xmins,xlasts = optFeature(xlines)
    tinits,tmeans,tstds,tmaxs,tmins,tlasts = optFeature(tlines)
    yinits,ymeans,ystds,ymaxs,ymins,ylasts = optFeature(ylines)
    a_s,b_s,c_s,d_s,e_s,f_s,g_s,h_s= getFits(xlines,tlines)
    # train_X.append(xinits,xmeans,xstds,xmaxs,xmins,xlasts)
    # train_X.append(tinits,tmeans,tstds,tmaxs,tmins,tlasts)
    # train_X.append(yinits,ymeans,ystds,ymaxs,ymins,ylasts)
    # train_X.append(a_s,b_s,c_s)
    #x的相关特征
    # train_X.append(xinits)
    train_X.append(xmeans)
    train_X.append(xstds)
    train_X.append(xmaxs)
    train_X.append(xmins)
    train_X.append(xlasts)
    #y的相关特征
    # train_X.append(yinits)
    train_X.append(ymeans)
    train_X.append(ystds)
    train_X.append(ymaxs)
    train_X.append(ymins)
    train_X.append(ylasts)
    #t的相关特征
    train_X.append(tinits)
    train_X.append(tmeans)
    train_X.append(tstds)
    train_X.append(tmaxs)
    train_X.append(tmins)
    train_X.append(tlasts)
    train_X.append(xgoals)
    # train_X.append(ygoals)
    #拟合的相关特征
    train_X.append(a_s)
    train_X.append(b_s)
    train_X.append(c_s)
    train_X.append(d_s)
    train_X.append(e_s)
    train_X.append(f_s)
    train_X.append(g_s)
    train_X.append(h_s)
    #终极特征矩阵
    train_X = np.array(train_X).T
    train_y = np.array(label).T

    if flag == 1 :
        return train_X,train_y
    else :
        return train_X,Ids





if __name__ == "__main__":
    fileName1 = '../data/dsjtzs_txfz_training.txt'
    fileName2 = '../data/dsjtzs_txfz_test1.txt'
    train_X,train_y = GetFeaMatrix(fileName1,1)
    test_X,Ids= GetFeaMatrix(fileName2,0)
    sc = StandardScaler()
    sc.fit(train_X) # 估算每个特征的平均值和标准差
    # sc.mean_ # 查看特征的平均值，由于Iris我们只用了两个特征，所以结果是array([ 3.82857143,  1.22666667])
    # sc.scale_ # 查看特征的标准差，这个结果是array([ 1.79595918,  0.77769705])
    X_train_std = sc.transform(train_X)
    # 注意：这里我们要用同样的参数来标准化测试集，使得测试集和训练集之间有可比性
    X_test_std = sc.transform(test_X)

    #以下开始测试:
    # print "Id:"
    # print Ids
    # 逻辑回归:
    #print "逻辑回归:"
    #from sklearn.linear_model import LogisticRegression
    #lc = LogisticRegression().fit(X_train_std,train_y)#默认参数
    # print lc.predict(X_test_std)
    #y_labels = lc.predict(X_test_std)
    #TIds,FIds= getTag(Ids,y_labels)
    #toText(FIds,"Logistic.txt")
    #print "Logistic Done!"
    #朴素贝叶斯：
    # print "朴素贝叶斯:"
    # from sklearn.naive_bayes import GaussianNB
    # clf = GaussianNB().fit(X_train_std,train_y)#默认参数
    # print clf.predict(X_test_std)
    # y_labels = clf.predict(X_test_std)
    # TIds,FIds= getTag(Ids,y_labels)
    # toText(FIds,"GaussianNB.csv")
    # print "GaussianNB Done!"
    #伯努利分布:
    # print "伯努利分布:"
    # from sklearn.naive_bayes import BernoulliNB
    # clf = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True).fit(X_train_std,train_y)
    # print clf.predict(X_test_std)
    # #多项式分布:
    # # print "多项式分布:"
    # # from sklearn.naive_bayes import MultinomialNB
    # # clf = MultinomialNB().fit(X_train_std,train_y)
    # # print clf.predict(X_test_std)
    #GBDT:
    # print "GBDT:"
    #from sklearn.ensemble import GradientBoostingClassifier
    #clf = GradientBoostingClassifier().fit(X_train_std,train_y)
    # # print clf.predict(X_test_std)
    #y_labels = clf.predict(X_test_std)
    #TIds,FIds= getTag(Ids,y_labels)
    #toText(FIds,"GradientBoostingClassifier.txt")
    # print "GradientBoostingClassifier Done!"
    # #SVM:
    #print "SVM"
    #from sklearn.svm import SVC
    #model = SVC(gamma=0.01, C=5, verbose=True)
    #clf = model.fit(X_train_std,train_y)
    #pred = clf.predict(X_test_std)
    
    #from sklearn.metrics import precision_score,recall_score
    #P = precision_score()
    #R = recall_score()
    #score = (5*P*R/(2P+3R))*100
    # # print clf.predict(X_test_std)
    #y_labels = clf.predict(X_test_std)
    #TIds,FIds= getTag(Ids,y_labels)
    #toText(FIds,"SVC.txt")
    # print "SVC Done!"
    #models = (svm.SVC(kernel='linear', C=C),
    #      svm.LinearSVC(C=C),
    #     svm.SVC(kernel='rbf', gamma=0.7, C=C),
    #      svm.SVC(kernel='poly', degree=3, C=C))
    #models = (clf.fit(X, y) for clf in models)
    
    '''
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier()
    parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [1000], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}
    clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                   cv=StratifiedKFold(train_y, n_folds=5, shuffle=True), 
                   scoring='roc_auc',
                   verbose=2, refit=True)
    clf.fit(X_train_std, train_y)

    #trust your CV!
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('Raw AUC score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    
    test_probs = clf.predict_proba(X_test_std)#[:,1]
    '''
    
    

