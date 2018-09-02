#coding:utf-8
import pandas as pd
import scipy as sp
import numpy as np
import sklearn
import gc
import warnings
from joblib import Parallel, delayed
# from sklearn.model_selection import train_test_split
# from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import GaussianNB  

# import xgboost as xgb
# from sklearn.grid_search import GridSearchCV
import lightgbm as lgb
import matplotlib
import os
warnings.filterwarnings("ignore")
  
cache = 'E:\DataMining\c4\mouse-github\cache'
sub = 'sub'
datadir = 'data'

train_path = os.path.join(datadir, 'E:\DataMining\c4\mirror\SchoolCompete\data\dsjtzs_txfz_training.txt')
test_path =  os.path.join(datadir, 'E:\DataMining\c4\mirror\SchoolCompete\data\dsjtzs_txfz_test1.txt')

if not os.path.exists(cache):
    os.mkdir(cache)
if not os.path.exists(sub):
    os.mkdir(sub)

def applyParallel(dfGrouped, func):
    with Parallel(n_jobs=32) as parallel:
        retLst = parallel( delayed(func)(pd.Series(value)) for key, value in dfGrouped )
        return pd.concat(retLst, axis=0)

def draw(df):
    import matplotlib.pyplot as plt
    if not os.path.exists('pic'):
        os.mkdir('pic')

    points = []
    for point in df.trajectory[:-1].split(';'):
        point = point.split(',')
        points.append(  ( float(point[0])/7, float(point[1] )/13 ))

    x, y = zip(*points)
    plt.figure(figsize=(16, 5))
    plt.subplot(121)
    plt.plot(x, y)
    plt.subplot(122)
    plt.plot(x, y)
    aim = df.aim.split(',')
    aim = (float(aim[0])/7, float(aim[1])/13)
    plt.scatter(aim[0], aim[1])
    plt.title(df.label)
    plt.savefig('pic/%s-label=%s' %(df.idx, df.label))
    plt.clf()
    plt.close()

def get_feature(df): 
    points = []

    for point in df.trajectory[:-1].split(';'):
        point = point.split(',')
        points.append( ( (float(point[0]), float(point[1])), float(point[2]) ) )

    xs =  pd.Series([point[0][0] for point in points])
    ys =  pd.Series([point[0][1] for point in points])
    ts =  pd.Series([point[1] for point in points])

    aim = df.aim.split(',')
    aim = (float(aim[0]), float(aim[1]))

    distance_deltas = pd.Series([ sp.spatial.distance.euclidean(points[i][0], points[i+1][0])  for i in range(len(points)-1)])

    time_deltas = pd.Series([ points[i+1][1] - points[i][1]  for i in range(len(points)-1)])
    xs_deltas = xs.diff(1)
    ys_deltas = ys.diff(1)

    speeds = pd.Series([ np.log1p(distance) - np.log1p(delta)  for (distance, delta) in zip(distance_deltas, time_deltas) ])
    a_speeds = pd.Series([ np.log1p(speed) - np.log1p(delta)  for (speed, delta) in zip(speeds, time_deltas) ])
    angles = pd.Series([  np.log1p((points[i+1][0][1] - points[i][0][1])) - np.log1p((points[i+1][0][0] - points[i][0][0])) for i in range(len(points)-1)])

    speed_diff = speeds.diff(1).dropna()
    angle_diff = angles.diff(1).dropna()
    a_speed_diff = a_speeds.diff(1).dropna()

    distance_aim_deltas = pd.Series([ sp.spatial.distance.euclidean(points[i][0], aim)  for i in range(len(points))])
    distance_aim_deltas_diff = distance_aim_deltas.diff(1).dropna()

    df['speed_diff_median'] = speed_diff.median()
    df['speed_diff_mean'] = speed_diff.mean()
    df['speed_diff_var'] =  speed_diff.var()
    df['speed_diff_max'] = speed_diff.max()
    df['angle_diff_median'] = angle_diff.median()
    df['angle_diff_mean'] = angle_diff.mean()
    df['angle_diff_var'] =  angle_diff.var()
    df['angle_diff_max'] = angle_diff.max()
    
    df['a_speed_diff_median'] = a_speed_diff.median()
    df['a_speed_diff_mean'] = a_speed_diff.mean()
    df['a_speed_diff_var'] =  a_speed_diff.var()
    df['a_speed_diff_max'] = a_speed_diff.max()

    df['time_delta_min'] =  time_deltas.min()
    df['time_delta_max'] = time_deltas.max()
    df['time_delta_var'] = time_deltas.var()
    df['time_delta_mean'] = time_deltas.mean()

    df['distance_deltas_max'] =  distance_deltas.max()
    df['distance_deltas_min'] =  distance_deltas.min()
    df['distance_deltas_mean'] =  distance_deltas.mean()
    df['distance_deltas_var'] =  distance_deltas.var()
    df['aim_distance_last'] = distance_aim_deltas.values[-1]

    df['aim_distance_diff_max'] = distance_aim_deltas_diff.max()
    df['aim_distance_diff_min'] = distance_aim_deltas_diff.min()
    df['aim_distance_diff_mean'] = distance_aim_deltas_diff.mean()
    df['aim_distance_diff_var'] = distance_aim_deltas_diff.var()
    if len(distance_aim_deltas_diff) > 0:
        df['aim_distance_last'] = distance_aim_deltas_diff.values[-1]
    else:
        df['aim_distance_last'] = -1

    aim_angle = pd.Series([ np.log1p( point[0][1] - aim[1] ) - np.log1p( point[0][0] - aim[0] ) for point in points])
    aim_angle_diff = aim_angle.diff(1).dropna()

    df['aim_angle_last'] = aim_angle.values[-1]
    df['aim_angle_mean'] =  aim_angle.mean()
    df['aim_angle_var'] =  aim_angle.var()
    df['aim_angle_max'] =  aim_angle.max()
    df['aim_angle_min'] =  aim_angle.min()

    df['aim_angle_diff_max'] = aim_angle_diff.max()
    df['aim_angle_diff_var'] = aim_angle_diff.var()
    df['aim_angle_diff_min'] = aim_angle_diff.min()
    df['aim_angle_diff_mean'] = aim_angle_diff.mean()
    if len(aim_angle_diff) > 0:
        df['aim_angle_diff_last'] = aim_angle_diff.values[-1]
    else:
        df['aim_angle_diff_last'] = -1

    df['max_speed'] = speeds.max()
    df['min_speed'] = speeds.min()
    df['mean_speed'] = speeds.mean()
    df['median_speed'] = speeds.median()
    df['var_speed'] = speeds.var()
    length = len(speeds)
    df['last5_speed'] = speeds[length-5:length].mean()
    
    df['max_a_speed'] = a_speeds.max()
    df['min_a_speed'] = a_speeds.min()
    df['mean_a_speed'] = a_speeds.mean()
    df['median_a_speed'] = a_speeds.median()
    df['var_a_speed'] = a_speeds.var()
    length = len(a_speeds)
    df['last5_a_speed'] = a_speeds[length-5:length].mean()

    df['mean_angle'] = angles.mean()
    df['min_angle'] = angles.min()
    df['max_angle'] = angles.max()
    df['var_angle'] = angles.var()
    df['kurt_angle'] = angles.kurt()
    
    df['t_first'] = ts[0]

    df['y_min'] = ys.min()
    df['y_max'] = ys.max()
    df['y_var'] = ys.var()

    df['x_min'] = xs.min()
    df['x_max'] = xs.max()
    df['x_var'] = xs.var()

    df['x_back_num'] = min( (xs_deltas.dropna() > 0).sum(), (xs_deltas.dropna() < 0).sum())
    df['y_back_num'] = min( (ys_deltas.dropna() > 0).sum(), (ys_deltas.dropna() < 0).sum())

    df['xs_delta_var'] = xs_deltas.var()
    df['xs_delta_max'] = xs_deltas.max()
    df['xs_delta_min'] = xs_deltas.min()
    df['xs_delta_mean'] = xs_deltas.mean()
    df['ys_delta_mean'] = ys_deltas.mean()
    df['time_delta_mean'] = time_deltas.mean()

    return df.to_frame().T

def get_single_feature(df):
    points = []

    for point in df.trajectory[:-1].split(';'):
        point = point.split(',')
        points.append( ( ( float(point[0]), float(point[1]) ), float(point[2]) ) )

    aim = df.aim.split(',')
    aim = (float(aim[0]), float(aim[1]))

    aim_angle = pd.Series([ np.log1p( point[0][1] - aim[1] ) - np.log1p( point[0][0] - aim[0] ) for point in points])
    aim_angle_diff = aim_angle.diff(1).dropna()

    df['aim_angle_last'] = aim_angle.values[-1]
    df['aim_angle_diff_max'] = aim_angle_diff.max()
    df['aim_angle_diff_var'] = aim_angle_diff.var()

    if len(aim_angle_diff) > 0:
        df['aim_angle_diff_last'] = aim_angle_diff.values[-1]
    else:
        df['aim_angle_diff_last'] = -1
    return df.to_frame().T

def make_train_set():
    dump_path = os.path.join(cache, 'train.hdf')
    if os.path.exists(dump_path):
        train = pd.read_hdf(dump_path, 'all')
    else:
        train = pd.read_csv(train_path, sep=' ', header=None, names=['id', 'trajectory', 'aim', 'label'])
        train['count'] = train.trajectory.map(lambda x: len(x.split(';')))
        train = applyParallel(train.iterrows(), get_feature).sort_values(by='id')
        train.to_hdf(dump_path, 'all')
    print 'train data is ok!'
    return train

def make_test_set():
    dump_path = os.path.join(cache, 'test.hdf')
    if os.path.exists(dump_path):
        test = pd.read_hdf(dump_path, 'all')
    else:
        test =  pd.read_csv(test_path, sep=' ', header=None, names=['id', 'trajectory', 'aim'])
        test['count'] = test.trajectory.map(lambda x: len(x.split(';')))
        test = applyParallel(test.iterrows(), get_feature).sort_values(by='id')
        test.to_hdf(dump_path, 'all')
    print 'test data is ok!'
    return test

def make_test_set2():
    dump_path = os.path.join(cache, 'test2.hdf')
    if os.path.exists(dump_path):
        test = pd.read_hdf(dump_path, 'all')
    else:
        test =  pd.read_csv(test_path2, sep=' ', header=None, names=['id', 'trajectory', 'aim'])
        test['count'] = test.trajectory.map(lambda x: len(x.split(';')))
        test = applyParallel(test.iterrows(), get_feature).sort_values(by='id')
        test.to_hdf(dump_path, 'all')
    print 'test data is ok!'
    return test

def draw_fea_importance(model,features_list):
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

def getBlackLabel(pre_result):
    result = []
    for i in range(len(pre_result)):
        if (pre_result[i] == 0):
            result.append(i+1)
    return result

def model_blending(X_train,y_train,X_test):

    np.random.seed(0)  # seed to shuffle the train set
    n_folds = 10
    verbose = True
    shuffle = False
    X,y,X_submission = X_train,y_train,X_test
    #y = [int(i) for i in y]
    print len(X),len(y),len(X_submission)

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]


    skf = list(StratifiedKFold(y,n_folds))

    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
        lgb.LGBMClassifier(boosting_type='gbdt', objective="binary", nthread=8, seed=42),
        ]

    print "Creating train and test sets for blending."

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        # print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train1, test1) in enumerate(skf):
            print "Fold", i
            # print train1,test1
            # print '....'
            X_train1 = X[train1]
            y_train1 = y[train1]
            X_test1 = X[test1]
            y_test1 = y[test1]
            clf.fit(X_train1, y_train1)
            y_submission = clf.predict_proba(X_test1)[:, 1]
            dataset_blend_train[test1, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    print
    print "Blending."
    clf = SVC(probability=True)
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

    print "Linear stretch of predictions to [0,1]"
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print "Saving Results."
    tmp = np.vstack([range(1, len(y_submission)+1), y_submission]).T
    #np.savetxt(fname='submission.csv', X=tmp, fmt='%d,%0.9f',comments='')
    tmp = pd.DataFrame(tmp,columns=['id','prob'])
    return tmp


if __name__ == '__main__':
    draw_if = False
    train, test = make_train_set(), make_test_set()
 
    X_train,y_train = train.drop(['id', 'trajectory', 'aim', 'label'], axis=1).astype(float), train['label']
    X_test = test.drop(['id', 'trajectory', 'aim'], axis=1).astype(float)
    print (X_train.shape)
    print (X_test.shape)

    ##特征矩阵及其标签
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_train.fillna(0,inplace=True)
    X_test.fillna(0,inplace=True)
    y_train = y_train.as_matrix().astype(np.int).reshape(-1)

    ###以下是对单模型的计算
    #model = GradientBoostingClassifier(learning_rate=0.005,min_samples_split=200,max_depth=3,n_estimators=1500,subsample=0.75)
    #model = GradientBoostingClassifier()
    model = lgb.LGBMClassifier()
    #model = GaussianNB()
    print 'start traning!'
    
    clf = model.fit(X_train,y_train)
    ##交叉验证
    scores = cross_val_score(model, X_train,y_train, cv = 5)
    print("Accuracy on cv: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    y_prob = clf.predict_proba(X_test)
    y_prob = pd.DataFrame(y_prob,columns=['black','white'])
    print y_prob[y_prob['white']<0.5].shape[0]
    #y_test = clf.predict(X_test)
    #result = getBlackLabel(y_test)
    #result = pd.DataFrame(result,columns=['label'])
    #print len(result) #10845 lgb12490
    # features_list = X_train.columns.values
    # draw_fea_importance(model,features_list)
    #
    # for i,j in enumerate(X_train):
    #     print i,j

    # print "type :"
    # print type(X_train)
    # print X_train.shape
    # print X_train.head()
    # print X_train.as_matrix()
    # print y_train.as_matrix()
    # print X_test.as_matrix()
    ###以下是模型融合的计算

    tmp = model_blending(X_train.as_matrix(),y_train,X_test.as_matrix())
    print tmp[tmp['prob']<0.5].shape[0]
