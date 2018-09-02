# -*- coding: utf-8 -*-
# encoding: utf-8  
"""
Created on Mon May 29 17:29:24 2017

@author: sky
"""
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing

fileName1 = '..\data\dsjtzs_txfz_training.txt'
fileName2 = '..\data\dsjtzs_txfz_test1.txt'
#min_max_scaler = preprocessing.MinMaxScaler()
#X_train = min_max_scaler.fit_transform(X_train)
#X_test = min_max_scaler.fit_trasform(X_test)
def giveValue(df,X):
    for i, s in enumerate(df['data']):
        for j,x in enumerate(s.split(';')[:-1]):
            X[i][j] = list(map(int,x.split(',')))
    return X

def getData(filename1,filename2):
    df1 = pd.read_csv(filename1,sep=' ',header=None)
    df1.columns = ['id','data','target','label']
    df1 = shuffle(df1)      
    n1 = len(df1)
    df2 = pd.read_csv(filename2,sep=' ',header=None)
    df2.columns = ['id','data','target']
    n2 = len(df2)
    X_train = np.zeros((n1,300,3),dtype=np.int32)
    y = np.zeros((n1,1))
    y[:,0] = df1['label']
    
    X_test = np.zeros((n2,300,3),dtype=np.int32)

    
    X_train = giveValue(df1,X_train)
    X_test = giveValue(df2,X_test)
    return X_train,y,X_test     

X_train, Y_train, X_test = getData(fileName1,fileName2)



from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import np_utils,plot_model  
from keras.optimizers import SGD
from keras import initializers

epochs = 50
batch_size = 300

#Y_train = np_utils.to_categorical(Y_train,2)

model = Sequential()
model.add(Conv1D(16, kernel_initializer='glorot_uniform',kernel_size=3,input_shape=(300,3)))
model.add(Activation('relu')) 
model.add(Conv1D(32, kernel_size=3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(32, kernel_size=3))
model.add(Activation('relu'))
model.add(Conv1D(64, kernel_size=2))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, kernel_size=3))
model.add(Activation('relu'))
model.add(Conv1D(64, kernel_size=3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))



'''
model.add(Conv1D(16, kernel_size=3))
model.add(Activation('relu'))
model.add(Conv1D(16, kernel_size=3))
model.add(Activation('relu'))
model.add(Conv1D(16, kernel_size=2))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(16, kernel_size=3))
model.add(Activation('relu'))
model.add(Conv1D(16, kernel_size=3))
model.add(Activation('relu'))
model.add(Conv1D(16, kernel_size=2))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Dense(16))
model.add(Activation('relu'))
'''
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,validation_split=0.3)
model.save_weights('cnn3.h5',overwrite=True)

'''
df = model.predict_classes(X_test, batch_size=batch_size) 
df = pd.DataFrame(df)
df.columns = ['behavior']
df2 = df[df['behavior'] == 0]
df2=df2.reset_index()
df2 = df2.drop(['behavior'],axis=1)
df2 = df2['index'] + 1
#df2.to_csv('cnn1.txt',index=False)
'''


