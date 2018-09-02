# -*- coding: utf-8 -*-
"""
Created on Tue May 15 18:59:45 2018

@author: sky
"""
import pandas as pd
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

action_train, userProfile_train, userComment_train, orderFuture_train, orderHistory_train = get_train_info()
action_test, userProfile_test, userComment_test, orderFuture_test, orderHistory_test = get_test_info()

action = pd.concat([action_train,action_test],axis=0)
action = action.sort_values(by=['userid', 'actionTime'], ascending=True)
action['actionType'] = action['actionType'].astype('str') #训练word2vec模型的序列必须设置为str

filter_texts = []
for userid,group in action.groupby(['userid'], as_index=True):
    filter_texts.append(list(group.actionType))

from gensim.models import word2vec
vector_length = 300
model = word2vec.Word2Vec(filter_texts, size=vector_length, window=2, workers=4)

import numpy as np
#动作类型有1-9种 补充动作0 记为空缺动作 因为CNN要求长度一致
embedding_matrix = np.zeros((10, vector_length))
for i in range(1, 10):
    embedding_matrix[i] = model.wv[str(i)]

#处理序列，使所有序列一致
from keras.preprocessing import sequence
max_len = 50
x_train = sequence.pad_sequences(x_original, maxlen=max_len)
y_train = np.array(y_original)
print(x_train.shape, y_train.shape)

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, BatchNormalization, Activation
from keras.layers.merge import Concatenate
from keras import optimizers

embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim = embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=True)

NUM_EPOCHS = 100
BATCH_SIZE = 64
DROP_PORB = (0.5, 0.8)
NUM_FILTERS = (64, 32)
FILTER_SIZES = (2, 3, 5, 8)
HIDDEN_DIMS = 1024
FEATURE_DIMS = 256
ACTIVE_FUNC = 'relu'

sequence_input = Input(shape=(max_len, ), dtype='int32')
embedded_seq = embedding_layer(sequence_input)

# Convolutional block
conv_blocks = []
for size in FILTER_SIZES:
    conv = Convolution1D(filters=NUM_FILTERS[0],
                         kernel_size=size,
                         padding="valid",
                         activation=ACTIVE_FUNC,
                         strides=1)(embedded_seq)
    conv = Convolution1D(filters=NUM_FILTERS[1],
                         kernel_size=2,
                         padding="valid",
                         activation=ACTIVE_FUNC,
                         strides=1)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)

model_tmp = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
model_tmp = Dropout(DROP_PORB[1])(model_tmp)
model_tmp = Dense(HIDDEN_DIMS, activation=ACTIVE_FUNC)(model_tmp)
model_tmp = Dropout(DROP_PORB[0])(model_tmp)
model_tmp = Dense(FEATURE_DIMS, activation=ACTIVE_FUNC)(model_tmp)
model_tmp = Dropout(DROP_PORB[0])(model_tmp)
model_output = Dense(1, activation="sigmoid")(model_tmp)
model = Model(sequence_input, model_output)

opti = optimizers.SGD(lr = 0.01, momentum=0.8, decay=0.0001)

model.compile(loss='binary_crossentropy',
              optimizer = opti,
              metrics=['binary_accuracy'])

model.fit(x_tra, y_tra, batch_size = BATCH_SIZE, validation_data = (x_val, y_val))

from sklearn import metrics
for i in range(NUM_EPOCHS):
    model.fit(x_tra, y_tra, batch_size = BATCH_SIZE, validation_data = (x_val, y_val))
    y_pred = model.predict(x_val)
    val_auc = metrics.roc_auc_score(y_val, y_pred)
    print('val_auc:{0:5f}'.format(val_auc))


model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2)))
# model.add(LSTM(256))
# model.add(Bidirectional(GRU(256)))
model.add(Dense(1, activation='sigmoid'))

opti = optimizers.SGD(lr = 0.01, momentum=0.8, decay=0.0001)
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

'''
本文提供了一种对动态序列建模的思路：将动作序列通过 word2vec，得到每个动作的 embedding 表示，然后将动作序列转化为 embedding 序列并作为 CNN/RNN 的输入。
'''
