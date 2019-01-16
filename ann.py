# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:39:56 2019

@author: dell-pc3
"""

import pandas as pd
from pandas import concat
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from numpy import concatenate
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation

#处理数据为监督学习所需的数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


#读取数据
data=pd.read_csv('fm.csv' ,header=0, index_col=0)
values=data.values

#数据归一化
scaler =MinMaxScaler(feature_range=(0, 1))
scaled=scaler.fit_transform(values)
reframed=series_to_supervised(scaled, 2, 1)
reframed.drop(reframed.columns[[9,10,11]], axis=1, inplace=True)
print(reframed.head())

#分割数据，x,y,train,test
values = reframed.values
n_train_hours = 1271
#分train,test
train=values[:n_train_hours, :]
test=values[n_train_hours:, :]
print(len(test))
#x,y
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y=test[:,:-1], test[:, -1]
print(test[:, -1].shape)

#建立模型
modelfile = 'modelweight.model'
model = Sequential()  #层次模型
model.add(Dense(50,input_dim=8,init='uniform')) #输入层，Dense表示BP层
model.add(Activation('relu'))  #添加激活函数
model.add(Dense(1))  #输出层
model.compile(loss='mean_squared_error', optimizer='adam') #编译模型
model.fit(train_x, train_y, nb_epoch = 200, batch_size = 20) #训练模型50次
model.save_weights(modelfile)

#对测试集进行预测
yhat = model.predict(test_x)
print(yhat.shape)

#把归一化的数据还原
#先处理预测值
inv_yhat = concatenate((yhat, test_x[:, 5:]), axis=1)
inv_yhat=scaler.inverse_transform(inv_yhat)
inv_yhat=inv_yhat[:,0]
#再处理观测值
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_x[:, 5:]), axis=1)
inv_y=scaler.inverse_transform(inv_y)
inv_y=inv_y[:,0]

#把未经还原的观测值和预测值绘出散点图
x=range(len(yhat))
pyplot.plot(x,yhat,label='forecast')
pyplot.plot(x,test_y,'r',label='ob')
pyplot.legend()
pyplot.show()

#把还原的观测值和实际值绘出
pyplot.plot(x,inv_yhat,label='forecast')
pyplot.plot(x, inv_y, 'r', label='ob')
pyplot.legend()
pyplot.show()

rmse = sqrt(mean_squared_error(yhat, test_y))
rmse2 = sqrt(mean_squared_error(inv_yhat, inv_y))
print('Test RMSE: %.3f' % rmse2)




























