# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 16:07:39 2018

@author: Shashwat
"""
# Data Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creating Data Structure with 120 timesteps and 1 output
X_train = []
y_train = []
for i in range(120, 1258):
    X_train.append(training_set_scaled[i-120:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building RNN
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

regressor = Sequential()
# First LSTM Layer
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# Second LSTM Layer
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))
# Third LSTM Layer
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))
# Fourth LSTM Layer
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))
# Fifth LSTM Layer
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))
# Sixth LSTM Layer
regressor.add(LSTM(units = 100))
regressor.add(Dropout(0.2))
# Output Layer
regressor.add(Dense(units = 1))
# Compiling RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the RNN to the training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Getting Real Stock Price of 2017
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting Predicted Stock Price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 120:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(120, 140):
    X_test.append(inputs[i-120:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualisation
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.savefig('graph_120timesteps_6lstm.png')
plt.show()