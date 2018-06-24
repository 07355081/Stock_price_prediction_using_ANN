#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 16:58:31 2018

@author: shashidhar

Prediction of google stock prices for next 20 days using Artificial Neural Networks

"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, training_set.shape[0]):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 30, activation = 'relu', input_dim = 60))

# Adding the second hidden layer
classifier.add(Dense(units = 30, activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, activation = 'relu'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Saving the trained model
model_json = classifier.to_json()
with open('model.json','w') as json_file:
    json_file.write(model_json)
classifier.save_weights('model.h5')


# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

pred_stock_price = []

Last_60_days_data = training_set_scaled[-60:,0].reshape(1,60)

# Predicting the stock prices for next 20 days
for i in range(1,21):
    pred = classifier.predict(Last_60_days_data)
    Last_60_days_data = np.concatenate((Last_60_days_data,pred),axis = 1)
    pred_stock_price.append(pred)
    Last_60_days_data = Last_60_days_data[0,-60:].reshape(1,60)

prediction = np.array(pred_stock_price).reshape(20,1)    
pred_stock_price = sc.inverse_transform(prediction)
    
# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(pred_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
