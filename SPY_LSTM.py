#Taken this code from Medium as written
import time
# Programın başlangıç zamanını kaydet
start_time = time.time()

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.dates as mdates
import pywt
import random
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Download historical data for SPY from Yahoo Finance
spy = yf.download('SPY', start='2010-01-01', end='2023-01-01')

# Split data into training (2010-2020) and validation (2020-2023) sets
training_data = spy['Adj Close']['2010-01-01':'2020-12-31']
validation_data = spy['Adj Close']['2021-01-01':'2023-01-01']

# Use 'Adj Close' prices as the stock price for training
training_set = training_data.values.reshape(-1, 1)
validation_set = validation_data.values.reshape(-1, 1)
# Feature scaling using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(training_set)
validation_set_scaled = scaler.transform(validation_set)

def create_sequences(data, seq_length=60):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(training_set_scaled)
X_validation, y_validation = create_sequences(validation_set_scaled)

# Reshape inputs for LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
model = keras.Sequential()
model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=50, return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=50, return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=50))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
# Training the Model and store history
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_validation, y_validation))
spy_test = yf.download('SPY', start='2023-01-01', end='2023-07-31')

# Use 'Adj Close' prices as the test data
real_stock_price = spy_test['Adj Close'].values.reshape(-1, 1)

dataset_total = pd.concat((spy['Adj Close'], spy_test['Adj Close']), axis=0)
inputs = dataset_total[len(dataset_total) - len(spy_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
X_test, y_test = create_sequences(inputs)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = y_test.reshape(-1, 1)
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
y_test = scaler.inverse_transform(y_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
# Evaluate the model
mse = mean_squared_error(y_test, predicted_stock_price)

mae = mean_absolute_error(y_test, predicted_stock_price)
r2 = r2_score(y_test,predicted_stock_price)
# Assuming y_pred and y_test are NumPy arrays
# Note: For MAPE, make sure y_test does not contain zeros to avoid division by zero.
# RMSE
rmse = np.sqrt(mean_squared_error(y_test, predicted_stock_price))
# MAPE
mape = np.mean(np.abs((y_test -predicted_stock_price) / y_test)) * 100
print('RMSE:', rmse)
print('MAPE:', mape)
print('MSE: ', mse)
print('MAE: ', mae)
print('R-squared: ', r2)

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(predicted_stock_price, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.show()
end_time = time.time()
elapsed_time = end_time - start_time

# Elapsed time değerini saniye cinsinden ekrana yazdır
print(f"Programın çalışma süresi: {elapsed_time} saniye")