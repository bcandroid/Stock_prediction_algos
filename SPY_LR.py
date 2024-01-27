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
from sklearn.linear_model import LinearRegression
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Download historical data for SPY from Yahoo Finance
spy = yf.download('SPY', start='2010-01-01', end='2023-01-01')

# Split data into training (2010-2020) and validation (2020-2023) sets
train_data = spy['Adj Close']['2010-01-01':'2020-12-31']
validation_data = spy['Adj Close']['2021-01-01':'2023-01-01']

train_data = train_data.values.reshape(-1, 1)
validation_data = validation_data.values.reshape(-1, 1)


scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
validation_data = scaler.transform(validation_data)
def create_sequences(data, seq_length=60):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
from sklearn.linear_model import LinearRegression
X_train, y_train =create_sequences(train_data)
X_validation, y_validation = create_sequences(validation_data)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_validation)

# Reshape y_pred to 2D array
y_pred = y_pred.reshape(-1, 1)

# Inverse transform predictions to original scale
y_pred = scaler.inverse_transform(y_pred)

# Reshape y_validation to 2D array
y_validation = y_validation.reshape(-1, 1)

# Inverse transform validation data to original scale
y_validation = scaler.inverse_transform(y_validation)

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
# Evaluate the model
mse = mean_squared_error(y_validation, y_pred)
mae = mean_absolute_error(y_validation, y_pred)
r2 = r2_score(y_validation, y_pred)

# Assuming y_pred and y_test are NumPy arrays
# Note: For MAPE, make sure y_test does not contain zeros to avoid division by zero.
# RMSE
rmse = np.sqrt(mean_squared_error(y_validation, y_pred))
# MAPE
mape = np.mean(np.abs((y_validation - y_pred) / y_validation)) * 100

print('RMSE:', rmse)
print('MAPE:', mape)
print('MSE: ', mse)
print('MAE: ', mae)
print('R-squared: ', r2)

plt.figure(figsize=(10, 6))
plt.plot(y_validation, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='red')
plt.xlabel('Timeline')
plt.ylabel('Values')
plt.title('IBM STOCK PREDICTION')
plt.legend()
plt.show()
end_time = time.time()
elapsed_time = end_time - start_time

# Elapsed time değerini saniye cinsinden ekrana yazdır
print(f"Programın çalışma süresi: {elapsed_time} saniye")