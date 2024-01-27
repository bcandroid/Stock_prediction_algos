import time

# Programın başlangıç zamanını kaydet
start_time = time.time()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
import pywt
import random
from keras.optimizers import Adam

import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
import pandas as pd
with open('IBM.txt', 'r') as file:
    lines = file.readlines()
    data = []
    dates = []

    for line in lines[1:]:
        parts = line.strip().split(',')
        date = parts[0]
        if '1980-12-12' <= date <= '2022-07-22':
            dates.append(date)
            data.append(float(parts[4]))  # 'Close' column


# Create a DataFrame from the loaded data
df = pd.DataFrame({'Date': pd.to_datetime(dates), 'Value': data})

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Extract the values from the DataFrame
signal = df['Value'].values
train_data, temp_data = train_test_split(signal, test_size=0.3, shuffle=False)
test_data, validation_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

# Remove '.values' when reshaping
train_data = train_data.reshape(-1, 1)
validation_data = validation_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)


scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)
validation_data = scaler.transform(validation_data)
def create_sequences(data, seq_length=60):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data)
X_validation, y_validation = create_sequences(validation_data)
X_test, y_test = create_sequences(test_data)
# Reshape inputs for LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
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
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(predicted_stock_price, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.show()