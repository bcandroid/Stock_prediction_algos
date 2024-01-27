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
train_data, validation_data = train_test_split(signal, test_size=0.3, shuffle=False)


# Remove '.values' when reshaping
train_data = train_data.reshape(-1, 1)
validation_data = validation_data.reshape(-1, 1)


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