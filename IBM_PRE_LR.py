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
def apply_dwt(data, wavelet='db1', level=None):
    if level is None:
        level = int(np.log2(len(data)))
    
    coeffs = pywt.wavedec(data, wavelet, level=min(level, pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet))))
    flattened_coeffs = [c for sublist in coeffs for c in sublist]
    return flattened_coeffs
def create_groups(dataset, window_size_2, window_size_3, timeslice_2, timeslice_3, step, d, p):
    X_data, y_data = [], []
    index = 0
    while index + (timeslice_3 * window_size_3) < len(dataset):
        i = 0
        t2, t3 = [], []
        while i < timeslice_3 * window_size_3:
            current_slice = dataset[index + i:index + i + window_size_3]
            if not np.isnan(current_slice).all():
                t3.append(np.mean(current_slice))
            i += window_size_3

        # Populate t2
        t2 = []
        j = timeslice_3 * window_size_3 - timeslice_2 * window_size_2
        while j < i:
            current_slice = dataset[index + j:index + j + window_size_2]
            if not np.isnan(current_slice).all():
                t2.extend(current_slice)
            j += window_size_2

        # Transpoz işlemi
        t3 = np.array(t3).reshape(-1, 1)
        t2 = np.array(t2).reshape(1, -1)
        t3 = np.transpose(t3)
        m1 = np.mean(t3)
        m2 = np.mean(t2)
        t3 = t3 - m1
        t2 = t2 - m2
        my_ar = np.full((timeslice_1,), m1)
        my_arr = np.full((timeslice_2,), m2)
        my_array = np.concatenate([my_arr, my_ar], axis=0)
        d.append(my_array)
        p.append(m1)
        t3=apply_dwt(t3)
        t2=apply_dwt(t2)
        concatenated_slices = np.concatenate([t2, t3], axis=1)
        X_data.append(concatenated_slices)
        y_data = np.append(y_data, dataset[index + timeslice_3 * window_size_3] - m1)
        index += step
    X_data = np.array(X_data)
    a, b, c = X_data.shape
    array = X_data.reshape(a * b, c)
    return array, np.array(y_data)



window_size_1 = 1
window_size_2 = 5
timeslice_1= 4
timeslice_2 = 4
step = 1
xtrm=[]
ytrm=[]
xtem=[]
ytem=[]
from sklearn.linear_model import LinearRegression
X_train, y_train =create_groups(train_data,window_size_1, window_size_2, timeslice_1,timeslice_2,step,xtrm,ytrm)
X_validation, y_validation = create_groups(validation_data,window_size_1, window_size_2, timeslice_1,timeslice_2,step,xtem,ytem)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_validation)
# Reshape y_pred to 2D array
y_pred = y_pred.reshape(-1, 1)
# Reshape y_validation to 2D array
y_validation = y_validation.reshape(-1, 1)
# Inverse transform validation data to original scale
ytem=np.array(ytem)
ytem = ytem.reshape(-1, 1)
y_validation=y_validation+ytem
y_pred=y_pred+ytem
y_validation = scaler.inverse_transform(y_validation)
# Inverse transform predictions to original scale
y_pred = scaler.inverse_transform(y_pred)
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