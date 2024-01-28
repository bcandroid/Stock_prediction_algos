import time

#Used high techs. preprocessing LSTM
# Programın başlangıç zamanını kaydet
start_time = time.time()
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
import pywt
import random
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
# 'Adj Close' fiyatlarını al ve numpy dizisine dönüştür
spy = yf.download('SPY', start='2000-01-01', end='2024-01-25')

closing_prices = spy['Adj Close']
training_data, validation_data = train_test_split(closing_prices, test_size=0.2, shuffle=False)

# Veriyi uygun şekilde yeniden şekillendirme
training_set = training_data.values.reshape(-1, 1)
validation_set = validation_data.values.reshape(-1, 1)

# Feature scaling için MinMaxScaler kullanma
sc = MinMaxScaler(feature_range=(0, 1))

# Eğitim setini ölçeklendirme ve uygun şekilde dönüştürme
training_set_scaled = sc.fit_transform(training_set)

# Validation setini aynı ölçeklendirme parametreleriyle ölçeklendirme
validation_set_scaled = sc.transform(validation_set)

# Creating input sequences for training
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
        my_ar = np.full((timeslice_2,), m1)
        my_arr = np.full((timeslice_3,), m2)
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
X_train, y_train =create_groups(training_set_scaled,window_size_1, window_size_2, timeslice_1,timeslice_2,step,xtrm,ytrm)
X_validation, y_validation = create_groups(validation_set_scaled,window_size_1, window_size_2, timeslice_1,timeslice_2,step,xtem,ytem)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# Reshape inputs for LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))
# Building the LSTM Model
model = keras.Sequential()
model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=32))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=500, batch_size=8, validation_data=(X_validation, y_validation))

def gr(dataset, window_size_2, window_size_3, timeslice_2, timeslice_3, step, d):
    X_data, y_data = [], []
    index = 0
    while index + (timeslice_3 * window_size_3) - 1 < len(dataset):
        
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
        t3 = apply_dwt(t3)
        t2 = apply_dwt(t2)
        my_ar = np.full((timeslice_2,), m1)  # Fix: timeslice_1 should be defined
        my_arr = np.full((timeslice_3,), m2)
        my_array = np.concatenate([my_arr, my_ar], axis=0)
        d.append(my_array)
        concatenated_slices = np.concatenate([t2, t3], axis=1)
        X_data.append(concatenated_slices)
        index += step
    X_data = np.array(X_data)
    if X_data.size == 0:
        print("No data in X_data.")
        return None
    a, b, c = X_data.shape
    array = X_data.reshape(a * b, c)
    return array


def prediction(model, X, days):
    results = []
    result = np.array([])
    for i in range(days):
        d = []
        t = X[i:]  # start from i.index
        last_element = X[-1][0]
        #print("Vektörün sonuncu elemanı:", last_element)
        #print("eleman syısı:", np.array(t).shape)
        z = gr(t, window_size_1, window_size_2, timeslice_1, timeslice_2, step, d) 
        if z is not None:
            z = np.reshape(z, (z.shape[0], z.shape[1], 1))
            pre = model.predict(z)
            d_array = np.array(d)
            first_element = d_array[0][0]
            #print("ortalama:", np.array([[first_element]]))
            #print("inverse_transformed pre:", pre)
            result = np.concatenate((result, [first_element]))
            #print("PREappending:", pre)
            pre=pre+ np.array([[first_element]])
            # Convert NumPy array to Pandas Series
            # Append the Pandas Series to the list
            #print("appending:", pre)
            X = np.concatenate([X, pre])

    return np.array(X)



yay=validation_set_scaled[-timeslice_2 * window_size_2:]

day = 365
predi = prediction(model, yay, day)
pred=predi[timeslice_2 * window_size_2-1:]
pred = sc.inverse_transform(pred)
print(pred)
arr1=validation_data

arr2_shifted = np.roll(pred, len(validation_data))
arr2_shifted = np.squeeze(arr2_shifted)
combined_array = np.concatenate((arr1, arr2_shifted), axis=0)

x_values_arr1 = np.arange(len(arr1))
x_values_arr2 = np.arange(len(arr1), len(arr1) + len(arr2_shifted))

# İki diziyi bir araya getirip çizdir
plt.plot(x_values_arr1, arr1, color='blue', label='arr1')  # arr1 mavi renk
plt.plot(x_values_arr2, arr2_shifted, color='red', label='arr2')  # arr2_shifted kırmızı renk

# Eksen etiketleri ve başlık eklemek için
plt.xlabel('X Ekseni')
plt.ylabel('Y Ekseni')
plt.title('')

# Legend eklemek için
plt.legend()

# Grafiği gösterme
plt.show()
