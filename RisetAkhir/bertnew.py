import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# 1. Membuat Data Simulasi
# Menghasilkan data deret waktu sederhana
np.random.seed(42)
data_size = 1000
time = np.arange(data_size)
data = np.sin(0.1 * time) + np.random.normal(0, 0.1, data_size)  # Sinusoidal data with noise

# Membuat DataFrame
df = pd.DataFrame(data, columns=['Value'])

# 2. Mempersiapkan Data untuk LSTM
# Menggunakan MinMaxScaler untuk normalisasi
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Menyiapkan dataset untuk LSTM
X, y = [], []
n_past = 10  # Menggunakan 10 langkah waktu sebelumnya untuk prediksi
for i in range(n_past, len(scaled_data)):
    X.append(scaled_data[i-n_past:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)

# Mengubah X menjadi bentuk [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# 3. Membangun Model LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))  # Mengurangi overfitting
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')

# Melatih model
model.fit(X, y, epochs=50, batch_size=32, verbose=1)

# 4. Membuat Prediksi
# Melakukan prediksi pada data yang sama (untuk tujuan contoh)
predictions = model.predict(X)
predictions_rescaled = scaler.inverse_transform(predictions)

# 5. Visualisasi Hasil
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Value'], label='Actual Data', color='blue')
plt.plot(df.index[n_past:,], predictions_rescaled, label='Predicted Data', color='red')
plt.title('LSTM Prediction on Time Series Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Menghitung RMSE
rmse = np.sqrt(mean_squared_error(df['Value'][n_past:], predictions_rescaled))
print(f"RMSE: {rmse:.4f}")
