import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

file_path = 'dataset/datapondasi.xlsx'
data = pd.read_excel(file_path, sheet_name='Joint Reactions', header=None)
new_header = data.iloc[1]
data_new = data[2:].copy()
data_new.columns = new_header
data_new = data_new.drop(2)

# Menampilkan beberapa baris data untuk diperiksa
print(data_new.head())

# Pilih kolom yang relevan dari dataset
kolom = ['Joint', 'OutputCase', 'CaseType', 'StepType', 'F1', 'F2', 'F3', 'M1', 'M2', 'M3']
valid_columns = [col for col in kolom if col in data_new.columns]
df_filtered = data_new[valid_columns].copy()

# Menangani nilai NaN
if 'StepType' in df_filtered.columns:
    df_filtered['StepType'] = df_filtered['StepType'].fillna('Beban layan')

# Mengonversi kolom non-numerik menjadi nilai numerik
le = LabelEncoder()
for col in ['CaseType', 'StepType', 'OutputCase', 'Joint']:
    if col in df_filtered.columns:
        df_filtered[col] = le.fit_transform(df_filtered[col])

# Memisahkan fitur dan target
X = df_filtered.drop(columns=['CaseType'])  
y = df_filtered['CaseType']  

# Mengonversi X dan y ke tipe data float dan int
X = X.astype(float)
y = np.array(y, dtype=int)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model DNN
model = Sequential()

# Lapisan input
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# Lapisan tersembunyi
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Lapisan output
model.add(Dense(1, activation='linear'))  # Gunakan 'softmax' untuk klasifikasi multi-kelas

# Kompilasi model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Menampilkan ringkasan arsitektur model
model.summary()

# Melatih model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Mengevaluasi model pada data uji
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Akurasi model: {accuracy * 100:.2f}%')

# Plot akurasi
plt.plot(history.history['accuracy'], label='Akurasi Training')
plt.plot(history.history['val_accuracy'], label='Akurasi Validation')
plt.title('Akurasi Training dan Validation')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'], label='Loss Validation')
plt.title('Loss Training dan Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
