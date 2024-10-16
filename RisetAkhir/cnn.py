import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
file_path = 'dataset/datapondasi.xlsx'
data = pd.read_excel(file_path, sheet_name='Joint Reactions', header=None)
new_header = data.iloc[1]
data_new = data[2:].copy()
data_new.columns = new_header
data_new = data_new.drop(2)

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
X = df_filtered.drop(columns=['StepType'])  # Semua kolom kecuali 'CaseType' sebagai fitur
y = df_filtered['StepType']  # Kolom 'CaseType' sebagai target

# Memeriksa distribusi target
print("Distribusi target (CaseType):")
print(pd.Series(y).value_counts())

# Mengonversi X dan y ke tipe data float dan int
X = X.astype(float)
y = np.array(y, dtype=int)

# Normalisasi fitur
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data untuk CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # (samples, timesteps, features)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Membangun model CNN yang lebih sederhana
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(y)), activation='softmax'))  # Untuk multi-kelas

# Kompilasi model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Menampilkan ringkasan arsitektur model
model.summary()

# Menggunakan Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Melatih model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

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
