import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
import tensorflow as tf

# Ganti dengan jalur file yang sesuai
file_path = 'dataset/datapondasi.xlsx'
data = pd.read_excel(file_path, sheet_name='Joint Reactions', header=None)
new_header = data.iloc[1]
data_new = data[2:].copy()
data_new.columns = new_header
data_new = data_new.drop(2)

# Menentukan kolom yang ingin dipilih
kolom = ['Joint', 'OutputCase', 'CaseType', 'StepType', 'F1', 'F2', 'F3', 'M1', 'M2', 'M3']
valid_columns = [col for col in kolom if col in data_new.columns]
filtered_data = data_new[valid_columns].copy()

# Mengganti NaN pada kolom 'StepType'
if 'StepType' in filtered_data.columns:
    filtered_data['StepType'] = filtered_data['StepType'].fillna('Beban layan')

# Encode kolom kategori ke angka
label_encoders = {}
kategori_kolom = filtered_data.select_dtypes(include=['object']).columns
for col in kategori_kolom:
    le = LabelEncoder()
    filtered_data[col] = le.fit_transform(filtered_data[col])
    label_encoders[col] = le

# Menentukan kolom teks dan label untuk analisis sentimen
text_column = 'OutputCase'  # Kolom ini berisi teks untuk analisis sentimen
label_column = 'StepType'  # Kolom ini berisi label sentimen

# Konversi kolom label ke integer jika belum
if filtered_data[label_column].dtype == 'object':
    le = LabelEncoder()
    filtered_data[label_column] = le.fit_transform(filtered_data[label_column])
    label_encoders[label_column] = le

# Periksa apakah ada nilai kosong (NaN) dalam kolom label
filtered_data = filtered_data.dropna(subset=[label_column])

# Ubah label ke integer jika perlu
filtered_data[label_column] = filtered_data[label_column].astype(int)

# Memisahkan fitur dan target
X = filtered_data[text_column].astype(str).values
y = filtered_data[label_column].values

# Penyeimbangan data menggunakan RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X.reshape(-1, 1), y)  # Reshape X untuk penyeimbangan

# Konversi kembali X_resampled ke bentuk asli
X_resampled = X_resampled.flatten()

# Tokenisasi dan padding
max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
tokenizer.fit_on_texts(X_resampled)
X_resampled = tokenizer.texts_to_sequences(X_resampled)
X_resampled = pad_sequences(X_resampled, maxlen=max_len)

# Membagi data ke dalam training dan test set
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Membuat model LSTM
model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(np.unique(y_resampled)), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Melatih model
batch_size = 32
epochs = 5

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)

# Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Membuat prediksi
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Menampilkan laporan klasifikasi
print(classification_report(y_test, y_pred_labels))
