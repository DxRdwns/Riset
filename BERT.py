import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 1. Muat Model dan Tokenizer
model = BertForSequenceClassification.from_pretrained('./finetuned_bert_model')
tokenizer = BertTokenizer.from_pretrained('./finetuned_bert_model')

# Atur model ke mode evaluasi
model.eval()

# 2. Baca Data dari File Excel
df = pd.read_excel('dataset/datapondasi.xlsx', sheet_name='Joint Reactions', header=None)

# Prepare header and data
new_header = df.iloc[1]
data_new = df[2:].copy()
data_new.columns = new_header
data_new = data_new.drop(2)

# Select relevant columns
kolom = [
    'F1', 'F2', 'F3', 'M1', 'M2', 'M3', 
]
valid_columns = [col for col in kolom if col in data_new.columns]
filtered_data = data_new[valid_columns].copy()

# Gabungkan kolom yang relevan menjadi teks input
filtered_data['text'] = (
    'F1: ' + filtered_data['F1'].astype(str) + ' ' +
    'F2: ' + filtered_data['F2'].astype(str) + ' ' +
    'F3: ' + filtered_data['F3'].astype(str) + ' ' +
    'M1: ' + filtered_data['M1'].astype(str) + ' ' +
    'M2: ' + filtered_data['M2'].astype(str) + ' ' +
    'M3: ' + filtered_data['M3'].astype(str)
)

# Ambil teks yang akan diprediksi
texts = filtered_data['text'].tolist()

# 3. Tokenisasi Teks
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

# 4. Prediksi dengan Model
with torch.no_grad():
    outputs = model(**encodings)
    predictions = torch.sigmoid(outputs.logits)  # Menggunakan sigmoid untuk mendapatkan probabilitas

# Konversi prediksi ke numpy array
predictions = predictions.numpy()

# 5. Interpretasi Hasil sebagai Presentase Noise
filtered_data['NoiseF1'] = predictions[:, 0] * 100  # Presentase noise untuk F1
filtered_data['NoiseF2'] = predictions[:, 1] * 100  # Presentase noise untuk F2
filtered_data['NoiseF3'] = predictions[:, 2] * 100  # Presentase noise untuk F3
filtered_data['NoiseM1'] = predictions[:, 3] * 100  # Presentase noise untuk M1
filtered_data['NoiseM2'] = predictions[:, 4] * 100  # Presentase noise untuk M2
filtered_data['NoiseM3'] = predictions[:, 5] * 100  # Presentase noise untuk M3

# 6. Simpan atau Tampilkan Hasil
filtered_data.to_excel('/PredictNoise/predicted_noise.xlsx', index=False)  # Simpan hasil prediksi ke file Excel
print(filtered_data[['NoiseF1', 'NoiseF2', 'NoiseF3', 'NoiseM1', 'NoiseM2', 'NoiseM3']])  # Tampilkan hasil
