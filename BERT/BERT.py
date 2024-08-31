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
data_new = df[2:203].copy()
data_new.columns = new_header
data_new = data_new.drop(2)

# Select relevant columns
kolom = [
    'Joint', 'OutputCase', 'CaseType', 'StepType',
    'F1', 'F2', 'F3', 'M1', 'M2', 'M3', 
]
valid_columns = [col for col in kolom if col in data_new.columns]
filtered_data = data_new[valid_columns].copy()

# Gabungkan kolom yang relevan menjadi teks input
filtered_data['text'] = (
    'Joint: ' + filtered_data['Joint'].astype(str) + ' ' +
    'OutputCase: ' + filtered_data['OutputCase'].astype(str) + ' ' +
    'CaseType: ' + filtered_data['CaseType'].astype(str) + ' ' +
    'StepType: ' + filtered_data['StepType'].astype(str) + ' ' +
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

# Konversi prediksi ke biner (0 atau 1)
binary_predictions = (predictions >= 0.5).astype(int)

# Tambahkan prediksi ke data
filtered_data['NoiseJoint'] = binary_predictions[:, 0]
filtered_data['NoiseOutputCase'] = binary_predictions[:, 1]
filtered_data['NoiseCaseType'] = binary_predictions[:, 2]
filtered_data['NoiseStepType'] = binary_predictions[:, 3]
filtered_data['NoiseF1'] = binary_predictions[:, 4]
filtered_data['NoiseF2'] = binary_predictions[:, 5]
filtered_data['NoiseF3'] = binary_predictions[:, 6]
filtered_data['NoiseM1'] = binary_predictions[:, 7]
filtered_data['NoiseM2'] = binary_predictions[:, 8]
filtered_data['NoiseM3'] = binary_predictions[:, 9]

# Hitung persentase data yang merupakan noise
total_predictions = len(filtered_data)
noise_predictions = filtered_data[
    ['NoiseJoint', 
     'NoiseOutputCase', 
    'NoiseCaseType',
    'NoiseStepType',
     'NoiseF1', 
     'NoiseF2', 
     'NoiseF3', 
     'NoiseM1', 
     'NoiseM2', 
     'NoiseM3']].sum().sum()
noise_percentage = (noise_predictions / (total_predictions * 6)) * 100

# 5. Simpan atau Tampilkan Hasil
filtered_data.to_excel('predicted_noise.xlsx', index=False)  # Simpan hasil prediksi ke file Excel
print(filtered_data[
    ['NoiseJoint', 
     'NoiseOutputCase', 
    'NoiseCaseType',
    'NoiseStepType',
     'NoiseF1', 
     'NoiseF2', 
     'NoiseF3', 
     'NoiseM1', 
     'NoiseM2', 
     'NoiseM3']])  # Tampilkan hasil
print(f"Persentase data yang merupakan noise: {noise_percentage:.2f}%")
