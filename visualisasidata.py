import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Menampilkan 5 baris pertama dari data yang telah difilter
print(filtered_data.head())

# Mengubah tipe data numerik agar dapat divisualisasikan
numerical_columns = ['F1', 'F2', 'F3', 'M1', 'M2', 'M3']
filtered_data[numerical_columns] = filtered_data[numerical_columns].apply(pd.to_numeric, errors='coerce')

# Membuat histogram untuk kolom numerik
filtered_data[numerical_columns].hist(bins=15, figsize=(15, 10))
plt.suptitle('Histogram dari Kolom Numerik')
plt.show()

# Membuat boxplot untuk kolom numerik
plt.figure(figsize=(15, 10))
sns.boxplot(data=filtered_data[numerical_columns])
plt.title('Boxplot dari Kolom Numerik')
plt.show()

# Membuat heatmap korelasi
plt.figure(figsize=(12, 8))
corr = filtered_data[numerical_columns].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Heatmap Korelasi')
plt.show()
