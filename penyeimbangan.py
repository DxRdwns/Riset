import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

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

# Encode semua kolom kategori ke angka
label_encoders = {}
kategori_kolom = filtered_data.select_dtypes(include=['object']).columns

for col in kategori_kolom:
    le = LabelEncoder()
    filtered_data[col] = le.fit_transform(filtered_data[col])
    label_encoders[col] = le

# Gabungkan 'OutputCase' dan 'Joint' sebagai target
filtered_data['Target'] = filtered_data['OutputCase'] * 1000 + filtered_data['Joint']

# Memisahkan fitur dan target
X = filtered_data.drop(['OutputCase', 'Joint', 'Target'], axis=1)
y = filtered_data['Target']

# Periksa distribusi kelas
print(y.value_counts())

# Pilihan metode penyeimbangan
# Metode penyeimbangan 1: Random Oversampling
ros = RandomOverSampler(random_state=42)
X_resampled_ros, y_resampled_ros = ros.fit_resample(X, y)

# Metode penyeimbangan 2: Random Undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled_rus, y_resampled_rus = rus.fit_resample(X, y)

# Metode penyeimbangan 3: SMOTE
smote = SMOTE(k_neighbors=3, random_state=42)  # Mengurangi k_neighbors jika terlalu banyak
X_resampled_smote, y_resampled_smote = smote.fit_resample(X, y)

# Metode penyeimbangan 4: SMOTEENN
smoteenn = SMOTEENN(random_state=42)
X_resampled_smoteenn, y_resampled_smoteenn = smoteenn.fit_resample(X, y)

# Pilih metode penyeimbangan yang diinginkan
X_resampled, y_resampled = X_resampled_smote, y_resampled_smote

# Menggabungkan kembali fitur dan target yang sudah disample ulang
filtered_data_resampled = pd.DataFrame(X_resampled, columns=X.columns)
filtered_data_resampled['Target'] = y_resampled

# Memisahkan kembali kolom 'OutputCase' dan 'Joint'
filtered_data_resampled['OutputCase'] = filtered_data_resampled['Target'] // 1000
filtered_data_resampled['Joint'] = filtered_data_resampled['Target'] % 1000

# Mengembalikan nama kategori ke label asli
for col, le in label_encoders.items():
    filtered_data_resampled[col] = le.inverse_transform(filtered_data_resampled[col])

# Mengembalikan label asli untuk kolom 'OutputCase' dan 'Joint'
output_case_map_inverse = {idx: value for value, idx in enumerate(filtered_data['OutputCase'].unique())}
filtered_data_resampled['OutputCase'] = filtered_data_resampled['OutputCase'].map(output_case_map_inverse)

joint_map_inverse = {idx: value for value, idx in enumerate(filtered_data['Joint'].unique())}
filtered_data_resampled['Joint'] = filtered_data_resampled['Joint'].map(joint_map_inverse)

# Visualisasi distribusi fitur setelah penyeimbangan
for col in filtered_data_resampled.columns:
    if col not in ['OutputCase', 'Joint', 'Target']:  # Menghindari visualisasi kolom target
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_data_resampled[col], kde=True)
        plt.title(f'Distribusi {col} (Setelah Penyeimbangan)')
        plt.xticks(rotation=45)
        plt.show()

# Visualisasi distribusi target setelah penyeimbangan
plt.figure(figsize=(10, 6))
sns.countplot(x='OutputCase', data=filtered_data_resampled)
plt.title('Distribusi OutputCase (Setelah Penyeimbangan)')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Joint', data=filtered_data_resampled)
plt.title('Distribusi Joint (Setelah Penyeimbangan)')
plt.xticks(rotation=45)
plt.show()
