import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Ganti dengan jalur file yang sesuai
file_path = 'dataset/datapondasi.xlsx'
data = pd.read_excel(file_path, sheet_name='Joint Reactions', header=None)
new_header = data.iloc[1]
data_new = data[2:].copy()
data_new.columns = new_header
data_new = data_new.drop(2)

# Menentukan kolom yang ingin dipilih
kolom = ['Joint', 'OutputCase', 'CaseType', 'StepType']
valid_columns = [col for col in kolom if col in data_new.columns]
filtered_data = data_new[valid_columns].copy()

# Mengganti NaN pada kolom 'StepType'
if 'StepType' in filtered_data.columns:
    filtered_data['StepType'] = filtered_data['StepType'].fillna('Beban layan')

# Menampilkan 5 baris pertama dari data yang telah difilter
print(filtered_data.head())

# Menentukan kolom kategori
kategori_kolom = filtered_data.select_dtypes(include=['object']).columns

# Visualisasi data sebelum penyeimbangan
for col in kategori_kolom:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=filtered_data)
    plt.title(f'Distribusi {col} (Sebelum Penyeimbangan)')
    plt.xticks(rotation=45)
    plt.show()
    
    # Visualisasi distribusi kategori menggunakan pie chart
    plt.figure(figsize=(10, 6))
    filtered_data[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
    plt.title(f'Distribusi {col} (Sebelum Penyeimbangan)')
    plt.ylabel('')
    plt.show()

# Encode kolom kategori ke angka
label_encoders = {}
for col in kategori_kolom:
    le = LabelEncoder()
    filtered_data[col] = le.fit_transform(filtered_data[col])
    label_encoders[col] = le

# Memisahkan fitur dan target
X = filtered_data.drop('StepType', axis=1)
y = filtered_data['StepType']

# Pastikan semua label di y ada di encoder
le_step_type = label_encoders['StepType']
y_encoded = le_step_type.transform(y)

# Menerapkan SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

# Menggabungkan kembali fitur dan target yang sudah disample ulang
filtered_data_resampled = pd.DataFrame(X_resampled, columns=X.columns)
filtered_data_resampled['StepType'] = y_resampled

# Mengembalikan nama kategori ke label asli
for col, le in label_encoders.items():
    filtered_data_resampled[col] = le.inverse_transform(filtered_data_resampled[col])
filtered_data_resampled['StepType'] = le_step_type.inverse_transform(filtered_data_resampled['StepType'])

# Visualisasi data setelah penyeimbangan
for col in kategori_kolom:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=filtered_data_resampled)
    plt.title(f'Distribusi {col} (Setelah Penyeimbangan)')
    plt.xticks(rotation=45)
    plt.show()
    
    # Visualisasi distribusi kategori menggunakan pie chart
    plt.figure(figsize=(10, 6))
    filtered_data_resampled[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
    plt.title(f'Distribusi {col} (Setelah Penyeimbangan)')
    plt.ylabel('')
    plt.show()
