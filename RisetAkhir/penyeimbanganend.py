import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

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

# Membuat peta (mapping) dari nilai-nilai 'OutputCase' ke integer
output_case_unique = filtered_data['OutputCase'].unique()
output_case_map = {value: idx for idx, value in enumerate(output_case_unique)}

# Menggunakan mapping untuk mengubah nilai-nilai 'OutputCase' menjadi integer
filtered_data['OutputCase'] = filtered_data['OutputCase'].map(output_case_map)

# Encode kolom kategori ke angka
label_encoders = {}
kategori_kolom = filtered_data.select_dtypes(include=['object']).columns

for col in kategori_kolom:
    le = LabelEncoder()
    filtered_data[col] = le.fit_transform(filtered_data[col])
    label_encoders[col] = le

# Daftar kolom target yang ingin dianalisis
target_columns = ['OutputCase', 'Joint', 'StepType', 'F1', 'F2', 'F3', 'M1', 'M2', 'M3']

# Menyimpan hasil untuk visualisasi dan evaluasi
resampled_results = {}
evaluation_results = {}

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report

for tar in target_columns:
    # Memisahkan fitur dan target
    X = filtered_data.drop(tar, axis=1)
    y = filtered_data[tar]

    # Periksa distribusi kelas
    if len(y.unique()) <= 1:
        print(f"Kolom target '{tar}' memiliki hanya satu kelas. Penyeimbangan tidak diterapkan.")
        continue

    # Hapus kelas dengan hanya satu sampel
    class_counts = y.value_counts()
    classes_to_remove = class_counts[class_counts <= 1].index
    if classes_to_remove.size > 0:
        print(f"Kolom target '{tar}' memiliki kelas dengan hanya satu sampel: {classes_to_remove}. Kelas ini akan dihapus.")
        filtered_data = filtered_data[~filtered_data[tar].isin(classes_to_remove)]
        X = filtered_data.drop(tar, axis=1)
        y = filtered_data[tar]

    # Metode penyeimbangan 1: Random Oversampling
    ros = RandomOverSampler(random_state=42)
    X_resampled_ros, y_resampled_ros = ros.fit_resample(X, y)
    X_train_ros, X_test_ros, y_train_ros, y_test_ros = split_data(X_resampled_ros, y_resampled_ros)
    accuracy_ros, report_ros = train_and_evaluate(X_train_ros, X_test_ros, y_train_ros, y_test_ros)
    resampled_results[tar] = {'RandomOverSampler': pd.DataFrame(X_resampled_ros, columns=X.columns).assign(**{tar: y_resampled_ros})}
    evaluation_results[tar] = {'RandomOverSampler': accuracy_ros * 100}  # Mengubah ke persentase

# Visualisasi distribusi setelah penyeimbangan
for tar, results in resampled_results.items():
    for method, data in results.items():
        plt.figure(figsize=(10, 6))
        sns.countplot(x=tar, data=data)
        plt.title(f'Distribusi {tar} - {method}')
        plt.xticks(rotation=45)
        plt.show()

# Membuat DataFrame untuk hasil akurasi
accuracy_df = pd.DataFrame(evaluation_results).T
accuracy_df.columns.name = 'Metode'
accuracy_df = accuracy_df.reset_index().rename(columns={'index': 'Target'})

# Format kolom akurasi menjadi persentase
accuracy_df['RandomOverSampler'] = accuracy_df['RandomOverSampler'].map('{:.2f}%'.format)

# Hitung rata-rata akurasi
average_accuracy = accuracy_df['RandomOverSampler'].str.rstrip('%').astype(float).mean()

print("\nHasil Akurasi dari Semua Kolom Target dan Metode:")
print(accuracy_df)

print(f"\nNilai Rata-Rata Akurasi: {average_accuracy:.2f}%")
