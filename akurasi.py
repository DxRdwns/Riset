import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

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

print(filtered_data['OutputCase'])
# Mengganti NaN pada kolom 'StepType'
if 'StepType' in filtered_data.columns:
    filtered_data['StepType'] = filtered_data['StepType'].fillna('Beban layan')

# Mengonversi kolom numerik
numerical_columns = ['F1', 'F2', 'F3', 'M1', 'M2', 'M3']
filtered_data[numerical_columns] = filtered_data[numerical_columns].apply(pd.to_numeric, errors='coerce')

# Mengonversi kolom yang berisi string menjadi numerik
label_encoders = {}
string_columns = [ 'OutputCase', 'CaseType', 'StepType']

for col in string_columns:
    le = LabelEncoder()
    filtered_data[col] = le.fit_transform(filtered_data[col])
    label_encoders[col] = le

# Menampilkan 5 baris pertama dari data yang telah difilter
print(filtered_data.head())

# Misalkan kita ingin memprediksi 'StepType'
X = filtered_data.drop('StepType', axis=1)
y = filtered_data['StepType']

# Memisahkan data menjadi pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menggunakan RandomForestClassifier sebagai contoh
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Memprediksi data pengujian
y_pred = model.predict(X_test)

# Mengevaluasi akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy * 100:.2f}%')

# Menampilkan laporan klasifikasi
print(classification_report(y_test, y_pred))

print(filtered_data['StepType'])