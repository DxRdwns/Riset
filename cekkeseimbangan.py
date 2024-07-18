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

# Iterasi melalui setiap kolom kategori dan menampilkan distribusi
for col in kategori_kolom:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=filtered_data)
    plt.title(f'Distribusi {col}')
    plt.xticks(rotation=45)
    plt.show()
    
    # Visualisasi distribusi kategori menggunakan pie chart
    plt.figure(figsize=(10, 6))
    filtered_data[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
    plt.title(f'Distribusi {col}')
    plt.ylabel('')
    plt.show()
