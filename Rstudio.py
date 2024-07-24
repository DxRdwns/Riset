import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import pingouin as pg
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from scipy.stats import shapiro, normaltest, anderson
from sklearn.preprocessing import LabelEncoder
# Membaca data dari file Excel
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

# Mengganti NaN pada kolom 'StepType' Menjado Beban Layan
if 'StepType' in filtered_data.columns:
    filtered_data['StepType'] = filtered_data['StepType'].fillna('BL')
    

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

# Menampilkan struktur dan data
print(filtered_data.info())
print(filtered_data.head())

# Korelasi data
correlation_matrix = filtered_data.corr()
print(correlation_matrix.round(2))

# Mendapatkan p-value dari matriks korelasi
corr_p = pg.rcorr(filtered_data, method='pearson')
print(corr_p)

# Visualisasi pasangan (pairplot)
sns.pairplot(filtered_data)
plt.show()

# Menghitung KMO (Kaiser-Meyer-Olkin) dan Bartlett's test
kmo_all, kmo_model = calculate_kmo(filtered_data)
print(f"KMO: {kmo_model}")

chi_square_value, p_value = calculate_bartlett_sphericity(filtered_data)
print(f"Bartlett's test: Chi-Square = {chi_square_value}, p-value = {p_value}")

# Menguji asumsi normalitas (univariate test menggunakan Shapiro-Wilk)
for column in filtered_data.columns:
    stat, p = shapiro(filtered_data[column].dropna())
    print(f'Shapiro-Wilk Test for {column}: Statistics={stat}, p-value={p}')

# Menguji normalitas multivariat menggunakan uji D'Agostino's K-squared
stat, p = normaltest(filtered_data.dropna())
print(f'D\'Agostino\'s K-squared Test: Statistics={stat}, p-value={p}')

# Melakukan analisis faktor
fa = FactorAnalysis(n_components=5, rotation='oblimin')
fa.fit(filtered_data)

# Menampilkan hasil analisis faktor
print("Loadings:\n", fa.components_)

# Menampilkan communalities
communalities = np.sum(fa.components_ ** 2, axis=0)
print("Communalities:\n", communalities)

# Plot Scree Plot
eigenvalues = fa.components_.sum(axis=1)
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-', linewidth=2)
plt.axhline(y=1, color='r', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Number of Factors')
plt.ylabel('Eigenvalues')
plt.show()
