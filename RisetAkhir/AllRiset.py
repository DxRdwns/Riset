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
import torch
import pandas as pd
from tqdm import tqdm 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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
df = data_new[valid_columns].copy()

# Mengganti NaN pada kolom 'StepType'
if 'StepType' in df.columns:
    df['StepType'] = df['StepType'].fillna('Beban layan')

# Menampilkan 5 baris pertama dari data yang telah difilter
print(df.head())

# Membuat peta (mapping) dari nilai-nilai 'OutputCase' ke integer
output_case_unique = df['OutputCase'].unique()
output_case_map = {value: idx for idx, value in enumerate(output_case_unique)}

# Menggunakan mapping untuk mengubah nilai-nilai 'OutputCase' menjadi integer
df['OutputCase'] = df['OutputCase'].map(output_case_map)

# Encode kolom kategori ke angka
label_encoders = {}
kategori_kolom = df.select_dtypes(include=['object']).columns

for col in kategori_kolom:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
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
    X = df.drop(tar, axis=1)
    y = df[tar]

    # Periksa distribusi kelas
    if len(y.unique()) <= 1:
        print(f"Kolom target '{tar}' memiliki hanya satu kelas. Penyeimbangan tidak diterapkan.")
        continue

    # Hapus kelas dengan hanya satu sampel
    class_counts = y.value_counts()
    classes_to_remove = class_counts[class_counts <= 1].index
    if classes_to_remove.size > 0:
        print(f"Kolom target '{tar}' memiliki kelas dengan hanya satu sampel: {classes_to_remove}. Kelas ini akan dihapus.")
        df = df[~df[tar].isin(classes_to_remove)]
        X = df.drop(tar, axis=1)
        y = df[tar]

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
# End Penyeimbangan ----------------------------------------------------------------

#BERT
# Label encoding
print('=========== Deteksi Noise BERT ===========')
possible_labels = df.StepType.unique()
label_dict = {possible_label: index for index, possible_label in enumerate(possible_labels)}

df['label'] = df.StepType.replace(label_dict)

# Split data into train and validation sets
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(df.index.values,
                                                df.label.values,
                                                test_size=0.30,
                                                random_state=17,
                                                stratify=df.label.values)

df['data_type'] = ['not_set'] * df.shape[0]
df.loc[x_train, 'data_type'] = 'train'
df.loc[x_val, 'data_type'] = 'val'

# Tokenization using BERT tokenizer
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type == 'train'].OutputCase.values,
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=255,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type == 'val'].OutputCase.values,
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=255,
    return_tensors='pt'
)

input_ids_train = encoded_data_train['input_ids']
attention_mask_train = encoded_data_train['attention_mask']
label_train = torch.tensor(df[df.data_type == 'train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_mask_val = encoded_data_val['attention_mask']
label_val = torch.tensor(df[df.data_type == 'val'].label.values)

dataset_train = TensorDataset(input_ids_train, attention_mask_train, label_train)
dataset_val = TensorDataset(input_ids_val, attention_mask_val, label_val)

# Load BERT model
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                    num_labels=len(label_dict),
                                                    output_attentions=False,
                                                    output_hidden_states=False)

# DataLoader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 32

dataloader_train = DataLoader(dataset_train,
                            sampler=RandomSampler(dataset_train),
                            batch_size=batch_size)

dataloader_val = DataLoader(dataset_val,
                            sampler=SequentialSampler(dataset_val),
                            batch_size=batch_size)

# Optimizer and scheduler
from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),
                lr=1e-5,
                eps=1e-8)

epochs = 5
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train) * epochs)

# Evaluation metrics
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def precision_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return precision_score(labels_flat, preds_flat, average='macro')

def recall_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return recall_score(labels_flat, preds_flat, average='macro')

def accuracy_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, preds_flat)

# Move model to device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def evaluate(dataloader_val):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2],
        }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

# Training loop
training_stats = []

for epoch_i in tqdm(range(0, epochs)):
    model.train()
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc=f'Epoch {epoch_i + 1}/{epochs}', leave=True)
    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2],
        }

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{0:.2f}'.format(loss.item() / len(batch))})

    # Save model after every epoch
    torch.save(model.state_dict(), f'finetuned_BERT_epoch_{epoch_i}.model')

    tqdm.write(f'\nEpoch {epoch_i + 1}')
    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')

    val_loss, predictions, true_vals = evaluate(dataloader_val)
    val_f1 = f1_score_func(predictions, true_vals)
    val_precision = precision_score_func(predictions, true_vals)
    val_recall = recall_score_func(predictions, true_vals)
    val_accuracy = accuracy_score_func(predictions, true_vals)

    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score: {val_f1}')
    tqdm.write(f'Precision Score: {val_precision}')
    tqdm.write(f'Recall Score: {val_recall}')
    tqdm.write(f'Accuracy Score: {val_accuracy}')

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': loss_train_avg,
            'Valid Loss': val_loss,
            'F1 Score': val_f1,
            'Precision Score': val_precision,
            'Recall Score': val_recall,
            'Accuracy Score': val_accuracy
        }
    )
#-------------------End BERT-----------------------------------------

#LSTM
# Define relevant columns
print('=========== Deteksi Noise LSTM ===========')
kolom2 = ['F1', 'F2', 'F3', 'M1', 'M2', 'M3']
valid_columns2 = [col for col in kolom2 if col in data_new.columns]
features = data_new[valid_columns2].copy()  # Extract features

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler.fit_transform(features)

# Prepare data for LSTM
X = []
y = []
n_past = 5  # Number of previous time steps to use for prediction

for i in range(n_past, len(features_scaled)):
    X.append(features_scaled[i-n_past:i, :])
    y.append(features_scaled[i, :])

X, y = np.array(X), np.array(y)

# Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(X.shape[2]))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# Make predictions
predictions = model.predict(X)
predictions_rescaled = scaler.inverse_transform(predictions)

# Calculate errors
errors = np.abs(predictions_rescaled - features[n_past:])

# Set threshold for detecting noise
threshold = 0.1  # Adjust this threshold based on your data
noise_indices = np.where(np.max(errors, axis=1) > threshold)[0] + n_past

# Create ground truth labels (1 for noise, 0 for non-noise)
true_labels = np.zeros(len(features))
true_labels[noise_indices] = 1

# Create predicted labels based on noise indices
predicted_labels = np.zeros(len(features))
predicted_labels[noise_indices] = 1  # 1 for noise

# Calculate metrics
accuracy = accuracy_score(true_labels[n_past:], predicted_labels[n_past:])
precision = precision_score(true_labels[n_past:], predicted_labels[n_past:])
recall = recall_score(true_labels[n_past:], predicted_labels[n_past:])
f1 = f1_score(true_labels[n_past:], predicted_labels[n_past:])

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# END LSTM

# CNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
print('=========== CNN ===========')
df3 = df

# Menangani nilai NaN
if 'StepType' in df3.columns:
    df3['StepType'] = df3['StepType'].fillna('Beban layan')

# Mengonversi kolom non-numerik menjadi nilai numerik
le = LabelEncoder()
for col in ['CaseType', 'StepType', 'OutputCase', 'Joint']:
    if col in df3.columns:
        df3[col] = le.fit_transform(df3[col])

# Memisahkan fitur dan target
X = df3.drop(columns=['StepType'])  # Semua kolom kecuali 'CaseType' sebagai fitur
y = df3['StepType']  # Kolom 'CaseType' sebagai target

# Memeriksa distribusi target
print("Distribusi target (CaseType):")
print(pd.Series(y).value_counts())

# Mengonversi X dan y ke tipe data float dan int
X = X.astype(float)
y = np.array(y, dtype=int)

# Normalisasi fitur
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data untuk CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # (samples, timesteps, features)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Membangun model CNN yang lebih sederhana
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(y)), activation='softmax'))  # Untuk multi-kelas

# Kompilasi model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Menampilkan ringkasan arsitektur model
model.summary()

# Menggunakan Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Melatih model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

# Mengevaluasi model pada data uji
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Akurasi model: {accuracy * 100:.2f}%')

# Plot akurasi
plt.plot(history.history['accuracy'], label='Akurasi Training')
plt.plot(history.history['val_accuracy'], label='Akurasi Validation')
plt.title('Akurasi Training dan Validation')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'], label='Loss Validation')
plt.title('Loss Training dan Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#DNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
print('=========== DNN ===========')

# Mengonversi kolom non-numerik menjadi nilai numerik
le = LabelEncoder()
for col in ['CaseType', 'StepType', 'OutputCase', 'Joint']:
    if col in df3.columns:
        df3[col] = le.fit_transform(df3[col])

# Memisahkan fitur dan target
X = df3.drop(columns=['CaseType'])  
y = df3['CaseType']  

# Mengonversi X dan y ke tipe data float dan int
X = X.astype(float)
y = np.array(y, dtype=int)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model DNN
model = Sequential()

# Lapisan input
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# Lapisan tersembunyi
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Lapisan output
model.add(Dense(1, activation='linear'))  # Gunakan 'softmax' untuk klasifikasi multi-kelas

# Kompilasi model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Menampilkan ringkasan arsitektur model
model.summary()

# Melatih model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Mengevaluasi model pada data uji
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Akurasi model: {accuracy * 100:.2f}%')

# Plot akurasi
plt.plot(history.history['accuracy'], label='Akurasi Training')
plt.plot(history.history['val_accuracy'], label='Akurasi Validation')
plt.title('Akurasi Training dan Validation')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'], label='Loss Validation')
plt.title('Loss Training dan Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
