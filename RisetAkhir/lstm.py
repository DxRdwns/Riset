import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch import nn, optim
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Baca data dan preprocessing seperti sebelumnya
df = pd.read_excel('DataTraining/datapondasi LbOutputCase.xlsx', sheet_name='Joint Reactions', header=None)

new_header = df.iloc[1]
data_new = df.iloc[2:2153].copy()  # Menggunakan iloc untuk pengindeksan yang benar
data_new.columns = new_header
data_new = data_new.drop(2)

# Pilih kolom yang relevan
kolom = [
    'Joint', 'OutputCase', 'LbOutputCase', 'CaseType', 'StepType',
    'F1', 'F2', 'F3', 'M1', 'M2', 'M3'
]
valid_columns = [col for col in kolom if col in data_new.columns]
df2 = data_new[valid_columns].copy()

# Encode labels
possible_labels = df2.LbOutputCase.unique()
label_dict = {label: idx for idx, label in enumerate(possible_labels)}
df2['label'] = df2.LbOutputCase.replace(label_dict)

# Split data
x_train, x_val, y_train, y_val = train_test_split(df2.OutputCase.values, df2.label.values, test_size=0.30, random_state=17, stratify=df2.label.values)

# Tokenization menggunakan tokenizer sederhana
def simple_tokenizer(text):
    return text.split()

# Encode data
def encode_data(tokenizer, texts, max_len=255):
    encoded_data = []
    for text in texts:
        tokens = tokenizer(text)
        token_ids = [ord(c) for c in text[:max_len]]  # Gunakan nilai ASCII sebagai contoh
        if len(token_ids) < max_len:
            token_ids += [0] * (max_len - len(token_ids))  # Padding
        encoded_data.append(token_ids)
    return torch.tensor(encoded_data)

max_len = 255
input_ids_train = encode_data(simple_tokenizer, x_train, max_len=max_len)
input_ids_val = encode_data(simple_tokenizer, x_val, max_len=max_len)

# Konversi label ke tensor
label_train = torch.tensor(y_train)
label_val = torch.tensor(y_val)

# Buat dataset dan dataloader
batch_size = 32
train_dataset = TensorDataset(input_ids_train, label_train)
val_dataset = TensorDataset(input_ids_val, label_val)

train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# Definisikan model LSTM
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(self.dropout(hidden[-1]))
        return output

# Hyperparameter dan inisialisasi model
vocab_size = 256  # Menggunakan rentang karakter ASCII
embedding_dim = 100
hidden_dim = 128
output_dim = len(label_dict)
n_layers = 2
dropout = 0.3

model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout)

# Optimizer dan loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion.to(device)

# Fungsi untuk menghitung metrik
def calculate_metrics(preds, labels):
    preds_flat = np.argmax(preds, axis=1)
    f1 = f1_score(labels, preds_flat, average='weighted')
    precision = precision_score(labels, preds_flat, average='macro')
    recall = recall_score(labels, preds_flat, average='macro')
    accuracy = accuracy_score(labels, preds_flat)
    return f1, precision, recall, accuracy

# Fungsi evaluasi
def evaluate(model, dataloader):
    model.eval()
    loss_total = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_total += loss.item()

            preds = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)

    avg_loss = loss_total / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return avg_loss, all_preds, all_labels

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    train_loss_total = 0

    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_total += loss.item()

    avg_train_loss = train_loss_total / len(train_loader)

    val_loss, val_preds, val_labels = evaluate(model, val_loader)
    val_f1, val_precision, val_recall, val_accuracy = calculate_metrics(val_preds, val_labels)

    print(f'Epoch {epoch+1}/{epochs}')
    print(f'Training Loss: {avg_train_loss}')
    print(f'Validation Loss: {val_loss}')
    print(f'Validation F1 Score: {val_f1}')
    print(f'Validation Precision: {val_precision}')
    print(f'Validation Recall: {val_recall}')
    print(f'Validation Accuracy: {val_accuracy}')
