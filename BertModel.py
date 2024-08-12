import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. Baca Data
df = pd.read_excel('DataTraining/datapondasi.xlsx', sheet_name='Joint Reactions', header=None)

# Prepare header and data
new_header = df.iloc[1]
data_new = df[2:].copy()
data_new.columns = new_header
data_new = data_new.drop(2)

# Select relevant columns
kolom = [
    'Joint', 'OutputCase', 'CaseType', 'StepType', 
    'F1', 'F2', 'F3', 'M1', 'M2', 'M3', 
    'LabelF1', 'LabelF2', 'LabelF3', 'LabelM1', 'LabelM2', 'LabelM3'
]
valid_columns = [col for col in kolom if col in data_new.columns]
filtered_data = data_new[valid_columns].copy()

# 2. Gabungkan semua F* dan M* sebagai teks input
filtered_data['text'] = (
    'F1: ' + filtered_data['F1'].astype(str) + ' ' +
    'F2: ' + filtered_data['F2'].astype(str) + ' ' +
    'F3: ' + filtered_data['F3'].astype(str) + ' ' +
    'M1: ' + filtered_data['M1'].astype(str) + ' ' +
    'M2: ' + filtered_data['M2'].astype(str) + ' ' +
    'M3: ' + filtered_data['M3'].astype(str)
)

# Gabungkan semua label menjadi satu kolom label multi-label
filtered_data['labels'] = filtered_data[['LabelF1', 'LabelF2', 'LabelF3', 'LabelM1', 'LabelM2', 'LabelM3']].values.tolist()

# 3. Tokenisasi Data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(filtered_data['text'].tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')

# 4. Membuat Dataset PyTorch
class MultiLabelDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.labels)

labels = filtered_data['labels'].tolist()
dataset = MultiLabelDataset(encodings, labels)

# Split data into training and validation sets
train_size = 0.8
train_len = int(train_size * len(dataset))
val_len = len(dataset) - train_len
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

# 5. Load Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

# 6. Fungsi untuk menghitung akurasi
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = np.argmax(p.label_ids, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# 7. Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics  # Menambahkan metrik akurasi
)

# 9. Train Model
trainer.train()

# 10. Evaluate Model
results = trainer.evaluate()

# 11. Print akurasi
print(f"Accuracy: {results['eval_accuracy']}")

# 12. Simpan Model
model.save_pretrained('./finetuned_bert_model')
tokenizer.save_pretrained('./finetuned_bert_model')
