import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np

# 1. Baca Data
df = pd.read_excel('BERT/datatraining.xlsx', sheet_name='Joint Reactions', header=None)

# Prepare header and data
new_header = df.iloc[1]
data_new = df[2:203].copy()
data_new.columns = new_header
data_new = data_new.drop(2)

# Select relevant columns
kolom = [
    'Joint', 'OutputCase', 'CaseType', 'StepType', 
    'F1', 'F2', 'F3', 'M1', 'M2', 'M3', 
    'LabelJoint', 'LabelOutputCase', 'LabelCaseType', 'LabelStepType', 
    'LabelF1', 'LabelF2', 'LabelF3', 'LabelM1', 'LabelM2', 'LabelM3'
]
valid_columns = [col for col in kolom if col in data_new.columns]
filtered_data = data_new[valid_columns].copy()

# 2. Gabungkan semua F* dan M* sebagai teks input
filtered_data['text'] = (
    'Joint: ' + filtered_data['Joint'].astype(str) + ' ' +
    'OutputCase: ' + filtered_data['OutputCase'].astype(str) + ' ' +
    'CaseType: ' + filtered_data['CaseType'].astype(str) + ' ' +
    'StepType: ' + filtered_data['StepType'].astype(str) + ' ' +
    'F1: ' + filtered_data['F1'].astype(str) + ' ' +
    'F2: ' + filtered_data['F2'].astype(str) + ' ' +
    'F3: ' + filtered_data['F3'].astype(str) + ' ' +
    'M1: ' + filtered_data['M1'].astype(str) + ' ' +
    'M2: ' + filtered_data['M2'].astype(str) + ' ' +
    'M3: ' + filtered_data['M3'].astype(str)
)

# Gabungkan semua label menjadi satu kolom label multi-label
filtered_data['labels'] = filtered_data[['LabelJoint', 'LabelOutputCase', 'LabelCaseType', 'LabelStepType', 'LabelF1', 'LabelF2', 'LabelF3', 'LabelM1', 'LabelM2', 'LabelM3']].values.tolist()

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
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10, problem_type="multi_label_classification")

# 6. Fungsi untuk menghitung metrik dengan threshold yang dioptimalkan
def compute_metrics(p):
    # Ubah threshold untuk prediksi
    threshold = 0.35  # Cobalah nilai lain seperti 0.4, 0.3, atau gunakan tuning
    preds = np.where(p.predictions > threshold, 1, 0)
    labels = p.label_ids
    
    # Cek jika tidak ada prediksi positif
    if np.sum(preds) == 0:
        print("Warning: No positive predictions.")
    
    # Hitung metrik
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='micro')
    precision = precision_score(labels, preds, average='micro')
    recall = recall_score(labels, preds, average='micro')
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# 7. Training Arguments dengan epoch lebih banyak
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,  # Tingkatkan jumlah epoch
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True  # Load model terbaik berdasarkan metrik evaluasi
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 9. Train Model
trainer.train()

# 10. Evaluate Model
results = trainer.evaluate()

# 11. Print hasil evaluasi
print(f"Accuracy: {results['eval_accuracy']}")
print(f"F1 Score: {results['eval_f1']}")
print(f"Precision: {results['eval_precision']}")
print(f"Recall: {results['eval_recall']}")

# 12. Simpan Model
model.save_pretrained('./finetuned_bert_model')
tokenizer.save_pretrained('./finetuned_bert_model')
