import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

# Load data
df = pd.read_excel('dataset/datapondasi.xlsx', sheet_name='Joint Reactions', header=None)

# Prepare header and data
new_header = df.iloc[1]
data_new = df[2:].copy()
data_new.columns = new_header
data_new = data_new.drop(2)

# Select relevant columns
kolom = ['Joint', 'OutputCase', 'CaseType', 'StepType', 'F1', 'F2', 'F3', 'M1', 'M2', 'M3']
valid_columns = [col for col in kolom if col in data_new.columns]
filtered_data = data_new[valid_columns].copy()

# Fill NaN values in 'StepType'
if 'StepType' in filtered_data.columns:
    filtered_data['StepType'] = filtered_data['StepType'].fillna('Beban layan')

# Map 'StepType' to integers
step_type_mapping = {'Beban Layan': 0, 'Min': 1, 'Max': 2}
filtered_data['StepType'] = filtered_data['StepType'].map(step_type_mapping)

# Check for unmapped values and remove them
filtered_data = filtered_data[filtered_data['StepType'].isin(step_type_mapping.values())]

# Extract texts and labels
texts = filtered_data['OutputCase'].tolist()
labels = filtered_data['StepType'].tolist()

# Tokenize texts
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

# Create Dataset class
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are long integers
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset
dataset = TextDataset(encodings, labels)

# Split data into training and validation sets
train_size = 0.8
train_len = int(train_size * len(dataset))
val_len = len(dataset) - train_len
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Update to 3 labels

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,  # Reduce the number of epochs
    per_device_train_batch_size=4,  # Reduce the batch size
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch"  # Save model checkpoints after each epoch
)

# Load smaller model if available
model = BertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train model
trainer.train()

# Evaluate model
results = trainer.evaluate()
print(f"Accuracy: {results['eval_accuracy']}")

# Save model
model.save_pretrained('./finetuned_bert_model')
tokenizer.save_pretrained('./finetuned_bert_model')
