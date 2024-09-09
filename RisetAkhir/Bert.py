import torch
import pandas as pd
from tqdm import tqdm  # Menggunakan tqdm standar

# Load dataset
df = pd.read_excel('DataTraining/datapondasi LbOutputCase.xlsx', sheet_name='Joint Reactions', header=None)

# Prepare header and data
new_header = df.iloc[1]
data_new = df.iloc[2:2153].copy()  # Changed to iloc for correct indexing
data_new.columns = new_header
data_new = data_new.drop(2)

# Select relevant columns
kolom = ['Joint', 'OutputCase', 'LbOutputCase', 'CaseType', 'StepType', 'F1', 'F2', 'F3', 'M1', 'M2', 'M3']
valid_columns = [col for col in kolom if col in data_new.columns]
df2 = data_new[valid_columns].copy()

# Label encoding
possible_labels = df2.LbOutputCase.unique()
label_dict = {possible_label: index for index, possible_label in enumerate(possible_labels)}

df2['label'] = df2.LbOutputCase.replace(label_dict)

# Split data into train and validation sets
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(df2.index.values,
                                                  df2.label.values,
                                                  test_size=0.30,
                                                  random_state=17,
                                                  stratify=df2.label.values)

df2['data_type'] = ['not_set'] * df2.shape[0]
df2.loc[x_train, 'data_type'] = 'train'
df2.loc[x_val, 'data_type'] = 'val'

# Tokenization using BERT tokenizer
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encoded_data_train = tokenizer.batch_encode_plus(
    df2[df2.data_type == 'train'].OutputCase.values,
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=255,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df2[df2.data_type == 'val'].OutputCase.values,
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=255,
    return_tensors='pt'
)

input_ids_train = encoded_data_train['input_ids']
attention_mask_train = encoded_data_train['attention_mask']
label_train = torch.tensor(df2[df2.data_type == 'train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_mask_val = encoded_data_val['attention_mask']
label_val = torch.tensor(df2[df2.data_type == 'val'].label.values)

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
