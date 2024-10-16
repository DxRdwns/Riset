import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Load dataset
df = pd.read_excel('dataset/datapondasi.xlsx', sheet_name='Joint Reactions', header=None)

# Prepare header and data
new_header = df.iloc[1]  # Set header
data = df.iloc[2:].copy()  # Extract data
data.columns = new_header  # Assign headers
data = data.drop(2)  # Drop any unnecessary rows

# Define relevant columns
kolom = ['F1', 'F2', 'F3', 'M1', 'M2', 'M3']
valid_columns = [col for col in kolom if col in data.columns]
features = data[valid_columns].copy()  # Extract features

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

