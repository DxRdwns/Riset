import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_excel('dataset/combined_resampled_data_original.xlsx', sheet_name='Sheet1', header=None)

# Prepare header and data
new_header = df.iloc[0]
data_new = df[1:].copy()
data_new.columns = new_header

# Define relevant columns
kolom = ['Joint', 'OutputCase', 'CaseType', 'StepType', 'F1', 'F2', 'F3', 'M1', 'M2', 'M3']
valid_columns = [col for col in kolom if col in data_new.columns]
features = data_new[valid_columns].copy()

# Encode categorical features
label_encoders = {}
kategori_kolom = features.select_dtypes(include=['object']).columns

for col in kategori_kolom:
    features[col] = features[col].astype(str)
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])
    label_encoders[col] = le

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler.fit_transform(features)

# Prepare data for LSTM
n_past = 5  # Use last 5 time steps for prediction
X, y = [], []
for i in range(n_past, len(features_scaled)):
    X.append(features_scaled[i-n_past:i, :])
    y.append(features_scaled[i, :])

X, y = np.array(X), np.array(y)

# Set up TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Initialize lists to store metrics for each fold
accuracies, precisions, recalls, f1_scores = [], [], [], []

# Perform cross-validation
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))  # Regularization
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))  # Regularization
    model.add(Dense(X_train.shape[2]))  # Output layer

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, mode='min')

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1, validation_data=(X_test, y_test),
              callbacks=[early_stopping])

    # Make predictions
    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(predictions)

    # Calculate errors and threshold for noise detection
    errors = np.abs(predictions_rescaled - scaler.inverse_transform(y_test))
    threshold = np.percentile(np.max(errors, axis=1), 95)

    # Create ground truth and predicted labels for noise
    true_labels = np.zeros(len(y_test))
    predicted_labels = np.zeros(len(y_test))
    noise_indices = np.where(np.max(errors, axis=1) > threshold)[0]
    true_labels[noise_indices] = 1
    predicted_labels[noise_indices] = 1

    # Calculate metrics for this fold
    accuracies.append(accuracy_score(true_labels, predicted_labels))
    precisions.append(precision_score(true_labels, predicted_labels, zero_division=1))
    recalls.append(recall_score(true_labels, predicted_labels, zero_division=1))
    f1_scores.append(f1_score(true_labels, predicted_labels, zero_division=1))

# Print average metrics across all folds
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print(f"Average Precision: {np.mean(precisions):.4f}")
print(f"Average Recall: {np.mean(recalls):.4f}")
print(f"Average F1 Score: {np.mean(f1_scores):.4f}")

# Use unseen data for additional validation
unseen_data = features_scaled[-int(0.1 * len(features_scaled)):]
X_unseen, y_unseen = [], []
for i in range(n_past, len(unseen_data)):
    X_unseen.append(unseen_data[i-n_past:i, :])
    y_unseen.append(unseen_data[i, :])

X_unseen, y_unseen = np.array(X_unseen), np.array(y_unseen)

# Predict with the trained model on unseen data
predictions_unseen = model.predict(X_unseen)
predictions_unseen_rescaled = scaler.inverse_transform(predictions_unseen)

# Calculate errors and threshold for unseen data
errors_unseen = np.abs(predictions_unseen_rescaled - scaler.inverse_transform(y_unseen))
noise_indices_unseen = np.where(np.max(errors_unseen, axis=1) > threshold)[0]

# Create ground truth and predicted labels for unseen data
true_labels_unseen = np.zeros(len(y_unseen))
predicted_labels_unseen = np.zeros(len(y_unseen))
true_labels_unseen[noise_indices_unseen] = 1
predicted_labels_unseen[noise_indices_unseen] = 1

# Calculate metrics for unseen data
accuracy_unseen = accuracy_score(true_labels_unseen, predicted_labels_unseen)
precision_unseen = precision_score(true_labels_unseen, predicted_labels_unseen, zero_division=1)
recall_unseen = recall_score(true_labels_unseen, predicted_labels_unseen, zero_division=1)
f1_unseen = f1_score(true_labels_unseen, predicted_labels_unseen, zero_division=1)

# Calculate MAE and RMSE for unseen data
mae_unseen = mean_absolute_error(scaler.inverse_transform(y_unseen), predictions_unseen_rescaled)
rmse_unseen = np.sqrt(mean_squared_error(scaler.inverse_transform(y_unseen), predictions_unseen_rescaled))

# Print metrics for unseen data
print(f"Unseen Data - Accuracy: {accuracy_unseen:.4f}")
print(f"Unseen Data - Precision: {precision_unseen:.4f}")
print(f"Unseen Data - Recall: {recall_unseen:.4f}")
print(f"Unseen Data - F1 Score: {f1_unseen:.4f}")
print(f"Unseen Data - MAE: {mae_unseen:.4f}")
print(f"Unseen Data - RMSE: {rmse_unseen:.4f}")
