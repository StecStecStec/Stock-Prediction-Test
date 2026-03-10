import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

SEQ_LEN = 60
LABEL_COLUMNS = ['up', 'down', 'neutral']

COLUMNS_TO_EXCLUDE = [
    'up', 'down', 'neutral', 'daily_change_pct',
    'STOCHd_14_3_3', 'STOCHh_14_3_3',
    'MACD_12_26_9', 'MACDs_12_26_9',
    'logr_SMA_5'
]

df = pd.read_csv("market_data_3class_final.csv")
FEATURE_COLUMNS = [col for col in df.columns if col not in COLUMNS_TO_EXCLUDE]

print(f"Features used in model ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

X_raw = df[FEATURE_COLUMNS].values
Y_raw = df[LABEL_COLUMNS].values

split_idx = int(0.8 * len(X_raw))
X_train_raw, X_val_raw = X_raw[:split_idx], X_raw[split_idx:]
Y_train_raw, Y_val_raw = Y_raw[:split_idx], Y_raw[split_idx:]

train_mean = X_train_raw.mean(axis=0)
train_std = X_train_raw.std(axis=0)
train_std[train_std == 0] = 1e-9

X_train_norm = (X_train_raw - train_mean) / train_std
X_val_norm = (X_val_raw - train_mean) / train_std

def create_sequences(X, Y, seq_len):
    X_seq, Y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        Y_seq.append(Y[i+seq_len])
    return np.array(X_seq), np.array(Y_seq)

X_train, Y_train = create_sequences(X_train_norm, Y_train_raw, SEQ_LEN)
X_val, Y_val = create_sequences(X_val_norm, Y_val_raw, SEQ_LEN)

class_indices = np.argmax(Y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(class_indices), y=class_indices)
class_weight_dict = dict(enumerate(class_weights))

def build_model_cnn_lstm(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='causal'),
        layers.MaxPool1D(pool_size=2),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(LABEL_COLUMNS), activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model_cnn_lstm((X_train.shape[1], X_train.shape[2]))

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,
    batch_size=16,
    shuffle=False,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

y_pred = np.argmax(model.predict(X_val), axis=1)
y_true = np.argmax(Y_val, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=LABEL_COLUMNS))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL_COLUMNS, yticklabels=LABEL_COLUMNS)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (3 Classes)')
plt.tight_layout()
plt.show()