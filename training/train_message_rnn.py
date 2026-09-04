import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from backend.config import settings
from backend.services.feature_extraction import clean_message_text
from training.data_loader import fetch_or_generate_message_dataset, split_dataset

# Set deterministic random seeds
np.random.seed(42)
tf.random.set_seed(42)

MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 80
EMBEDDING_DIM = 32

def train_message_rnn():
    print("=" * 60)
    print("PHISHGUARD-AI: Training Pipeline B - Message RNN Classifier")
    print("=" * 60)

    # 1. Load dataset (fetch real UCI SMS Spam Collection)
    df = fetch_or_generate_message_dataset(use_real_if_available=True)
    
    # Balanced sampling for optimal neural representation
    if len(df) > 1000:
        ham_df = df[df["label"] == 0]
        spam_df = df[df["label"] == 1]
        n_each = min(len(spam_df), 600)
        df = pd.concat([
            ham_df.sample(n=n_each, random_state=42),
            spam_df.sample(n=n_each, random_state=42)
        ]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"Sampled balanced subset of {len(df)} messages ({n_each} ham, {n_each} spam).")
    else:
        print(f"Total Message dataset size: {len(df)} samples")
    print(f"Class balance: {df['label'].value_counts().to_dict()}")

    # 2. Text cleaning
    print("Cleaning and standardizing message texts...")
    df["cleaned_text"] = df["message"].apply(clean_message_text)

    # 3. Splits: 70% Train / 15% Val / 15% Test
    train_df, val_df, test_df = split_dataset(df, label_col="label", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 4. Strict ML Rule: Fit Tokenizer on TRAIN only
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["cleaned_text"].tolist())
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)
    print(f"Vocabulary size: {vocab_size} unique tokens (capped at {MAX_VOCAB_SIZE})")

    # Sequence padding
    train_seqs = tokenizer.texts_to_sequences(train_df["cleaned_text"].tolist())
    val_seqs = tokenizer.texts_to_sequences(val_df["cleaned_text"].tolist())
    test_seqs = tokenizer.texts_to_sequences(test_df["cleaned_text"].tolist())

    X_train = pad_sequences(train_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    X_val = pad_sequences(val_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    X_test = pad_sequences(test_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

    y_train = train_df["label"].values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.float32)

    # 5. Build RNN Model: Embedding (mask_zero=True) -> SimpleRNN -> Dropout -> Dense -> Sigmoid
    model = models.Sequential([
        layers.Input(shape=(MAX_SEQUENCE_LENGTH,)),
        layers.Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM, mask_zero=True, name="embedding"),
        layers.SimpleRNN(32, return_sequences=False, name="simple_rnn"),
        layers.Dropout(0.3, name="dropout_1"),
        layers.Dense(16, activation="relu", name="dense_1"),
        layers.Dense(1, activation="sigmoid", name="output_sigmoid")
    ], name="PhishGuard_Message_RNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # 6. Train model
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    epochs = 30
    batch_size = 16
    print(f"Training Message RNN for up to {epochs} epochs (batch_size={batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    # 7. Evaluate on Test Split
    print("\nEvaluating Message RNN on unseen Test Split...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_probs = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    conf_mat = confusion_matrix(y_test, y_pred).tolist()

    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1-Score:  {f1:.4f}")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Confusion Matrix: {conf_mat}")

    # 8. Save Artifacts
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save Model
    model.save(str(settings.MESSAGE_MODEL_PATH))
    print(f"Saved model to: {settings.MESSAGE_MODEL_PATH}")

    # Save Tokenizer via pickle
    with open(settings.MESSAGE_TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved tokenizer to: {settings.MESSAGE_TOKENIZER_PATH}")

    # Save Metrics JSON
    metrics_data = {
        "model_type": "Message Recurrent Neural Network (RNN)",
        "architecture": f"Embedding({MAX_VOCAB_SIZE}, {EMBEDDING_DIM}) -> SimpleRNN(32) -> Dropout(0.3) -> Dense(16, ReLU) -> Dense(1, Sigmoid)",
        "accuracy": round(float(test_acc), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "loss": round(float(test_loss), 4),
        "confusion_matrix": conf_mat,
        "dataset_splits": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df)
        },
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "vocab_size": vocab_size,
        "trained_at": datetime.now(timezone.utc).isoformat()
    }
    with open(settings.MESSAGE_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics to: {settings.MESSAGE_METRICS_PATH}")
    print("Message RNN training completed successfully!\n")

if __name__ == "__main__":
    train_message_rnn()
