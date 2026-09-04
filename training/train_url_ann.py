import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from backend.config import settings
from backend.services.feature_extraction import (
    URL_FEATURE_NAMES,
    extract_url_features,
    url_features_to_vector
)
from training.data_loader import fetch_or_generate_url_dataset, split_dataset

# Set deterministic random seeds for reproducible runs
np.random.seed(42)
tf.random.set_seed(42)

def train_url_ann():
    print("=" * 60)
    print("PHISHGUARD-AI: Training Pipeline A - URL ANN Classifier")
    print("=" * 60)

    # 1. Load dataset
    df = fetch_or_generate_url_dataset(use_real_if_available=False)
    print(f"Total URL dataset size: {len(df)} samples")
    print(f"Class balance: {df['label'].value_counts().to_dict()}")

    # 2. Extract static features for each URL
    print(f"Extracting {len(URL_FEATURE_NAMES)} static features per URL...")
    feature_vectors = []
    labels = []
    for _, row in df.iterrows():
        try:
            feats = extract_url_features(str(row["url"]))
            vec = url_features_to_vector(feats)
            feature_vectors.append(vec)
            labels.append(int(row["label"]))
        except Exception as e:
            print(f"Error extracting features from {row['url']}: {e}")
            continue

    X = np.array(feature_vectors, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    # Convert to DataFrame for stratified splitting
    df_features = pd.DataFrame(X, columns=URL_FEATURE_NAMES)
    df_features["label"] = y

    # 3. Splits: 70% Train / 15% Val / 15% Test
    train_df, val_df, test_df = split_dataset(df_features, label_col="label", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    X_train_raw = train_df[URL_FEATURE_NAMES].values
    y_train = train_df["label"].values

    X_val_raw = val_df[URL_FEATURE_NAMES].values
    y_val = val_df["label"].values

    X_test_raw = test_df[URL_FEATURE_NAMES].values
    y_test = test_df["label"].values

    # 4. Strict ML Rule: Fit Scaler on TRAIN only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # 5. Build ANN Model: Dense -> Dropout -> Dense -> Sigmoid
    input_dim = len(URL_FEATURE_NAMES)
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu", name="dense_1"),
        layers.Dropout(0.3, name="dropout_1"),
        layers.Dense(32, activation="relu", name="dense_2"),
        layers.Dropout(0.2, name="dropout_2"),
        layers.Dense(1, activation="sigmoid", name="output_sigmoid")
    ], name="PhishGuard_URL_ANN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # 6. Train model
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    epochs = 35
    batch_size = 16
    print(f"Training ANN for up to {epochs} epochs (batch_size={batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    # 7. Evaluate strictly on Test Split
    print("\nEvaluating URL ANN on unseen Test Split...")
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
    model.save(str(settings.URL_MODEL_PATH))
    print(f"Saved model to: {settings.URL_MODEL_PATH}")

    # Save Scaler
    joblib.dump(scaler, str(settings.URL_SCALER_PATH))
    print(f"Saved scaler to: {settings.URL_SCALER_PATH}")

    # Save Metrics JSON
    metrics_data = {
        "model_type": "URL Artificial Neural Network (ANN)",
        "architecture": "Input(22) -> Dense(64, ReLU) -> Dropout(0.3) -> Dense(32, ReLU) -> Dropout(0.2) -> Dense(1, Sigmoid)",
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
        "features_count": input_dim,
        "feature_names": URL_FEATURE_NAMES,
        "trained_at": datetime.now(timezone.utc).isoformat()
    }
    with open(settings.URL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics to: {settings.URL_METRICS_PATH}")
    print("URL ANN training completed successfully!\n")

if __name__ == "__main__":
    train_url_ann()
