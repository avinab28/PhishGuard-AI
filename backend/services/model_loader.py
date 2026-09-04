import os
import json
import pickle
import joblib
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone

from backend.config import settings
from backend.services.feature_extraction import (
    URL_FEATURE_NAMES,
    extract_url_features,
    url_features_to_vector,
    clean_message_text
)
from backend.services.risk_engine import evaluate_risk
from backend.services.explanation_engine import (
    analyze_url_heuristics,
    analyze_message_heuristics
)
from backend.schemas import PredictionResponse, HeuristicIndicator

class ModelManager:
    """
    Manages lifecycle of ML models, scalers, tokenizers, and metrics.
    Loads once on application startup.
    Handles missing models gracefully without crashing the server.
    """
    def __init__(self):
        self.url_model = None
        self.url_scaler = None
        self.message_model = None
        self.message_tokenizer = None
        self.is_initialized = False

    def load_artifacts(self) -> Dict[str, bool]:
        """Loads all ML models and preprocessing assets from disk."""
        status = {
            "url_model": False,
            "url_scaler": False,
            "message_model": False,
            "message_tokenizer": False
        }

        # Lazy import of tensorflow so non-TF processes or tests can import manager
        try:
            import tensorflow as tf
        except ImportError:
            tf = None

        # 1. Load URL ANN Model
        if settings.URL_MODEL_PATH.exists() and tf is not None:
            try:
                self.url_model = tf.keras.models.load_model(str(settings.URL_MODEL_PATH))
                status["url_model"] = True
            except Exception as e:
                print(f"[ERROR] Failed to load URL model from {settings.URL_MODEL_PATH}: {e}")
                self.url_model = None

        # 2. Load URL Scaler
        if settings.URL_SCALER_PATH.exists():
            try:
                self.url_scaler = joblib.load(str(settings.URL_SCALER_PATH))
                status["url_scaler"] = True
            except Exception as e:
                print(f"[ERROR] Failed to load URL scaler from {settings.URL_SCALER_PATH}: {e}")
                self.url_scaler = None

        # 3. Load Message RNN Model
        if settings.MESSAGE_MODEL_PATH.exists() and tf is not None:
            try:
                self.message_model = tf.keras.models.load_model(str(settings.MESSAGE_MODEL_PATH))
                status["message_model"] = True
            except Exception as e:
                print(f"[ERROR] Failed to load Message model from {settings.MESSAGE_MODEL_PATH}: {e}")
                self.message_model = None

        # 4. Load Message Tokenizer
        if settings.MESSAGE_TOKENIZER_PATH.exists():
            try:
                with open(settings.MESSAGE_TOKENIZER_PATH, "rb") as f:
                    self.message_tokenizer = pickle.load(f)
                status["message_tokenizer"] = True
            except Exception as e:
                print(f"[ERROR] Failed to load Message tokenizer: {e}")
                self.message_tokenizer = None

        self.is_initialized = True
        return status

    def predict_url(self, url: str) -> PredictionResponse:
        """
        Executes Pipeline A: Static Feature Extraction -> Scaler -> ANN -> Risk & Explanations.
        STRICT SECURITY: Never performs external network calls.
        """
        raw_features = extract_url_features(url)
        indicators = analyze_url_heuristics(url, raw_features)

        is_fallback = False
        now_str = datetime.now(timezone.utc).isoformat()

        if self.url_model is not None and self.url_scaler is not None:
            try:
                vec = url_features_to_vector(raw_features)
                X = np.array([vec], dtype=np.float32)
                X_scaled = self.url_scaler.transform(X)
                # Forward pass
                raw_pred = self.url_model(X_scaled, training=False).numpy()
                probability = float(raw_pred[0][0])
            except Exception as e:
                print(f"[WARN] URL model inference error: {e}. Falling back to heuristics.")
                probability = self._heuristic_url_probability(indicators, raw_features)
                is_fallback = True
        else:
            # Model not loaded: Graceful heuristic assessment
            probability = self._heuristic_url_probability(indicators, raw_features)
            is_fallback = True

        risk_level, verdict, confidence, action = evaluate_risk(probability, scan_type="url")

        return PredictionResponse(
            target=url,
            scan_type="url",
            probability=round(probability, 4),
            confidence_score=round(confidence, 2),
            risk_level=risk_level,
            verdict=verdict,
            indicators=indicators,
            features=raw_features,
            recommended_action=action,
            analyzed_at=now_str,
            model_version="1.0.0-ANN",
            is_fallback=is_fallback
        )

    def predict_message(self, message: str) -> PredictionResponse:
        """
        Executes Pipeline B: Text Cleaning -> Tokenizer/Padding -> RNN -> Risk & Explanations.
        """
        indicators = analyze_message_heuristics(message)
        cleaned_text = clean_message_text(message)

        is_fallback = False
        now_str = datetime.now(timezone.utc).isoformat()

        if self.message_model is not None and self.message_tokenizer is not None:
            try:
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                seq = self.message_tokenizer.texts_to_sequences([cleaned_text])
                X = pad_sequences(seq, maxlen=80, padding="post", truncating="post")
                raw_pred = self.message_model(X, training=False).numpy()
                probability = float(raw_pred[0][0])
            except Exception as e:
                print(f"[WARN] Message model inference error: {e}. Falling back to heuristics.")
                probability = self._heuristic_message_probability(indicators)
                is_fallback = True
        else:
            probability = self._heuristic_message_probability(indicators)
            is_fallback = True

        risk_level, verdict, confidence, action = evaluate_risk(probability, scan_type="message")

        return PredictionResponse(
            target=message,
            scan_type="message",
            probability=round(probability, 4),
            confidence_score=round(confidence, 2),
            risk_level=risk_level,
            verdict=verdict,
            indicators=indicators,
            features={"cleaned_text_preview": cleaned_text[:120], "character_count": len(message)},
            recommended_action=action,
            analyzed_at=now_str,
            model_version="1.0.0-RNN",
            is_fallback=is_fallback
        )

    def _heuristic_url_probability(self, indicators: list, features: dict) -> float:
        """Fallback probabilistic scoring based on triggered heuristic severity."""
        score = 0.10
        for ind in indicators:
            if ind.severity == "CRITICAL" and ind.name != "Clean Static Profile":
                score += 0.35
            elif ind.severity == "WARNING":
                score += 0.18
        if features.get("is_https", 1.0) == 0.0:
            score += 0.15
        return min(0.95, max(0.05, score))

    def _heuristic_message_probability(self, indicators: list) -> float:
        """Fallback probabilistic scoring based on message heuristic severity."""
        score = 0.10
        for ind in indicators:
            if ind.severity == "CRITICAL" and ind.name != "Clean Text Profile":
                score += 0.35
            elif ind.severity == "WARNING":
                score += 0.20
        return min(0.95, max(0.05, score))

# Global singleton model manager instance
model_manager = ModelManager()
