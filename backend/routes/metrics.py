import json
from fastapi import APIRouter, HTTPException, status
from typing import List, Dict, Any
from backend.config import settings
from backend.schemas import ModelsMetricsResponse, ModelMetricDetails, ScanHistoryItem
from backend.database import get_recent_scans, clear_scan_history

router = APIRouter(prefix="/api/v1", tags=["Metrics & History"])

@router.get("/models/metrics", response_model=ModelsMetricsResponse)
def get_model_metrics() -> ModelsMetricsResponse:
    """
    Serves authentic model evaluation metrics directly from the saved JSON training artifacts.
    Does NOT fabricate metrics; returns explicit status if models are pending training.
    """
    url_details = None
    message_details = None
    notes = []

    # 1. Read URL ANN Metrics
    if settings.URL_METRICS_PATH.exists():
        try:
            with open(settings.URL_METRICS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                url_details = ModelMetricDetails(
                    model_type=data.get("model_type", "URL ANN"),
                    architecture=data.get("architecture", "Dense -> Dropout -> Dense -> Sigmoid"),
                    accuracy=data.get("accuracy", 0.0),
                    precision=data.get("precision", 0.0),
                    recall=data.get("recall", 0.0),
                    f1_score=data.get("f1_score", 0.0),
                    loss=data.get("loss", 0.0),
                    confusion_matrix=data.get("confusion_matrix", [[0, 0], [0, 0]]),
                    dataset_splits=data.get("dataset_splits", {"train": 0, "validation": 0, "test": 0}),
                    trained_at=data.get("trained_at"),
                    features_count=data.get("features_count")
                )
        except Exception as e:
            notes.append(f"Failed to parse URL metrics file: {e}")
    else:
        notes.append("URL ANN metrics file not found. Training has not been executed yet.")

    # 2. Read Message RNN Metrics
    if settings.MESSAGE_METRICS_PATH.exists():
        try:
            with open(settings.MESSAGE_METRICS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                message_details = ModelMetricDetails(
                    model_type=data.get("model_type", "Message RNN"),
                    architecture=data.get("architecture", "Embedding -> SimpleRNN -> Dropout -> Dense -> Sigmoid"),
                    accuracy=data.get("accuracy", 0.0),
                    precision=data.get("precision", 0.0),
                    recall=data.get("recall", 0.0),
                    f1_score=data.get("f1_score", 0.0),
                    loss=data.get("loss", 0.0),
                    confusion_matrix=data.get("confusion_matrix", [[0, 0], [0, 0]]),
                    dataset_splits=data.get("dataset_splits", {"train": 0, "validation": 0, "test": 0}),
                    trained_at=data.get("trained_at"),
                    features_count=data.get("vocab_size")
                )
        except Exception as e:
            notes.append(f"Failed to parse Message metrics file: {e}")
    else:
        notes.append("Message RNN metrics file not found. Training has not been executed yet.")

    status_str = "complete" if (url_details and message_details) else "partial"

    return ModelsMetricsResponse(
        status=status_str,
        url_model=url_details,
        message_model=message_details,
        notes="; ".join(notes) if notes else "All model metrics loaded from disk."
    )

@router.get("/history", response_model=List[ScanHistoryItem])
def get_history(limit: int = 20) -> List[ScanHistoryItem]:
    """Retrieves recent scan audit history records from SQLite."""
    scans = get_recent_scans(limit=min(limit, 100))
    return [ScanHistoryItem(**item) for item in scans]

@router.delete("/history")
def clear_history() -> Dict[str, str]:
    """Clears scan history."""
    clear_scan_history()
    return {"status": "success", "message": "Scan history cleared."}
