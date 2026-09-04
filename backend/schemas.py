from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
import re

class URLPredictionRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=3,
        max_length=2048,
        description="The raw URL string to analyze statically without making any outbound requests."
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("URL cannot be empty or solely whitespace.")
        # Basic sanity check to reject obviously non-URL binary or garbage data
        if any(c in clean for c in ['\x00', '\r', '\n']):
            raise ValueError("URL contains illegal control characters.")
        return clean

class MessagePredictionRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The message text (SMS, email body, chat) to evaluate for phishing or scam indicators."
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Message cannot be empty or solely whitespace.")
        return clean

class HeuristicIndicator(BaseModel):
    name: str
    description: str
    severity: Literal["CRITICAL", "WARNING", "INFO"]
    triggered: bool
    details: Optional[str] = None

class PredictionResponse(BaseModel):
    target: str
    scan_type: Literal["url", "message"]
    probability: float
    confidence_score: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    verdict: str
    indicators: List[HeuristicIndicator]
    features: Optional[Dict[str, Any]] = None
    recommended_action: str
    analyzed_at: str
    model_version: str
    is_fallback: bool = False

class ModelMetricDetails(BaseModel):
    model_type: str
    architecture: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float
    confusion_matrix: List[List[int]]
    dataset_splits: Dict[str, int]
    trained_at: Optional[str] = None
    features_count: Optional[int] = None

class ModelsMetricsResponse(BaseModel):
    status: str
    url_model: Optional[ModelMetricDetails] = None
    message_model: Optional[ModelMetricDetails] = None
    notes: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    models_status: Dict[str, bool]
    timestamp: str

class ScanHistoryItem(BaseModel):
    id: int
    scan_type: str
    target: str
    risk_level: str
    probability: float
    verdict: str
    created_at: str
