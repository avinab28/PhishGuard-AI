from fastapi import APIRouter, HTTPException, status
from backend.schemas import (
    URLPredictionRequest,
    MessagePredictionRequest,
    PredictionResponse
)
from backend.services.model_loader import model_manager
from backend.database import record_scan

router = APIRouter(prefix="/api/v1/predict", tags=["Prediction"])

@router.post("/url", response_model=PredictionResponse)
def predict_url_threat(payload: URLPredictionRequest) -> PredictionResponse:
    """
    Evaluates a URL using Pipeline A (Static Feature Extraction + ANN).
    Strict Zero-Execution Policy: The URL is never visited or requested.
    """
    try:
        response = model_manager.predict_url(payload.url)
        # Log to SQLite audit history
        record_scan(
            scan_type="url",
            target=payload.url,
            risk_level=response.risk_level,
            probability=response.probability,
            verdict=response.verdict
        )
        return response
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while evaluating the URL static profile."
        )

@router.post("/message", response_model=PredictionResponse)
def predict_message_threat(payload: MessagePredictionRequest) -> PredictionResponse:
    """
    Evaluates a message (SMS/Email/Chat) using Pipeline B (Text Normalization + RNN).
    """
    try:
        response = model_manager.predict_message(payload.message)
        # Log to SQLite audit history
        record_scan(
            scan_type="message",
            target=payload.message,
            risk_level=response.risk_level,
            probability=response.probability,
            verdict=response.verdict
        )
        return response
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while analyzing the message sequence."
        )
