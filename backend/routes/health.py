from fastapi import APIRouter
from datetime import datetime, timezone
from backend.config import settings
from backend.schemas import HealthResponse
from backend.services.model_loader import model_manager

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """
    Returns system operational health and readiness of ML models.
    """
    models_status = {
        "url_model_loaded": model_manager.url_model is not None,
        "url_scaler_loaded": model_manager.url_scaler is not None,
        "message_model_loaded": model_manager.message_model is not None,
        "message_tokenizer_loaded": model_manager.message_tokenizer is not None,
    }
    
    all_ready = all(models_status.values())
    status_str = "healthy" if all_ready else "degraded_mode"
    
    return HealthResponse(
        status=status_str,
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        models_status=models_status,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
