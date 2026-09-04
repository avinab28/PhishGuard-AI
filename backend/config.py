import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

class Settings(BaseSettings):
    APP_NAME: str = "PhishGuard-AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Storage paths
    BASE_DIR: Path = BASE_DIR
    MODELS_DIR: Path = MODELS_DIR
    DATA_DIR: Path = DATA_DIR
    
    # URL Model Artifacts
    URL_MODEL_PATH: Path = MODELS_DIR / "url_ann_model.keras"
    URL_SCALER_PATH: Path = MODELS_DIR / "url_scaler.joblib"
    URL_METRICS_PATH: Path = MODELS_DIR / "url_metrics.json"
    
    # Message Model Artifacts
    MESSAGE_MODEL_PATH: Path = MODELS_DIR / "message_rnn_model.keras"
    MESSAGE_TOKENIZER_PATH: Path = MODELS_DIR / "message_tokenizer.pkl"
    MESSAGE_METRICS_PATH: Path = MODELS_DIR / "message_metrics.json"
    
    # Database
    DATABASE_PATH: Path = BASE_DIR / "backend" / "scans.db"
    
    # Risk Thresholds
    RISK_THRESHOLD_LOW: float = 0.40
    RISK_THRESHOLD_HIGH: float = 0.70
    
    # Safety Limits
    MAX_URL_LENGTH: int = 2048
    MAX_MESSAGE_LENGTH: int = 10000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
