import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from backend.config import settings
from backend.services.model_loader import model_manager
from backend.database import init_db
from backend.routes.health import router as health_router
from backend.routes.predict import router as predict_router
from backend.routes.metrics import router as metrics_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB and load ML model artifacts
    print(f"[{settings.APP_NAME}] Starting up. Initializing SQLite audit database...")
    init_db()
    print(f"[{settings.APP_NAME}] Loading ML models and vectorizers...")
    loaded_status = model_manager.load_artifacts()
    print(f"[{settings.APP_NAME}] Artifact loading summary: {loaded_status}")
    yield
    # Shutdown
    print(f"[{settings.APP_NAME}] Shutting down gracefully.")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-Powered Phishing URL & Message Threat Analyzer using Dual Machine Learning Pipelines (ANN & RNN).",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clean JSON error handlers (Prevent raw stack traces)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field = ".".join(str(loc) for loc in err.get("loc", []))
        errors.append({
            "field": field,
            "message": err.get("msg", "Invalid input value"),
            "type": err.get("type", "value_error")
        })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "Input validation failed. Please review your submission.",
            "details": errors
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log internal error safely on the server console without sending traceback to client
    print(f"[UNHANDLED EXCEPTION] {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing your request. Please try again later."
        }
    )

# Include API Routers
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(metrics_router)

# Mount Frontend static files
frontend_dir = settings.BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_path = frontend_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return JSONResponse(
            status_code=200,
            content={"message": "PhishGuard-AI API is running. Frontend index.html not yet initialized."}
        )
