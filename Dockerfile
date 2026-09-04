# Build and runtime container for PhishGuard-AI
FROM python:3.12-slim

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, training scripts, frontend assets
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY training/ ./training/
COPY models/ ./models/
COPY tests/ ./tests/

# Expose default port
EXPOSE 8000

# Healthcheck (dynamically checks port assigned by cloud host)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, os; p=os.environ.get('PORT', 8000); urllib.request.urlopen(f'http://localhost:{p}/health')" || exit 1

# Start via Python entrypoint (dynamically adapts to $PORT across cloud hosts)
CMD ["python", "-m", "backend.main"]
