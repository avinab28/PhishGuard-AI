---
title: PhishGuard AI
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# PhishGuard-AI: AI-Powered Phishing URL & Message Threat Analyzer

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00.svg)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

**PhishGuard-AI** is an autonomous threat intelligence application engineered to identify and explain phishing campaigns across web links (URLs) and text communications (SMS, email, instant messages). Built around a strict **Zero-Execution Policy**, the system evaluates targets using dual isolated Machine Learning pipelines paired with a deterministic forensic heuristics engine.

---

## 1. System Architecture

PhishGuard-AI operates two decoupled ML pipelines served through a unified asynchronous FastAPI backend and an interactive Vanilla HTML5/CSS3/JavaScript frontend:

```
                            +-------------------------------------------+
                            |       Vanilla Web Interface (SPA)         |
                            | (URL Scanner, Message Scanner, Metrics)   |
                            +---------------------+---------------------+
                                                  | HTTP REST / JSON
                                                  v
                            +-------------------------------------------+
                            |           FastAPI Threat Gateway          |
                            |   (Validation, CORS, Model Lifecycle)     |
                            +---------------------+---------------------+
                                                  |
                    +-----------------------------+-----------------------------+
                    |                                                           |
                    v                                                           v
     +------------------------------+                            +------------------------------+
     |   Pipeline A: URL Threat     |                            |  Pipeline B: Message Threat  |
     +------------------------------+                            +------------------------------+
     | 1. Static Feature Extraction |                            | 1. Text Sanitization/Clean   |
     |    (Length, Entropy, IP, TLD)|                            |    (URL/Email/Money tokens)  |
     | 2. StandardScaler Transform  |                            | 2. Keras Word Tokenizer      |
     | 3. Dense ANN (Sigmoid)       |                            | 3. Sequence Padding (maxlen) |
     | 4. Forensic Rule Indicators  |                            | 4. SimpleRNN (Sigmoid)       |
     | 5. Risk Tiering (<0.4/0.7)   |                            | 5. Forensic Rule Indicators  |
     +--------------+---------------+                            +--------------+---------------+
                    |                                                           |
                    +-----------------------------+-----------------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |       SQLite Audit Store (scans.db) |
                               |   (Historical Auditing & Reporting) |
                               +-------------------------------------+
```

### Core Security Principles
1. **Zero-Execution Guarantee**: PhishGuard-AI never resolves, pings, fetches, or renders user-submitted URLs. Static analysis protects internal infrastructure from malicious payloads, zero-days, cloaking kits, and attacker tracking beacons.
2. **Probabilistic Risk Assessments**: Predictions are calibrated risk probabilities, never absolute binaries. Scores map directly to three operational action tiers:
   - **LOW (< 0.40)**: Structural patterns align with benign baselines.
   - **MEDIUM (0.40 – 0.69)**: Suspicious signals or anomalies observed; caution advised.
   - **HIGH (≥ 0.70)**: High probability of phishing or credential theft; immediate block/deletion recommended.
3. **Decoupled Heuristics**: Neural scores are accompanied by deterministic rule-based explainability flags (e.g., raw IP in hostname, homograph punycode, urgency coercion triggers).

---

## 2. Machine Learning Pipelines

### Pipeline A: URL Artificial Neural Network (ANN)
- **Input**: Raw URL string.
- **Feature Extraction**: 22 purely static indicators:
  - Total length, hostname length, path length.
  - Frequency of symbols: `.`, `-`, `_`, `/`, `@`, `?`, `=`, `%`, digits.
  - Ratio of numeric characters to total length.
  - Presence of raw IPv4/IPv6 addresses in host.
  - Protocol verification (`https://` vs `http://`).
  - Subdomain count and depth.
  - High-value authentication/credential keyword count.
  - Shannon character entropy of the domain.
  - High-risk TLDs (`.xyz`, `.top`, `.tk`, `.buzz`, etc.).
  - Punycode homograph indicators (`xn--`).
  - Non-standard port exposure and path redirection markers (`//`).
- **Architecture**:
  - `Input(22)`
  - `Dense(64, activation='relu')`
  - `Dropout(0.3)`
  - `Dense(32, activation='relu')`
  - `Dropout(0.2)`
  - `Dense(1, activation='sigmoid')`
- **Optimizer & Loss**: Adam (`lr=0.002`), Binary Crossentropy.
- **Artifacts Generated**: `models/url_ann_model.keras`, `models/url_scaler.joblib`, `models/url_metrics.json`.

### Pipeline B: Message Recurrent Neural Network (RNN)
- **Input**: Message body (SMS, email, chat).
- **Preprocessing**: Lowercase conversion, entity tokenization (`<URL>`, `<EMAIL>`, `<NUMBER>`, `<MONEY>`), punctuation stripping, and whitespace normalization.
- **Sequence Processing**: Keras `Tokenizer` (`vocab_size=5000`) with post-padding (`max_length=80`).
- **Architecture**:
  - `Input(shape=(80,))`
  - `Embedding(input_dim=5000, output_dim=32)`
  - `SimpleRNN(32, return_sequences=False)`
  - `Dropout(0.3)`
  - `Dense(16, activation='relu')`
  - `Dense(1, activation='sigmoid')`
- **Optimizer & Loss**: Adam (`lr=0.002`), Binary Crossentropy.
- **Artifacts Generated**: `models/message_rnn_model.keras`, `models/message_tokenizer.pkl`, `models/message_metrics.json`.

---

## 3. Project Directory Structure

```
PhishGuard-AI1/
├── backend/
│   ├── __init__.py
│   ├── config.py                 # Central settings and risk thresholds
│   ├── database.py               # SQLite audit history operations
│   ├── main.py                   # FastAPI application & lifespan manager
│   ├── schemas.py                # Pydantic v2 schemas and validators
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py             # GET /health
│   │   ├── predict.py            # POST /api/v1/predict/url & message
│   │   └── metrics.py            # GET /api/v1/models/metrics & history
│   └── services/
│       ├── __init__.py
│       ├── feature_extraction.py # 22 static features + text cleaner
│       ├── explanation_engine.py # Rule-based heuristic indicators
│       ├── model_loader.py       # Singleton ML artifact manager
│       └── risk_engine.py        # Probability-to-tier evaluation
├── frontend/
│   ├── index.html                # Responsive Vanilla JS SPA
│   ├── css/
│   │   └── styles.css            # Modern dark-mode cybersecurity theme
│   └── js/
│       └── app.js                # State, API handlers, meter animation
├── training/
│   ├── __init__.py
│   ├── data_loader.py            # UCI SMS Spam / URL datasets + synthetic fallback
│   ├── train_url_ann.py          # Pipeline A training script
│   └── train_message_rnn.py      # Pipeline B training script
├── models/                       # Model artifacts and metrics storage
├── tests/
│   ├── __init__.py
│   ├── test_features.py          # Vectorization & cleaning unit tests
│   ├── test_risk_engine.py       # Risk scoring & explanation unit tests
│   └── test_api.py               # FastAPI integration test suite
├── Dockerfile                    # Containerization specification
├── docker-compose.yml            # Multi-container orchestration
├── requirements.txt              # Production Python dependencies
└── README.md                     # Comprehensive documentation
```

---

## 4. Local Setup & Installation

### Prerequisites
- Python 3.12+
- `pip` or `uv`

### Step 1: Clone and Create Virtual Environment
```bash
# Windows
py -3.12 -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train Neural Network Models
```bash
# Train Pipeline A (URL ANN)
python -m training.train_url_ann

# Train Pipeline B (Message RNN)
python -m training.train_message_rnn
```
*Both scripts perform a stratified 70% Train / 15% Val / 15% Test split and export genuine evaluation metrics to `models/`.*

### Step 4: Run Automated Tests
```bash
pytest tests/ -v
```

### Step 5: Start Application Server
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to `http://127.0.0.1:8000` in your web browser.

---

## 5. Docker Deployment

### Run with Docker Compose:
```bash
docker-compose up --build -d
```
The application will be accessible at `http://localhost:8000`.

### Health Check:
```bash
curl http://localhost:8000/health
```

---

## 6. API Reference

### Health Check
- **Endpoint**: `GET /health`
- **Response**:
```json
{
  "status": "healthy",
  "app_name": "PhishGuard-AI",
  "version": "1.0.0",
  "models_status": {
    "url_model_loaded": true,
    "url_scaler_loaded": true,
    "message_model_loaded": true,
    "message_tokenizer_loaded": true
  },
  "timestamp": "2026-09-04T08:50:00.000000+00:00"
}
```

### URL Threat Assessment
- **Endpoint**: `POST /api/v1/predict/url`
- **Request Body**:
```json
{
  "url": "http://paypal-security-update.account-verification-login.com/auth/login.php"
}
```
- **Response**:
```json
{
  "target": "http://paypal-security-update.account-verification-login.com/auth/login.php",
  "scan_type": "url",
  "probability": 0.9612,
  "confidence_score": 96.12,
  "risk_level": "HIGH",
  "verdict": "High Risk - Strong indicators of a phishing or deceptive domain.",
  "indicators": [
    {
      "name": "Insecure HTTP Protocol",
      "description": "The URL does not use TLS/HTTPS encryption.",
      "severity": "WARNING",
      "triggered": true,
      "details": "Scheme: http://"
    },
    {
      "name": "Authentication & Security Keywords",
      "description": "Target contains words associated with high-value security, billing, or login workflows.",
      "severity": "CRITICAL",
      "triggered": true,
      "details": "Detected keywords: login, verify, security, update, account"
    }
  ],
  "recommended_action": "CRITICAL: Do NOT visit this link or interact with the page. Block and report this URL immediately.",
  "analyzed_at": "2026-09-04T08:50:00.000000+00:00",
  "model_version": "1.0.0-ANN",
  "is_fallback": false
}
```

### Message Threat Assessment
- **Endpoint**: `POST /api/v1/predict/message`
- **Request Body**:
```json
{
  "message": "URGENT: Your Chase Bank account has been suspended! Verify your credentials immediately at http://chase-bank-alert.service-notification.top"
}
```

### Model Evaluation Metrics
- **Endpoint**: `GET /api/v1/models/metrics`
- **Response**: Returns authentic accuracy, precision, recall, F1 score, confusion matrices, and split counts stored in `models/url_metrics.json` and `models/message_metrics.json`.

---

## 7. Operational Limitations & Defense in Depth
- **Adversarial Drift**: Attackers continually experiment with novel TLDs, URL shortener chains, and linguistic obfuscation. PhishGuard-AI should be deployed alongside DNS filtering, DMARC/DKIM email enforcement, and user awareness training.
- **Context Boundaries**: Model predictions represent static risk scores. While low-risk outputs indicate compliance with standard patterns, users should always practice standard credential hygiene.
