import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db

@pytest.fixture(scope="module")
def client():
    init_db()
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["app_name"] == "PhishGuard-AI"
    assert "models_status" in data

def test_predict_url_valid(client):
    payload = {"url": "https://www.google.com/search?q=test"}
    response = client.post("/api/v1/predict/url", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["scan_type"] == "url"
    assert "probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "indicators" in data
    assert len(data["indicators"]) > 0
    assert "recommended_action" in data

def test_predict_url_validation_rejection(client):
    # Empty URL
    res_empty = client.post("/api/v1/predict/url", json={"url": "   "})
    assert res_empty.status_code == 422
    data = res_empty.json()
    assert "error" in data
    assert data["error"] == "Validation Error"

    # Excessively short or invalid control characters
    res_ctrl = client.post("/api/v1/predict/url", json={"url": "http://test\x00.com"})
    assert res_ctrl.status_code == 422

def test_predict_message_valid(client):
    payload = {"message": "Hey friend, let's grab coffee this afternoon at 3."}
    response = client.post("/api/v1/predict/message", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["scan_type"] == "message"
    assert "probability" in data
    assert "risk_level" in data
    assert "verdict" in data

def test_predict_message_phishing(client):
    payload = {"message": "URGENT: Your Chase Bank account has been suspended! Verify credentials at http://phish.xyz"}
    response = client.post("/api/v1/predict/message", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] in ["MEDIUM", "HIGH"]
    assert any(ind["severity"] == "CRITICAL" for ind in data["indicators"])

def test_predict_message_empty(client):
    res_empty = client.post("/api/v1/predict/message", json={"message": ""})
    assert res_empty.status_code == 422

def test_get_metrics(client):
    response = client.get("/api/v1/models/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_get_and_clear_history(client):
    # Perform a scan to ensure history item exists
    client.post("/api/v1/predict/url", json={"url": "https://example.org"})
    res_history = client.get("/api/v1/history")
    assert res_history.status_code == 200
    items = res_history.json()
    assert isinstance(items, list)
    assert len(items) > 0

    # Clear history
    res_clear = client.delete("/api/v1/history")
    assert res_clear.status_code == 200
    res_after = client.get("/api/v1/history")
    assert len(res_after.json()) == 0
