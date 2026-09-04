import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_endpoints():
    print("--- 1. Testing GET /health ---")
    res = requests.get(f"{BASE_URL}/health")
    print(f"Status: {res.status_code}")
    print(json.dumps(res.json(), indent=2))
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    print("\n--- 2. Testing POST /api/v1/predict/url (Safe Google URL) ---")
    payload_safe = {"url": "https://www.google.com/search?q=cybersecurity+best+practices"}
    res = requests.post(f"{BASE_URL}/api/v1/predict/url", json=payload_safe)
    print(f"Status: {res.status_code}")
    data = res.json()
    print(f"Risk Level: {data['risk_level']}, Probability: {data['probability']}, Verdict: {data['verdict']}")
    assert res.status_code == 200
    assert data["risk_level"] == "LOW"

    print("\n--- 3. Testing POST /api/v1/predict/url (Phishing Target) ---")
    payload_phish = {"url": "http://paypal-security-update.account-verification-login.com/auth/login.php"}
    res = requests.post(f"{BASE_URL}/api/v1/predict/url", json=payload_phish)
    print(f"Status: {res.status_code}")
    data = res.json()
    print(f"Risk Level: {data['risk_level']}, Probability: {data['probability']}, Verdict: {data['verdict']}")
    print("Indicators triggered:")
    for ind in data["indicators"]:
        print(f"  - [{ind['severity']}] {ind['name']}: {ind.get('details')}")
    assert res.status_code == 200
    assert data["risk_level"] == "HIGH"

    print("\n--- 4. Testing POST /api/v1/predict/message (Safe Chat) ---")
    payload_msg_safe = {"message": "Hey, are we still meeting for lunch at 12:30 today?"}
    res = requests.post(f"{BASE_URL}/api/v1/predict/message", json=payload_msg_safe)
    print(f"Status: {res.status_code}")
    data = res.json()
    print(f"Risk Level: {data['risk_level']}, Probability: {data['probability']}, Verdict: {data['verdict']}")
    assert res.status_code == 200
    assert data["risk_level"] == "LOW"

    print("\n--- 5. Testing POST /api/v1/predict/message (Urgent Phishing SMS) ---")
    payload_msg_phish = {
        "message": "URGENT: Your Chase Bank account has been suspended! Verify your credentials immediately at http://chase-bank-alert.service-notification.top"
    }
    res = requests.post(f"{BASE_URL}/api/v1/predict/message", json=payload_msg_phish)
    print(f"Status: {res.status_code}")
    data = res.json()
    print(f"Risk Level: {data['risk_level']}, Probability: {data['probability']}, Verdict: {data['verdict']}")
    print("Indicators triggered:")
    for ind in data["indicators"]:
        print(f"  - [{ind['severity']}] {ind['name']}: {ind.get('details')}")
    assert res.status_code == 200
    assert data["risk_level"] == "HIGH"

    print("\n--- 6. Testing GET /api/v1/models/metrics ---")
    res = requests.get(f"{BASE_URL}/api/v1/models/metrics")
    print(f"Status: {res.status_code}")
    data = res.json()
    print(f"URL Model Acc: {data['url_model']['accuracy']}, F1: {data['url_model']['f1_score']}")
    print(f"Message Model Acc: {data['message_model']['accuracy']}, F1: {data['message_model']['f1_score']}")
    assert res.status_code == 200
    assert data["status"] == "complete"

    print("\n--- 7. Testing GET /api/v1/history ---")
    res = requests.get(f"{BASE_URL}/api/v1/history")
    print(f"Status: {res.status_code}")
    items = res.json()
    print(f"Total audit scan entries recorded: {len(items)}")
    assert len(items) >= 4

    print("\n--- 8. Testing Frontend Root GET / ---")
    res = requests.get(f"{BASE_URL}/")
    print(f"Status: {res.status_code}, Length: {len(res.text)} bytes")
    assert res.status_code == 200
    assert "PhishGuard" in res.text

    print("\n[SUCCESS] All End-to-End API and Model checks passed successfully!")

if __name__ == "__main__":
    test_endpoints()
